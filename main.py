import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check, visualize_matrix
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime
import logging

from models.utils.correlation import covariance #, corrcoef

def corrcoef(x=None, c=None):
    # breakpoint()
    c = covariance(x) if c is None else c
    std = c.diagonal(0).sqrt()
    c /= std[:,None] * std[None,:]
    eps = 1e-5
    return c.clamp(-1+eps, 1-eps)
def correlation(feature):

    corr = corrcoef(feature.detach()).abs()
    D = corr.shape[0]
    corr = corr.fill_diagonal_(0).sum() / (D*(D-1))
    return corr.item()

def uniformity(feature):
    feature = F.normalize(feature, dim=1)
    t=2
    return torch.pdist(feature, p=2)#.pow(2).mul(-t).exp().mean().log().item()

def alignment(f1, f2):
    alpha = 2
    x = F.normalize(f1, dim=1)
    y = F.normalize(f2, dim=1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean().item()
# def meter(feature1, feature2):



def main(device, args):
    logging.basicConfig(filename=os.path.join(args.log_dir, 'train.log'), filemode='a+', level=logging.INFO)
    logger = logging.getLogger(__name__)
    # import pdb
    # pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.augmentations), 
            train=True,
            split='unlabeled' if args.dataset['name'] == 'stl10' else None,
            **args.dataset),
        batch_size=args.batch_size,
        shuffle=True,
        **args.dataloader
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.augmentations), 
            train=True,
            split='train' if args.dataset['name'] == 'stl10' else None,
            **args.dataset),
        batch_size=args.batch_size,
        shuffle=False,
        **args.dataloader
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(train=False, train_classifier=False, **args.augmentations), 
            train=False,
            split='test' if args.dataset['name'] == 'stl10' else None,
            **args.dataset),
        batch_size=args.batch_size,
        shuffle=False,
        **args.dataloader
    )

    model = get_model(**args.model).to(device)
    model = torch.nn.DataParallel(model)

    optimizer = get_optimizer(
        model,
        **args.optimizer
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        iter_per_epoch=len(train_loader),
        **args.lr_scheduler
    )


    accuracy = 0 
    # Start training
    global_progress = tqdm(range(0, args.stop_at_epoch), desc=f'Training', ncols=0)
    loss_meter = AverageMeter('loss')
    rank_meter = AverageMeter('rank')
    corr_meter = AverageMeter('corr')
    std_meter = AverageMeter('std')
    bias_meter = AverageMeter('bias')
    f1_uniformity = AverageMeter('uniform')
    f1_correlation = AverageMeter('f1corr')
    f1_alignment = AverageMeter('align')
    f2_uniformity = AverageMeter('uniform')
    f2_correlation = AverageMeter('f2corr')
    f2_alignment = AverageMeter('align')

    # save_dict
    for epoch in global_progress:
        model.train()
        loss_meter.reset()
        rank_meter.reset()
        corr_meter.reset()
        std_meter.reset()

        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', disable=args.hide_progress, ncols=0)
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            model.zero_grad()

            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True), labels=labels.to(device))
            loss = data_dict['loss'].mean() # ddp
            if args.model.get('get_feature'):
                feature11, feature12 = data_dict.pop('feature1')
                feature21, feature22 = data_dict.pop('feature2')

                f1_uniformity.update(uniformity(feature11))
                f1_correlation.update(correlation(feature11))
                f1_alignment.update(alignment(feature11, feature12))
                f2_uniformity.update(uniformity(feature21))
                f2_correlation.update(correlation(feature21))
                f2_alignment.update(alignment(feature21, feature22))

                # feature2 =

                # exit()





            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if args.model['name'] == 'byol':
                model.module.update_moving_average(*lr_scheduler.byol_stat())
            elif args.model['name'] == 'covnorm':
                model.module.update_dropout_rate(*lr_scheduler.byol_stat())
            data_dict.update({'lr':lr_scheduler.get_lr()})
            loss_meter.update(loss.item())
            rank_meter.update(data_dict['rank'].item())
            corr_meter.update(data_dict['corr'].item())
            std_meter.update(data_dict.get('std', torch.tensor(-1)).item())
            # bias_meter.update(data_dict['bias'].item())
            data_dict.update({'loss': loss.item()})
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.item()

            if data_dict.get('label') is not None:
                label = data_dict.pop('label')
                feature = data_dict.pop('feature')
            local_progress.set_postfix(data_dict)
            # logger.update_scalers(data_dict)
        # if args.model.get('get_feature') is not None:
            # feature_dir = os.path.join(args.log_dir, 'feature_viz')
            # os.makedirs(feature_dir, exist_ok=True)
            # with open(os.path.join(feature_dir, f"epoch{epoch}_iter{idx}"), 'w+') as f:
            #     # feature = data_dict['feature']
            #     # label = data_dict['label']
            #     assert feature.shape[0] == label.shape[0]
            #     for feat, lab in zip(feature, label):
            #         f.write(f'{feat[0]} {feat[1]} {lab}\n')




        out = knn_monitor(model.module.backbone, epoch, memory_loader, test_loader, device, hide_progress=args.hide_progress, **args.knn_monitor)
        accuracy = out['accuracy']
        if args.knn_monitor.get('p_dist', False) == True:
            pdist = out['pdist']
        else:
            pdist = 0 

        epoch_dict = {"epoch":epoch, "accuracy":accuracy, "loss":loss_meter.avg, "F1Unif":f1_uniformity.avg, "F1Corr":f1_correlation.avg, "F2Unif":f2_uniformity.avg, "F2Corr":f2_correlation.avg, 'f1align':f1_alignment.avg, 'f2align':f2_alignment.avg}
        # "rank": rank_meter.avg, "corr": corr_meter.avg, "std":std_meter.avg}
        global_progress.set_postfix(epoch_dict)
        # epoch_dict.update({"mat": mat.detach().cpu().numpy()})
        # logger.update_scalers(epoch_dict)
        logger.info(f'Train: [{epoch}/{args.num_epochs}] Accuracy:{accuracy} Loss:{loss_meter.avg} Rank:{rank_meter.avg} Corr:{corr_meter.avg} Std: {std_meter.avg}\
            F1Unif:{f1_uniformity.avg} F1Corr:{f1_correlation.avg} F2Unif:{f2_uniformity.avg} F2Corr:{f2_correlation.avg}\
                F1Align:{f1_alignment.avg} F2Align:{f2_alignment.avg}')
    
    # Save checkpoint
    if not args.no_save:
        model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.module.state_dict()
        }, model_path)
        print(f"Model saved to {model_path}")
        with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
            f.write(f'{model_path}')
        args.eval_from = model_path
        # logger.save()

    if args.eval is not False:

        for key, value in args.eval.items():
            vars(args)[key] = value

        # linear_eval(args.device, args, model=model.module.backbone)

        vars(args)['world_size'] = 1 if torch.cuda.is_available() else 0
        linear_eval(args, model.module.backbone)

if __name__ == "__main__":
    args = get_args()

    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














