import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main_worker(device, args, model=None):
    if device == 0 or device is None:
        logging.basicConfig(filename=os.path.join(args.log_dir, 'eval.log'), filemode='a+', level=logging.INFO)
        logger = logging.getLogger(__name__)


    if args.distributed:
        setup(device, args.world_size)
    train_set = get_dataset(
        transform=get_aug(train=False, train_classifier=True, **args.augmentations), 
        train=True,
        split='train' if args.dataset['name'] == 'stl10' else None,
        **args.dataset
    )

    train_sampler = DistributedSampler(train_set) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        
        **args.dataloader
    )

    test_set = get_dataset( 
        transform=get_aug(train=False, train_classifier=False, **args.augmentations), 
        train=False,
        split='test' if args.dataset['name'] == 'stl10' else None,
        **args.dataset)
    test_sampler = DistributedSampler(test_set, shuffle=False) if args.distributed else None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        **args.dataloader
    )
    if model is None:
        model = get_model(**args.model)
        assert args.eval_from is not None
        save_dict = torch.load(args.eval_from, map_location='cpu')
        # state_dict = save_dict['state_dict']
        # for k in list(state_dict.keys()):
        #     if k.startswith('encoder.0.'):
        #         state_dict[k[len('encoder.0.'):]] = state_dict[k]
        #     del state_dict[k]
        # breakpoint()
        # import pdb
        # pdb.set_trace()
        # msg = model.backbone.load_state_dict(state_dict, strict=False)
        msg = model.load_state_dict(save_dict['state_dict'], strict=False)
        print(msg)
        model = model.backbone
    # print(len(train_loader.dataset.classes))
    classifier = nn.Linear(in_features=model.output_dim, out_features=len(train_loader.dataset.classes), bias=True).to(device)


    model = model.to(device)
    # 
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            find_unused_parameters=True
        )
        classifier = nn.parallel.DistributedDataParallel(
            classifier,
            device_ids=[device],
            find_unused_parameters=True
        )
    else:
        model = torch.nn.DataParallel(model)
        classifier = torch.nn.DataParallel(classifier)

    # define optimizer
    optimizer = get_optimizer(
        classifier,
        **args.optimizer
    )
    # define lr scheduler

    lr_scheduler = LR_Scheduler(
        optimizer,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        iter_per_epoch=len(train_loader),
        world_size=args.world_size,
        **args.lr_scheduler
    )

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    # Start training
    global_progress = tqdm(range(0, args.num_epochs), desc=f'Evaluating', ncols=0)
    for epoch in global_progress:
        if args.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', disable=True)
        
        for idx, (images, labels) in enumerate(local_progress):
            # print(images.shape, labels.shape)
            # print(labels)
            classifier.zero_grad()
            with torch.no_grad():
                feature = model(images.to(device))

            preds = classifier(feature)

            loss = F.cross_entropy(preds, labels.to(device))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            local_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg})

    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model(images.to(device))
            
            preds = classifier(feature).argmax(dim=1)
            correct += (preds == labels.to(device)).sum().item()
            total += preds.shape[0]
    print(f'Accuracy = {(correct/total)*100:.2f}')
    logger.info(f'Accuracy = {(correct/total)*100:.2f}')
    if args.distributed:
        cleanup()


def main(args, model=None):

    if args.world_size > 1:
        vars(args)['distributed'] = True
        mp.spawn(main_worker, args=(args, model,), nprocs=args.world_size, join=True)
    else:
        vars(args)['distributed'] = False
        if args.world_size == 1:
            # breakpoint()
            main_worker(0, args, model)
        else:
            main_worker(None, args, model)
if __name__ == "__main__":
    args = get_args()
    vars(args)['world_size'] = torch.cuda.device_count()
    # main(device=0, args=args)
    main(args=args)
















