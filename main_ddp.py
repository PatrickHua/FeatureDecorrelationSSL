import os, sys
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(device, args):

    setup(device, args.world_size)
    train_set = get_dataset(
        transform=get_aug(train=True, **args.augmentations), 
        train=True,
        **args.dataset
    )
    sampler = DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        **args.dataloader
    )

    model = get_model(**args.model).to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        find_unused_parameters=True
    )

    optimizer = get_optimizer(
        model,
        **args.optimizer
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        iter_per_epoch=len(train_loader),
        world_size=args.world_size,
        **args.lr_scheduler
    )

    logger = Logger(log_dir=args.log_dir, **args.logger)

    accuracy = 0 
    # Start training
    global_progress = tqdm(range(0, args.stop_at_epoch), desc=f'Training', disable=(device!=0)) 
    loss_meter = AverageMeter('loss')
    for epoch in global_progress:
        model.train()
        loss_meter.reset()
        sampler.set_epoch(epoch)
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', disable=args.hide_progress or (device!=0))
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            # breakpoint()
            model.zero_grad()
            # print(device)
            images1 = images1.to(device, non_blocking=True)
            images2 = images2.to(device, non_blocking=True)
            data_dict = model.forward(images1, images2, labels=labels.to(device))
            loss = data_dict['loss'] #.mean() # ddp
            loss.backward()
            optimizer.step()
            
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            loss_meter.update(loss.item())
            data_dict.update({'loss_avg': loss_meter.avg})
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.item()
            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)
        
        epoch_dict = {"epoch":epoch}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
    
        # Save checkpoint
        if not args.no_save and device == 0:
            model_path = os.path.join(args.ckpt_dir, f"{args.name}_epoch_{epoch+1}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.module.state_dict(),
                'logger': logger
            }, model_path)
            print(f"Model saved to {model_path}")
            with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'a+') as f:
                f.write(f'{model_path}')
            args.eval_from = model_path

    if args.eval is not False and device == 0:
        for key, value in args.eval.items():
            vars(args)[key] = value

        # linear_eval(device, args, model=model.module.backbone)
        # mp.spawn(linear_eval, args=(args, model.module.backbone,), nprocs=args.world_size, join=True)
        linear_eval(args, model.module.backbone)
    cleanup()



def main(main_worker, args):
    mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)





if __name__ == "__main__":
    args = get_args()
    vars(args)['world_size'] = torch.cuda.device_count()

    main(main_worker, args=args)
    # mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














