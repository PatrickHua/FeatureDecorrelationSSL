import argparse
import os
import torch

import numpy as np
import torch
import random

import re 
import yaml

import shutil
import warnings

from datetime import datetime


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            # assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            # if isinstance(value, dict):
            #     self.__dict__[key] = Namespace(value)
            # else:
            self.__dict__[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=32)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--no_backward', action='store_true')
    args = parser.parse_args()


    with open(args.config_file, 'r') as f:
        for key, value in yaml.load(f, Loader=yaml.FullLoader).items():
            vars(args)[key] = value

    # args.augmentations.update(
    #     name=args.model['name']
    # )

    args.dataset.update(
        data_dir=args.data_dir,
        download=args.download,
        debug_subset_size=args.debug_subset_size if args.debug else None
    )

    if args.debug:
        args.batch_size = 32
        args.num_epochs = 1
        args.stop_at_epoch = 1 
        if args.eval is not None:
            args.eval['batch_size'] = 32
            args.eval['num_epochs'] = 1
            args.eval['stop_at_epoch'] = 1       


    assert not None in [args.log_dir, args.data_dir, args.name]
    if not args.no_save:
        assert args.ckpt_dir is not None
        os.makedirs(args.ckpt_dir, exist_ok=True)
    args.log_dir = os.path.join(args.log_dir, 'in-progress_'+datetime.now().strftime('%m%d%H%M%S_')+args.name)
    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')
    print(f'To view: "watch -n 1 tail {os.path.join(os.path.abspath(args.log_dir), "train.log")}"')
    

    with open(os.path.join(args.log_dir, os.path.basename(args.config_file)), 'w') as f:
        yaml.dump(args.__dict__, f, default_flow_style=False)

    set_deterministic(args.seed)



    # vars(args)['dataloader_kwargs'] = {
    #     'drop_last': True,
    #     'pin_memory': True,
    #     'num_workers': args.dataset['num_workers'],
    # }

    return args
