import torch
import torchvision
from .random_dataset import RandomDataset
import os


def get_dataset(name, data_dir, transform, train=True, download=False, debug_subset_size=None, **kwargs):
    if name == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif name == 'stl10':
        
        dataset = torchvision.datasets.STL10(data_dir, split=kwargs['split'], transform=transform, download=download)
        
        dataset.targets = dataset.labels
        # if kwargs['split'] != 'unlabeled':
            # breakpoint()
        
    elif name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif name == 'imagenet':
        assert download == False
        dataset = torchvision.datasets.ImageNet(os.path.join(data_dir, 'ImageNet'), split='train' if train == True else 'val', transform=transform)
    elif name == 'tinyimagenet':
        assert download == False
        dataset = torchvision.datasets.ImageNet(os.path.join(data_dir, 'tiny-imagenet-200'), split='train' if train == True else 'val', transform=transform)

    elif name == 'random':
        dataset = RandomDataset()
    else:
        raise NotImplementedError
    # if kwargs['split'] == 'test':
    #     breakpoint()
    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        dataset.classes = dataset.dataset.classes
        # try:
        dataset.targets = dataset.dataset.targets
        # except Exception:
        #     print("no targets")
    return dataset