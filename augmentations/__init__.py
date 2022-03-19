from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
def get_aug(name='simsiam', image_size=224, num_channels=3, train=True, train_classifier=None):
    

    if num_channels == 1:
        mean_std = [[0],[1]]
    elif num_channels == 3:
        mean_std = imagenet_mean_std
    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size, mean_std=mean_std)
        elif name == 'covnorm':
            augmentation = SimSiamTransform(image_size, mean_std=mean_std)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size, normalize=mean_std)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size, mean_std=mean_std)
        elif name == 'wmse':
            augmentation = SimSiamTransform(image_size, mean_std=mean_std)
        elif name == 'cstprop':
            # augmentation = SimSiamTransform(image_size, mean_std=[[0],[1]])
            augmentation = SimSiamTransform(image_size, mean_std=mean_std)
        elif name == 'mnist':
            augmentation = SimSiamTransform(image_size, mean_std=mean_std)
        else:
            # print(name)
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        if name == 'cstprop':
            # augmentation = Transform_single(image_size, train=train_classifier, normalize=[[0],[1]])
            augmentation = Transform_single(image_size, train=train_classifier, normalize=mean_std)
        else:
            augmentation = Transform_single(image_size, train=train_classifier, normalize=mean_std)
    else:
        raise Exception
    
    return augmentation








