from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from .covnorm import CovNorm
from .supervised import Supervised
from .barlowtwins import BarlowTwins
from .wmse import WMSE
# from .cstprop import CstProp
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2, AlexNet, LeNet5
from .utils.iterative_normalization import IterNorm
def get_backbone(name, castrate=True):

    if name is None:
        return None
    else:
        backbone = eval(f"{name}")()

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(name, backbone, projector, **model_kwargs):    
    backbone = get_backbone(**backbone)
    if name == 'simsiam':
        model =  SimSiam(backbone, projector, **model_kwargs)
        # if model_cfg.proj_layers is not None:
        # model.projector.set_layers(2)
    elif name == 'covnorm':
        # breakpoint()
        model = CovNorm(backbone, projector,**model_kwargs)
    elif name == 'barlowtwins':
        model = BarlowTwins(backbone, projector, **model_kwargs)
    elif name == 'supervised':
        model = Supervised(backbone, projector, **model_kwargs)
    elif name == 'byol':
        model = BYOL(backbone, projector, **model_kwargs)
    elif name == 'simclr':
        model = SimCLR(backbone, projector=projector, **model_kwargs)
    elif name == 'swav':
        raise NotImplementedError
    elif name == 'wmse':
        model = WMSE(backbone, output_dim=model_cfg.output_dim, normalization=model_cfg.normalization, loss_fn=model_cfg.loss_fn)
    elif name == 'cstprop':
        model = CstProp(backbone, projector)
    else:
        raise NotImplementedError
    return model






