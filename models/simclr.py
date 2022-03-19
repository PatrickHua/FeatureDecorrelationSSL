import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from .utils.InfoNCE import NT_XentLoss
from .utils.shuffled_group_whitening import ShuffledGroupWhitening

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=256, num_layers=2, norm_layer=None):
        super().__init__()
        assert num_layers == 2
        if hidden_dim is None:
            hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        
        if norm_layer is None:
            self.norm = nn.Identity()
        elif norm_layer == 'bn':
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm_layer == 'dbn':
            self.norm = ShuffledGroupWhitening(out_dim, num_groups=None, shuffle=False)
        elif norm_layer == 'sdbn':
            self.norm = ShuffledGroupWhitening(out_dim, num_groups=None, shuffle=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.norm(x)

class SimCLR(nn.Module):

    def __init__(self, backbone=resnet50(), projector=None, get_feature=False):
        super().__init__()
        
        self.backbone = backbone
        self.get_feature = get_feature
        # if self.get_feature:
        #     self.projector = nn.Linear(projector['in_dim'], projector['hidden_dim'])
        #     self.predictor = nn.Linear(projector['hidden_dim'], projector['out_dim'])
        # else:
        self.projector = projection_MLP(**projector)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )



    def forward(self, x1, x2, labels=None):


        
        if self.get_feature:
            f1 = self.backbone(x1)
            f2 = self.backbone(x2)
            z1 = self.projector(f1)
            z2 = self.projector(f2)

            loss = NT_XentLoss(z1, z2)

            return {'loss':loss, 'rank':torch.tensor(0), 'corr':torch.tensor(0), 'feature1':[f1.detach(), f2.detach()], 'feature2':[z1.detach(), z2.detach()]}
        
        else:
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            loss = NT_XentLoss(z1, z2)
            return {'loss':loss, 'rank':torch.tensor(0), 'corr':torch.tensor(0)}




















