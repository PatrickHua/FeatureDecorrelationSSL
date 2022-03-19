import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet18
class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        if self.num_layers == 1:
            hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer22 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
            # get_norm(out_dim, **normalization)
        )
        
        
    def forward(self, x):
        # print(x.shape)
        # print(self.layer1)
        if self.num_layers == 3:
            # print("Here")
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 4:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer22(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 1:
            x = self.layer3(x)
        else:
            raise Exception
        return x


class Supervised(nn.Module):
    def __init__(self, backbone, projector, get_feature=False):
        super().__init__()

        # self.supervised = supervised

        self.backbone = backbone
        self.projector = projection_MLP(**projector)
        # self.projector = nn.Linear(projector['in_dim'], projector['hidden_dim'])
        # self.predictor = nn.Linear(projector['hidden_dim'], projector['out_dim'])

        self.get_feature = get_feature

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        # self.loss_fn = loss_fn

    def forward(self, x1, x2=None, labels=None):

        if self.get_feature:
            f1 = self.backbone(x1)
            z1 = self.projector(f1)
            with torch.no_grad():
                f2 = self.backbone(x2)
                z2 = self.projector(f2)

            L = torch.nn.functional.cross_entropy(z1, labels)
            return {'loss':L, 'rank':torch.tensor(0), 'corr':torch.tensor(0), 'feature1':[f1.detach(), f2.detach()], 'feature2':[z1.detach(), z2.detach()]}

        else:

            feature = self.encoder(x1)
            y1 = self.predictor(feature)

            L = torch.nn.functional.cross_entropy(y1, labels)
            return {'loss': L, 'corr':torch.tensor(0), 'rank': torch.tensor(0), 'bias': torch.tensor(0)}






