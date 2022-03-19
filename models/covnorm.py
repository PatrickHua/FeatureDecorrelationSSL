import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet18
from .utils.correlation import covariance, corrcoef
from .utils.batch_norm import BatchNorm1d
from .utils.whiten_norm import Whitening1d, whiten_tensor_svd
from .utils.iterative_normalization import IterNorm
from .utils.decorrelated_batch_norm import DBN
from .utils.decorrelated_batch_renorm import DecorrelatedReNorm
from .utils.my_dbn import DecorrelatedNorm
from .utils.memory_wn import MemoryWhitening1d
from .utils.group_whitening import GroupWhitening1d
from .utils.shuffled_group_whitening import ShuffledGroupWhitening
from math import cos, pi

class ShuffledBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, eps=0, affine=False)
    def forward(self, x):
        new_idx = torch.randperm(self.num_features)
        x = x.t()[new_idx].t()
        x = self.bn(x)
        return x.t()[torch.argsort(new_idx)].t()

class ShuffledIterNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.itn = IterNorm(num_features, num_groups=2, dim=2, eps=0, affine=False)
    def forward(self, x):
        new_idx = torch.randperm(self.num_features)
        x = x.t()[new_idx].t()
        x = self.itn(x)
        return x.t()[torch.argsort(new_idx)].t()


class RotatedBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, eps=0, affine=False)
    def forward(self, x):

        N, D = x.shape
        W = 0.1*torch.randn((D, D)).cpu()
        W_inv = torch.inverse(W).to(x)
        # breakpoint()
        x = x.mm(W.to(x))
        x = self.bn(x)
        x = x.mm(W_inv)
        return x


import numpy as np

def get_norm(num_features, name='sdbn', **kwargs):
    if name == 'sdbn':
        return ShuffledGroupWhitening(num_features, **kwargs)
    elif name == 'bn':
        return BatchNorm1d(num_features, **kwargs)
    elif name is None:
        return nn.Identity()
    elif name == 'softmax_bn':
        return nn.Sequential(
            # nn.Dropout(p=0.001),
            nn.Softmax(dim=-1),
            nn.BatchNorm1d(num_features, eps=0, affine=False)
        )

    else:
        raise NotImplementedError

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, normalization, num_layers=2):
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
            nn.Linear(hidden_dim, out_dim),
            get_norm(out_dim, **normalization)
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



class CovNorm(nn.Module):
    def __init__(self, backbone, projector, supervised=False, get_feature=False, bottleneck=False, loss_fn='l22', dropout=0):
        super().__init__()

        self.supervised = supervised
        self.bottleneck = bottleneck

        self.backbone = backbone
        if supervised:
            self.projector = projection_MLP(
                **projector
            )
            # if projector['num_layers'] == 2:
            #     self.projector = nn.Sequential(
            #         nn.Linear(projector['in_dim'], projector['hidden_dim']),
            #         nn.Linear(projector['hidden_dim', projector['out_dim']])
            #     )
            # else:
            #     raise NotImplementedError

            # projection_MLP(
            #     **projector
            # )#nn.Linear(projector['in_dim'], projector['out_dim'])
        elif self.bottleneck:
            self.projector = nn.Linear(projector['in_dim'], projector['hidden_dim'])
            self.predictor = nn.Sequential(
                nn.Linear(projector['hidden_dim'], projector['out_dim']),
                get_norm(projector['out_dim'], **projector['normalization'])
            )
        else:

            self.projector = projection_MLP(
                **projector
            )
        self.get_feature = get_feature
        # if self.get_feature:
            # self.encoder = nn.Sequential
        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.dropout_rate = dropout
        self.current_dropout_rate = dropout
    def update_dropout_rate(self, global_step, max_steps):
        k, K = global_step, max_steps
        self.current_dropout_rate = self.dropout_rate * (cos(pi*k/K) + 1)/2
        # return p
        
        
        
    def forward(self, x1, x2=None, labels=None):

        if self.get_feature:

            f1 = self.backbone(x1)
            f2 = self.backbone(x2)
            z1 = self.projector(f1)
            z2 = self.projector(f2)
            L = (z1 - z2).norm(dim=1, p=2).pow(2).mean()

            return {'loss':L, 'rank':torch.tensor(0), 'corr':torch.tensor(0),  'feature1':[f1.detach(), f2.detach()], 'feature2':[z1.detach(), z2.detach()]}


        if self.supervised:
            y1 = self.encoder(x1)
            L = torch.nn.functional.cross_entropy(y1, labels) #+ torch.nn.functional.cross_entropy(y2, labels)
            return {'loss': L}

        else:
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            if self.bottleneck:
                feature = z1.detach().cpu().numpy()
                label = labels.detach().cpu().numpy()

                z1 = self.predictor(z1)
                z2 = self.predictor(z2)
            # z1, z2 = self.encoder(torch.cat([x1, x2])).chunk(2)
            if self.loss_fn == 'l22':
                L = (z1 - z2).norm(dim=1, p=2).pow(2).mean()
            if self.loss_fn == 'l2':
                L = (z1 - z2).norm(dim=1, p=2).mean()
            elif self.loss_fn == 'l33':
                L = (z1 - z2).norm(dim=1, p=3).pow(3).mean()
            elif self.loss_fn == 'l44':
                L = (z1 - z2).norm(dim=1, p=4).pow(4).mean()
            elif self.loss_fn == 'neg_cos':
                L = - F.cosine_similarity(z1, z2, dim=-1).mean()
            elif self.loss_fn == 'cross_entropy':
                # L = F.cross_entropy(z1, z2.argmax(dim=1).detach()) + F.cross_entropy(z2, z1.argmax(dim=1).detach())
                L = F.cross_entropy(z1, z2.argmax(dim=1)) + F.cross_entropy(z2, z1.argmax(dim=1))
            elif self.loss_fn == 'cross_entropy_drop':
                # z1 = self.dropout(z1)
                # z2 = self.dropout(z2)
                if self.training:
                    z1 = F.dropout(z1, p=self.current_dropout_rate)
                    z2 = F.dropout(z2, p=self.current_dropout_rate)
                    
                L = F.cross_entropy(z1, z2.argmax(dim=1)) + F.cross_entropy(z2, z1.argmax(dim=1))

            elif self.loss_fn == 'cross_entropy2':
                t = 10
                L = F.cross_entropy(z1/t, z2.detach().argmax(dim=1)) + F.cross_entropy(z2/t, z1.detach().argmax(dim=1))
            elif self.loss_fn == 'cross_entropy3':
                mean_label = (z1+z2).argmax(dim=1)
                L = F.cross_entropy(z1, mean_label) + F.cross_entropy(z2, mean_label)
            elif self.loss_fn == 'cross_entropy4':
                L = - F.softmax(z1) * F.log_softmax(z2.detach()) - F.softmax(z2) * F.log_softmax(z1.detach())
            elif self.loss_fn == 'cross_entropy5':
                t = 0.1
                L = - F.softmax(z1) * F.log_softmax(z2.detach()/t) - F.softmax(z2) * F.log_softmax(z1.detach()/t)
            elif self.loss_fn == 'l2_pseudo_label':
                # z1 = F.normalize(z1, dim=-1)
                # z2 = F.normalize(z2, dim=-1)
                # breakpoint()
                N = z1.shape[0]
                # L = - z1[z2.argmax(dim=1)] - z2[z1.argmax(dim=1)]
                L = - z1[range(N),z2.argmax(dim=1)].mean() - z2[range(N), z1.argmax(dim=1)].mean()
            elif self.loss_fn == 'nll':
                L = F.nll_loss(z1, z2.argmax(dim=1)) + F.nll_loss(z2, z1.argmax(dim=1))
            elif self.loss_fn == 'l1_dot_prod':
                eps = 1e-5
                z1 = F.normalize(z1, p=1, dim=-1)
                z2 = F.normalize(z2, p=1, dim=-1)
                dot_prod = (z1 * z2).sum(dim=-1)
                L = - ((dot_prod/2)+1).clamp(eps).log().mean()
                
            elif self.loss_fn == 'l0.5_dot_prod':
                eps = 1e-5
                z1 = z1 / z1.abs().clamp(eps).sqrt().sum(dim=-1, keepdim=True)
                z2 = z2 / z2.abs().clamp(eps).sqrt().sum(dim=-1, keepdim=True)
                dot_prod = (z1 * z2).sum(dim=-1)
                # L = - (z1 * z2).sum(dim=-1).exp().mean()
                L = - ((dot_prod/2)+1).clamp(eps).log().mean()
            elif self.loss_fn == 'soft_margin_loss':
                raise NotImplementedError
            elif self.loss_fn == 'multi_label_margin':
                raise NotImplementedError

        # return {'loss': L}

        corr = corrcoef(z1.detach()).abs()
        D = corr.shape[0]
        corr = corr.fill_diagonal_(0).sum() / (D*(D-1))


        if torch.isnan(corr):
            corr = torch.tensor(-1)

        tol = 1e-1


        # else:
        try:
            rank = torch.matrix_rank(z1.detach().cpu(), tol=tol)
        except Exception:
            rank = torch.tensor(-1)
        try:
            std = z1.detach().std(dim=0).mean()
        except Exception:
            std = torch.tensor(-1)
        return {'loss': L, 'corr':corr, 'rank': rank, 'std':std, 'bias': self.current_dropout_rate}


        # return {'loss': L, 'corr':corr, 'rank': torch.matrix_rank(z1.detach().cpu()), 'bias': torch.tensor(0)} # self.projector.layer3[-1].bias.norm(p=2, dim=1).mean()

#  'mat':corrcoef(z1).abs(), 








# x1 = aug(x)
# x2 = aug(x)

# z1 = encoder(x1)
# z2 = encoder(x2)
# with torch.no_grad():
#     z = encoder(x) #forward

# loss = F.cross_entropy(z1, z.argmax(dim=1)) + F.cross_entropy(z2, z.argmax(dim=1))


# x1 = aug(x)
# x2 = aug(x)

# z1 = encoder(x1)
# z2 = encoder(x2)
# loss = F.cross_entropy(z1, z2.argmax(dim=1)) + F.cross_entropy(z2, z1.argmax(dim=1))


