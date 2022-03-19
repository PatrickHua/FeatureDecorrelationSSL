import copy
import random 
import torch 
from torch import nn 
import torch.nn.functional as F 
from torchvision import transforms 
from math import pi, cos 
from collections import OrderedDict
from .utils.shuffled_group_whitening import ShuffledGroupWhitening
HPS = dict(
    max_steps=int(1000. * 1281167 / 4096), # 1000 epochs * 1281167 samples / batch size = 100 epochs * N of step/epoch
    # = total_epochs * len(dataloader) 
    mlp_hidden_size=4096,
    projection_size=256,
    base_target_ema=4e-3,
    optimizer_config=dict(
        optimizer_name='lars', 
        beta=0.9, 
        trust_coef=1e-3, 
        weight_decay=1.5e-6,
        exclude_bias_from_adaption=True),
    learning_rate_schedule=dict(
        base_learning_rate=0.2,
        warmup_steps=int(10.0 * 1281167 / 4096), # 10 epochs * N of steps/epoch = 10 epochs * len(dataloader)
        anneal_schedule='cosine'),
    batchnorm_kwargs=dict(
        decay_rate=0.9,
        eps=1e-5), 
    seed=1337,
)

# def loss_fn(x, y, version='simplified'):
    
#     if version == 'original':
#         y = y.detach()
#         x = F.normalize(x, dim=-1, p=2)
#         y = F.normalize(y, dim=-1, p=2)
#         return (2 - 2 * (x * y).sum(dim=-1)).mean()
#     elif version == 'simplified':
#         return (2 - 2 * F.cosine_similarity(x,y.detach(), dim=-1)).mean()
#     else:
#         raise NotImplementedError

from .simsiam import D  # a bit different but it's essentially the same thing: neg cosine sim & stop gradient


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=HPS['mlp_hidden_size'], out_dim=HPS['projection_size'], norm_layer=None):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=HPS['batchnorm_kwargs']['eps'], momentum=1-HPS['batchnorm_kwargs']['decay_rate']),
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

class BYOL(nn.Module):
    def __init__(self, backbone, projector, get_feature=False):
        super().__init__()

        self.backbone = backbone
        self.projector = MLP(**projector)

        self.online_encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.get_feature = get_feature
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(**projector)
        # raise NotImplementedError('Please put update_moving_average to training')

    def target_ema(self, k, K, base_ema=HPS['base_target_ema']):
        # tau_base = 0.996 
        # base_ema = 1 - tau_base = 0.996 
        return 1 - base_ema * (cos(pi*k/K)+1)/2 
        # return 1 - (1-self.tau_base) * (cos(pi*k/K)+1)/2 

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
            
    def forward(self, x1, x2, labels=None):
        f_o, h_o = self.online_encoder, self.online_predictor
        f_t      = self.target_encoder

        if self.get_feature:
            f1_o = self.backbone(x1)
            z1_o = self.projector(f1_o)

            f2_o = self.backbone(x2)
            z2_o = self.projector(f2_o)
            
            # z1_o = f_o(x1)
            # z2_o = f_o(x2)
            # f1 = self.backbone(x1)
            # f2 = self.backbone(x2)
            # z1 = self.projector(f1)
            # z2 = self.projector(f2)
        else:


            z1_o = f_o(x1)
            z2_o = f_o(x2)

        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)

        with torch.no_grad():
            z1_t = f_t(x1)
            z2_t = f_t(x2)
        
        L = D(p1_o, z2_t) / 2 + D(p2_o, z1_t) / 2 

        if self.get_feature:
            return {"loss":L, 'corr':torch.tensor(0), 'rank':torch.tensor(0),  'feature1':[f1_o.detach(), f2_o.detach()], 'feature2':[z1_o.detach(), z2_o.detach()]}
        else:
            return {"loss":L, 'corr':torch.tensor(0), 'rank':torch.tensor(0)}

    

if __name__ == "__main__":
    pass  