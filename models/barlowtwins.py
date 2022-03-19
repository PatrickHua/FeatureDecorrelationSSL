import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50
# from .utils.InfoNCE import NT_XentLoss
from .utils.correlation import covariance, corrcoef

# class projection_MLP(nn.Module):
#     def __init__(self, in_dim, out_dim=256):
#         super().__init__()
#         hidden_dim = in_dim
#         self.layer1 = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(inplace=True)
#         )
#         self.layer2 = nn.Linear(hidden_dim, out_dim)
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         return x 
class projection_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=2048, num_layers=3):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
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
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            # nn.BatchNorm1d(out_dim, eps=0, affine=False)
        )
        # self.bn = nn.BatchNorm1d(hidden_dim)
        self.num_layers = num_layers
    # def set_layers(self, num_layers):
    #     self.num_layers = num_layers
        from .utils.shuffled_group_whitening import ShuffledGroupWhitening
        self.sdbn = ShuffledGroupWhitening(512)
    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return self.sdbn(x)


def off_diagonal(matrix):
    n = matrix.shape[0]
    return matrix.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)

class BarlowTwins(nn.Module):

    def __init__(self, backbone, projector, get_feature=False, lamb=5e-3):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(**projector)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.lamb = lamb
        self.get_feature = get_feature
    def forward(self, x1, x2, labels=None):
        if self.get_feature:

            f1 = self.backbone(x1)
            f2 = self.backbone(x2)
            z1 = self.projector(f1)
            z2 = self.projector(f2)

        else:
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
        N, D = z1.shape

        z1_norm = (z1 - z1.mean(dim=0, keepdim=True)) / z1.std(dim=0, keepdim=True)
        z2_norm = (z2 - z2.mean(dim=0, keepdim=True)) / z2.std(dim=0, keepdim=True)

        c = torch.mm(z1_norm.t(), z2_norm) / N
        # breakpoint()
        c_diff = (c - torch.eye(D).to(c)).pow(2)
        # print(c_diff[:5,:5])
        c_diff.flatten()[1:].view(D-1, D+1)[:,:-1].mul_(self.lamb)
        # off_diagonal(c_diff)
        # print(c_diff[:5,:5])
        # breakpoint()
        loss = c_diff.sum()
        if self.get_feature:
            return {'loss':loss, 'rank':torch.tensor(0), 'corr':torch.tensor(0),  'feature1':[f1.detach(), f2.detach()], 'feature2':[z1.detach(), z2.detach()]}
        else:
            return {'loss':loss, 'rank':torch.matrix_rank(z1.detach().cpu()), 'corr': corrcoef(z1.detach()).abs().fill_diagonal_(0).sum() / (D*(D-1))}




















