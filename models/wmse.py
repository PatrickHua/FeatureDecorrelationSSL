import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50, resnet18
from .utils.correlation import covariance, corrcoef
from .utils.decorrelated_batch_norm import DBN
from .utils.batch_norm import BatchNorm1d
from .utils.whiten_norm import Whitening1d, whiten_tensor_svd
from .utils.iterative_normalization import IterNorm

def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=64):
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
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class WMSE(nn.Module):
    def __init__(self, backbone=resnet18(), output_dim=64, normalization='DBN', loss_fn='neg_cos_similarity'):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim, out_dim=output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.normalization = normalization

        if self.normalization == 'DBN':
            self.wn = DBN(output_dim, num_channels=1, dim=2, affine=False, eps=0)
        elif self.normalization == 'Whitening1d':
            self.wn = Whitening1d(output_dim, eps=0)
        elif self.normalization == 'IterNorm':
            self.wn = IterNorm(output_dim, num_groups=1, dim=2, affine=False, eps=1e-5)
        elif self.normalization == 'whiten_tensor_svd':
            self.wn = lambda x: whiten_tensor_svd(x)
        else:
            raise NotImplementedError

        if loss_fn == 'neg_cos_similarity':
            self.loss_fn = lambda x1,x2: - F.cosine_similarity(x1, x2, dim=-1).mean()
        elif loss_fn == 'l2_dist':
            self.loss_fn = lambda x1, x2: (x1 - x2).norm(dim=1, p=2).mean()
        elif loss_fn == 'norm_mse_loss':
            self.loss_fn = norm_mse_loss
        else:
            raise NotImplementedError


    def forward(self, x1, x2):

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        corr_in = corrcoef(z1).abs().fill_diagonal_(0).mean()
        # rank_c = torch.matrix_rank(cov.cpu().detach())
        # rank_z_in = torch.matrix_rank(z1.detach().t().cpu())
        rank_c_in = torch.matrix_rank(covariance(z1.detach()).cpu().detach())
        z1 = self.wn(z1)
        z2 = self.wn(z2)
        corr_out = corrcoef(z1).abs().fill_diagonal_(0).mean()
        rank_c_out = torch.matrix_rank(covariance(z1.detach()).cpu().detach())
        # L = - F.cosine_similarity(z1, z2, dim=-1).mean()
        L = self.loss_fn(z1, z2)

        return {'loss': L,
                'corr_in': corr_in,
                'corr_out': corr_out, 'rank_in':rank_c_in, 'rank_out':rank_c_out}






if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469












