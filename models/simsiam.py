import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
from .utils.correlation import covariance, corrcoef
from .utils.shuffled_group_whitening import ShuffledGroupWhitening


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=2048, num_layers=3, norm_layer='bn'):
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


        # self.bn = nn.BatchNorm1d(hidden_dim)
        self.num_layers = num_layers
    # def set_layers(self, num_layers):
    #     self.num_layers = num_layers
        from .utils.shuffled_group_whitening import ShuffledGroupWhitening
        self.sdbn = ShuffledGroupWhitening(512)
        
        if norm_layer is None:
            self.norm = nn.Identity()
        elif norm_layer == 'bn':
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm_layer == 'dbn':
            self.norm = ShuffledGroupWhitening(out_dim, num_groups=None, shuffle=False)
        elif norm_layer == 'sdbn':
            self.norm = ShuffledGroupWhitening(out_dim, num_groups=None, shuffle=True)

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            # nn.BatchNorm1d(out_dim)
            self.norm,
        )
        
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


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048, norm_layer=None): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            # nn.BatchNorm1d(out_dim)
            # self.norm
        )
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone, projector, predictor, get_feature=False):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(**projector)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP(**predictor)
        self.get_feature = get_feature

    def forward(self, x1, x2, labels=None):
        if self.get_feature:
        
            f1 = self.backbone(x1)
            f2 = self.backbone(x2)
            z1 = self.projector(f1)
            z2 = self.projector(f2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            L = D(p1, z2) + D(p2, z1)
            return {'loss':L, 'corr':torch.tensor(0), 'rank':torch.tensor(0), 'feature1':[f1.detach(), f2.detach()], 'feature2':[z1.detach(), z2.detach()]}
        else:
            f, h = self.encoder, self.predictor
            z1, z2 = f(x1), f(x2)
            p1, p2 = h(z1), h(z2)
            L = D(p1, z2) + D(p2, z1)
            return {"loss":L, 'corr':torch.tensor(0), 'rank':torch.tensor(0)}


        # L_sim = - F.cosine_similarity(p1, p2, dim=-1).mean()
        # L = - F.cosine_similarity(p1, z1.detach(), dim=-1).mean() - F.cosine_similarity(p2, z2.detach(), dim=-1).mean() \
        #     + L_sim
        # L = (p1 - z2.detach()).norm(dim=1, p=2).mean() + (p2 - z1.detach()).norm(dim=1, p=2).mean() 
        # \
        #     + 1000* torch.abs(corrcoef(p1).abs().mean() - corrcoef(z1).abs().mean()) \
        #     + 1000* torch.abs(corrcoef(p2).abs().mean() - corrcoef(z2).abs().mean())

        # L = - F.cosine_similarity(p2, z2.detach(), dim=-1).mean() - F.cosine_similarity(z1, z2.detach(), dim=-1).mean()
        # L = - F.cosine_similarity(p1, z2.detach(), dim=-1).mean()#  - F.cosine_similarity(z1, z2, dim=-1).mean()

        # L = - F.cosine_similarity(p1, z1.detach(), dim=-1).mean() - F.cosine_similarity(p1, p2, dim=-1).mean()
        
        # L_sim = D(p1, z2) / 2 + D(p2, z1) / 2


        # 
        # L = L_sim \
        #     + (covariance(p1) - covariance(z1).detach()).pow(2).clamp(1e-5).log().mean() \
        #     + (covariance(p2) - covariance(z2).detach()).pow(2).clamp(1e-5).log().mean()
        # L = L_sim \
        #     + (covariance(p1) - covariance(z1.detach()).detach()).abs().mean() \
        #     + (covariance(p2) - covariance(z2.detach()).detach()).abs().mean()

        # p1 = F.normalize(p1, dim=1) 
        # z1 = F.normalize(z1, dim=1) 
        # p2 = F.normalize(p2, dim=1) 
        # z2 = F.normalize(z2, dim=1) 
        # L = L_sim \
        #     + (corrcoef(p1) - corrcoef(z1).detach()).pow(2).mean() \
        #     + (corrcoef(p2) - corrcoef(z2).detach()).pow(2).mean()

        # L = L_sim \
        #     + (corrcoef(p1).abs().clamp(1e-5).log().mean() - corrcoef(z1.detach()).abs().clamp(1e-5).log().mean()).pow(2).clamp(1e-5).log() \
        #     + (corrcoef(p2).abs().clamp(1e-5).log().mean() - corrcoef(z2.detach()).abs().clamp(1e-5).log().mean()).pow(2).clamp(1e-5).log()

            # + (corrcoef(p1) - corrcoef(z1).detach()).pow(2).clamp(1e-5).log().mean() \
            # + (corrcoef(p2) - corrcoef(z2).detach()).pow(2).clamp(1e-5).log().mean()
        
        # L = L_sim \
        #     + D(p1, z1) + D(p2, z2)


        # return {'loss': L,
        #         'L_sim': L_sim,
        #         # 'gamma':self.projector.bn.weight.detach().mean(), 
        #         # 'beta': self.projector.bn.bias.detach().mean(),
        #         'z1_c':corrcoef(z1).abs().fill_diagonal_(0).mean(),
        #         'z2_c':corrcoef(z2).abs().fill_diagonal_(0).mean(),
        #         'p1_c':corrcoef(p1).abs().fill_diagonal_(0).mean(),
        #         'p2_c':corrcoef(p2).abs().fill_diagonal_(0).mean(),
        #         'z_mean': z1.mean(),
        #         'p_mean': p1.mean(),
        #         'z_std': z1.std(),
        #         'p_std': p1.std()}






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












