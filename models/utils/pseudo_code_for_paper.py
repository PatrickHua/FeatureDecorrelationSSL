import torch
import torch.nn as nn
from torchvision.models import resnet18


class ShuffledDBN(nn.Module):
    def __init__(self, num_features, num_groups):
        super(ShuffledDBN, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups

    def forward(self, x):
        shuffle_idx = torch.randperm(self.num_features)
        x = x.t()[shuffle_idx].t()
        x = x.view(-1, self.num_groups, self.num_features//self.num_groups)
        x = (x - x.mean(dim=0, keepdim=True)).transpose(0,1)
        covs = x.transpose(1,2).bmm(x) / x.shape[0]
        S, U = covs.symeig(eigenvectors=True)
        W = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1,2))
        x = x.bmm(W)
        return x.transpose(1,2).flatten(0,1)[torch.argsort(shuffle_idx)].t()

# class MLP(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()

#         self.layer1 = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Linear(hidden_dim, out_dim),
#             ShuffledDBN(out_dim)
#         )
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         return x 

class Model(nn.Module):
    def __init__(self, backbone=resnet18(), in_dim=512, hidden_dim=1024, out_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            backbone,
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            ShuffledDBN(out_dim)
        )
    def forward(self, x):
        return self.encoder(x)

f = Model()
for x in loader:
    x1, x2 = aug(x), aug(x)
    z1, z2 = f(x1), f(x2)
    L = (z1 - z2).norm(dim=1, p=2).pow(2).mean()
    L.backward()
    update(f)

if __name__ == '__main__':
    sdbn = ShuffledDBN(200, 4)
    x = torch.randn((256, 200))
    y = sdbn(x)

    print(y.shape)







