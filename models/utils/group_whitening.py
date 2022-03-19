import torch
import torch.nn as nn

import numpy as np

class GroupWhitening1d(nn.Module):
    def __init__(self, num_features, num_groups=4, shuffle=False, momentum=0.9):
        super(GroupWhitening1d, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        self.momentum = momentum
        # self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer('running_mean', None)
        # self.register_buffer("running_covariance", torch.eye(self.num_features))
        self.register_buffer('running_covariance', None)
        self.x_last_batch = None
        self.shuffle = shuffle




    def forward(self, x):
        G, N, D = self.num_groups, *x.shape
        if self.shuffle:
            new_idx = torch.randperm(x.shape[1])
            reverse_shuffle = torch.argsort(new_idx)
            x = x.t()[new_idx].t()
        x = x.view(N, G, D//G)
        x = x - x.mean(dim=0, keepdim=True)
        x = x.transpose(0,1) # G, N, D//G
        covs = x.transpose(1,2).bmm(x) / (x.size(1) - 1) # G, D//G, D//G
        eigenvalues, eigenvectors = torch.symeig(covs.cpu(), eigenvectors=True, upper=True)
        S, U = eigenvalues.to(x.device), eigenvectors.to(x.device)
        self.eig = eigenvalues.min()
        whitening_transform = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1,2))
        x = x.bmm(whitening_transform)
        if self.shuffle:
            return x.transpose(1,2).flatten(0,1)[reverse_shuffle].t()
        else:
            return x.transpose(0,1).flatten(1)





if __name__ == "__main__":
    from time import time
    batch_size = 256
    num_features = 512

    # dn = DecorrelatedNorm(num_features, memory_size=None)
    gw = GroupWhitening1d(num_features, num_groups=32)

    tic = time()
    x = torch.randn((batch_size, num_features), requires_grad=True)
    y = gw(x)
    toc = time()
    print(toc - tic)

    bn = torch.nn.BatchNorm1d(num_features)

    tic = time()
    x = torch.randn((batch_size, num_features), requires_grad=True)
    
    y = bn(x)

    toc = time()
    print(toc - tic)
    exit()



    print(get_corrcoef(y))
    y.mean().backward()
    x = torch.randn((batch_size*2, num_features), requires_grad=True)
    # dn.zero_grad()
    y = dn(x)
    y.mean().backward()
    # print(dn.memory)





