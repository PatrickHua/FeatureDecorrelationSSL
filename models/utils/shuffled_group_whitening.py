import torch
import torch.nn as nn
import math

# try:
#     torch.randn(2,2).to('cuda').svd()
#     linalg_device = 'cuda'
# except Exception:
linalg_device = 'cpu'

class ShuffledGroupWhitening(nn.Module):
    def __init__(self, num_features, num_groups=None, shuffle=True, engine='symeig'):
        super(ShuffledGroupWhitening, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        if self.num_groups is not None:
            assert self.num_features % self.num_groups == 0
        # self.momentum = momentum
        self.register_buffer('running_mean', None)
        self.register_buffer('running_covariance', None)
        self.shuffle = shuffle if self.num_groups != 1 else False
        self.engine = engine

    def forward(self, x):
        N, D = x.shape
        if self.num_groups is None:
            G = math.ceil(2*D/N) # automatic, the grouped dimension 'D/G' should be half of the batch size N
            # print(G, D, N)
        else:
            G = self.num_groups
        if self.shuffle:
            new_idx = torch.randperm(D)
            x = x.t()[new_idx].t()
        x = x.view(N, G, D//G)
        x = (x - x.mean(dim=0, keepdim=True)).transpose(0,1) # G, N, D//G
        # covs = x.transpose(1,2).bmm(x) / (N-1) #  G, D//G, N @ G, N, D//G -> G, D//G, D//G
        covs = x.transpose(1,2).bmm(x) / N
        W = transformation(covs, x.device, engine=self.engine)
        x = x.bmm(W)
        if self.shuffle:
            return x.transpose(1,2).flatten(0,1)[torch.argsort(new_idx)].t()
        else:
            return x.transpose(0,1).flatten(1)

def transformation(covs, device, engine='symeig'):
    covs = covs.to(linalg_device)
    if engine == 'cholesky':
        C = torch.cholesky(covs.to(linalg_device))
        W = torch.triangular_solve(torch.eye(C.size(-1)).expand_as(C).to(C), C, upper=False)[0].transpose(1,2).to(x.device)
    else:
        if engine == 'symeig':
            
            S, U = torch.symeig(covs.to(linalg_device), eigenvectors=True, upper=True)
        elif engine == 'svd':
            U, S, _ = torch.svd(covs.to(linalg_device))
        elif engine == 'svd_lowrank':
            U, S, _ = torch.svd_lowrank(covs.to(linalg_device))
        elif engine == 'pca_lowrank':
            U, S, _ = torch.pca_lowrank(covs.to(linalg_device), center=False)
        S, U = S.to(device), U.to(device)
        W = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1,2))
    return W

if __name__ == '__main__':
    num_features = 2048
    batch_size = 1024


    from time import time

    engine = 'symeig'
    tic = time()
    x = torch.randn((batch_size, num_features), requires_grad=True)
    gw = ShuffledGroupWhitening(num_features, num_groups=32, shuffle=True, engine=engine)
    y = gw(x).mean().backward()
    toc = time()
    print(engine)
    print(toc - tic)


    engine = 'cholesky'
    tic = time()
    x = torch.randn((batch_size, num_features), requires_grad=True)
    gw = ShuffledGroupWhitening(num_features, num_groups=32, shuffle=True, engine=engine)
    y = gw(x).mean().backward()
    toc = time()
    print(engine)
    print(toc - tic)

    engine = 'svd'
    tic = time()
    x = torch.randn((batch_size, num_features), requires_grad=True)
    gw = ShuffledGroupWhitening(num_features, num_groups=32, shuffle=True, engine=engine)
    y = gw(x).mean().backward()
    toc = time()
    print(engine)
    print(toc - tic)

    engine = 'svd_lowrank'
    tic = time()
    x = torch.randn((batch_size, num_features), requires_grad=True)
    gw = ShuffledGroupWhitening(num_features, num_groups=32, shuffle=True, engine=engine)
    y = gw(x).mean().backward()
    toc = time()
    print(engine)
    print(toc - tic)

    engine = 'pca_lowrank'
    tic = time()
    x = torch.randn((batch_size, num_features), requires_grad=True)
    gw = ShuffledGroupWhitening(num_features, num_groups=32, shuffle=True, engine=engine)
    y = gw(x).mean().backward()
    toc = time()
    print(engine)
    print(toc - tic)


    exit()







    from time import time

    import matplotlib.pyplot as plt

    tmp =  [1, 2, 4, 8, 16, 32, 64, 128, 256]
    out = []
    for i in tmp:

        tic = time()
        y = gw(x)
        toc = time()
        # print(toc - tic)
        t = toc - tic
        print(t)
        out.append(t)







    # bn = torch.nn.BatchNorm1d(num_features)
    # x = torch.randn((batch_size, num_features))
    # tic = time()
    # y = bn(x)
    # toc = time()
    # print(t:=toc - tic)

    # plt.plot(tmp, out)
    # plt.show()




