
from decorrelated_batch_norm import DBN
from iterative_normalization import IterNorm
from whiten_norm import Whitening1d
from my_dbn import DecorrelatedNorm
import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time
import matplotlib.pyplot as plt


import torch
import numpy as np

def get_corrcoef(x):
    corr_mat = np.corrcoef(x, rowvar=False)
    print(corr_mat)
    print(np.cov(x, rowvar=False))
    # print(np.cov(x, rowvar=False))
    np.fill_diagonal(corr_mat, 0)
    return np.abs(corr_mat).mean()  


def vanilla_whitening(x):
    mu = x.mean(dim=0)
    m = x.size(0)
    x = x - mu
    sigma = x.t().matmul(x) / (m-1)
    # breakpoint()
    return x.matmul(sigma.inverse().sqrt())
    # return sigma.inverse().sqrt().matmul(x)

def whiten_tensor_svd(X):

    X_c = X - X.mean()
    Sigma = X_c.transpose(0,1).matmul(X_c) / X_c.shape[0]
    Sigma = Sigma.cpu()
    # try:
    U, Lambda, _ = torch.svd(Sigma)
    # except:                     # torch.svd may have convergence issues for GPU and CPU.
    # U, Lambda, _ = torch.svd(Sigma + 1e-4*Sigma.mean()*torch.randn_like(Sigma))
    # U, Lambda, _ = torch.svd()
    U = U.to(X.device)
    Lambda = Lambda.to(X.device)
    W = U.matmul(torch.diag(1.0/torch.sqrt(Lambda + 1e-5)).matmul(U.transpose(0,1)))
    return X_c.matmul(W.transpose(0,1))



def show_scatter(data, size=5, corrcoef=True):

    fig, axes = plt.subplots(nrows=1, ncols=len(data), figsize=(len(data)*size, size))
    if len(data) == 1:
        axes = [axes]

    for i, (ax, (title, x)) in enumerate(zip(axes, data.items())):

        ax.set_aspect('equal', adjustable='box')
        ax.scatter(x[:,0], x[:,1], s=20)
        ax.axis([-5, 5, -5, 5])
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        print(title)
        ax.set_title(f'{title} {get_corrcoef(x):.5f}' if corrcoef else title)

    plt.show()
    plt.close()


    
# batch_size = 3
# feature_dim = 2
# x = torch.rand((batch_size, feature_dim))
# print(get_corrcoef(x))
# w = torch.rand((feature_dim, feature_dim))
# y = x.matmul(w)
# print(get_corrcoef(y))


num_features = 2
batch_size = 20

corr = -0.9
std = 0.5
covs = [[std, std * corr], [std * corr, std]]
means = [1.5, 1.5]

x = torch.from_numpy(np.random.multivariate_normal(means, covs, batch_size)).float()

irn = IterNorm(num_features, num_groups=1, T=5, dim=2, affine=False)
wn = Whitening1d(num_features, eps=0)
bn = nn.BatchNorm1d(num_features, affine=False)
dbn = DBN(num_features, num_channels=1, dim=2, affine=False)
dn_svd = DecorrelatedNorm(num_features, mode='svd')
dn_svd_lrk = DecorrelatedNorm(num_features, mode='svd_lowrank')
dn_cholesky = DecorrelatedNorm(num_features, mode='cholesky')
dn_pca = DecorrelatedNorm(num_features, mode='pca')
dn_pca_lrk = DecorrelatedNorm(num_features, mode='pca_lowrank')
y_it = irn(x)

y_wn = wn(x)


# y_dbn = dbn(x).detach().numpy()

# y_vw = vanilla_whitening(x).detach().numpy()

y_dn_svd = dn_svd(x).detach().numpy()
y_dn_svd_lrk = dn_svd_lrk(x).detach().numpy()
y_dn_pca = dn_pca(x).detach().numpy()
y_dn_pca_lrk = dn_pca_lrk(x).detach().numpy()
y_dn_cholesky = dn_cholesky(x).detach().numpy()

# breakpoint()
show_scatter({'origin':x, 'iter norm':y_it, 'wn':y_wn, 'cholesky':y_dn_cholesky, 'svd': y_dn_svd, 'svd_lowrank': y_dn_svd_lrk, 'pca': y_dn_pca, 'pca_lowrank':y_dn_pca_lrk})
# , 


exit()
num_features = 2048
batch_size = 512

irn = IterNorm(num_features, num_groups=1, affine=False, eps=0)
dbn = DBN(num_features, num_channels=1, dim=2, num_groups=1, affine=False)
bn = nn.BatchNorm1d(num_features, affine=False)


x = torch.randn((batch_size, num_features), requires_grad=True)

tic = time()
y = irn(x).mean().backward()
toc = time()
print(toc - tic)

tic = time()
y = dbn(x) #.mean().backward()
toc = time()
print(toc - tic)

tic = time()
y = bn(x).mean().backward()
toc = time()
print(toc - tic)



















