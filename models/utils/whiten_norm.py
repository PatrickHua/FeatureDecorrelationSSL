import torch
import torch.nn as nn
from scipy.linalg import solve_triangular
import numpy as np

class Whitening1d(nn.Module):
    def __init__(self, num_features, momentum=0.01, eps=1e-5):
        super(Whitening1d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_covariance", torch.eye(self.num_features))
        self.eps = eps
    def forward(self, x, numpy=False):
        if self.training:
            mean = x.mean(dim=0)
            x = x - mean
            cov = x.t().matmul(x) / (x.size(0) - 1)
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_covariance = self.momentum * cov + (1 - self.momentum) * self.running_covariance
        else:
            mean = self.running_mean
            cov = self.running_covariance
            x = x - mean

        cov = (1 - self.eps) * cov + self.eps * torch.eye(self.num_features).to(cov)
        if numpy:

            I = torch.eye(x.size(1)).to(cov).detach().cpu().numpy()
            cv = np.linalg.cholesky(cov.detach().cpu().numpy())
            whitening_transform = solve_triangular(cv, I, lower=True).T
            
            whitening_transform = torch.tensor(whitening_transform).to(x)
        else:
            I = torch.eye(x.size(1)).to(cov).cpu()
            C = torch.cholesky(cov.cpu())
            whitening_transform = torch.triangular_solve(I, C, upper=False)[0].t().to(x.device)
        return x.matmul(whitening_transform)


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



if __name__ == "__main__":
    from time import time
    num_features = 2048
    batch_size = 4
    # assert batch_size >= num_features
    wn = Whitening1d(num_features)
    bn = torch.nn.BatchNorm1d(num_features)
    torch.manual_seed(0)
    x = torch.randn((batch_size, num_features), requires_grad=True)
    tic = time()
    y = wn(x, numpy=False)
    y.mean().backward()
    toc = time()
    print(toc - tic)
    # print()

    tic = time()
    y = whiten_tensor(x)
    y.mean().backward()
    toc = time()
    print(toc - tic)

    tic = time()
    y = bn(x)
    y.mean().backward()
    toc = time()
    print(toc - tic)
    exit()

    from InfoNCE import NT_XentLoss
    tic = time()
    NT_XentLoss(x, x)
    toc = time()
    print(toc - tic)


