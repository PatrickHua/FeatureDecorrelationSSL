import torch
import torch.nn as nn

import numpy as np

def get_corrcoef(x):
    if type(x) is torch.Tensor:
        x = x.detach().cpu().numpy()
    corr_mat = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(corr_mat, 0)
    return np.abs(corr_mat).mean()




class DecorrelatedReNorm(nn.Module):
    def __init__(self, num_features, momentum=0.01):
        super(DecorrelatedReNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_W", torch.eye(self.num_features))
        

    def forward(self, X):
        X_mean = X.mean(dim=0, keepdim=True)
        X_c = X - X_mean
        eigenvalues, eigenvectors = torch.symeig(X_c.t().matmul(X_c).cpu(), eigenvectors=True, upper=True)
        S, U = eigenvalues.to(X.device), eigenvectors.to(X.device)
        self.eig = eigenvalues.min()
        U_t = U.t()
        W = U.matmul(S.rsqrt().diag()).matmul(U_t)
        # with torch.no_grad:
        W_inv = U.detach().matmul(S.detach().sqrt().diag()).matmul(U_t.detach())
        # breakpoint()
        X = ( X_c.matmul(W).matmul(W_inv) + X_mean.detach() - self.running_mean ).matmul(self.running_W)
        self.running_mean = self.momentum * X_mean.detach() + (1 - self.momentum) * self.running_mean
        self.running_W = self.momentum * W.detach() + (1 - self.momentum) * self.running_W
        return X

if __name__ == "__main__":

    batch_size = 512
    num_features = 512

    drn = DecorrelatedReNorm(num_features)
    x = torch.randn((batch_size*2, num_features), requires_grad=True)
    y = drn(x)
    # print(get_corrcoef(x))
    
    # '''
    # # y = torch.cat([), ])
    # # print(y)
    # dn(x[:batch_size])
    # y = dn(x[batch_size:])
    
    # print(get_corrcoef(y))
    # # print(dn.memory)
    # '''

    # # dn = DecorrelatedNorm(num_features, memory_size=None)
    # # dn = torch.nn.BatchNorm1d(num_features)
    
    # y = dn(x)
    # print(y)
    # print(get_corrcoef(y))
    # y.mean().backward()
    # x = torch.randn((batch_size*2, num_features), requires_grad=True)
    # # dn.zero_grad()
    # y = dn(x)
    # y.mean().backward()
    # # print(dn.memory)





