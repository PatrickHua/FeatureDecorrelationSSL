import torch
import torch.nn as nn

import numpy as np

def get_corrcoef(x):
    if type(x) is torch.Tensor:
        x = x.detach().cpu().numpy()
    corr_mat = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(corr_mat, 0)
    return np.abs(corr_mat).mean()




class MemoryWhitening1d(nn.Module):
    def __init__(self, num_features, shuffle=False, momentum=0.9):
        super(MemoryWhitening1d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        # self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer('running_mean', None)
        # self.register_buffer("running_covariance", torch.eye(self.num_features))
        self.register_buffer('running_covariance', None)
        self.x_last_batch = None
        self.shuffle = shuffle
        
    def forward(self, x):
        N = x.shape[0]
        # if self.x_last_batch is None:
        #     self.x_last_batch = torch.randn_like(x)
        # x, self.x_last_batch = torch.cat([x, self.x_last_batch]), x.detach()
        mean = x.mean(dim=0)
        if self.running_mean is None:
            self.running_mean = mean
        else:
            mean = self.running_mean = (1. - self.momentum) * self.running_mean.detach() + self.momentum * mean

        x = x - mean
        # import pdb
        # pdb.set_trace()
        cov = x.t().matmul(x) / (x.size(0) - 1)
        if self.running_covariance is None:
            self.running_covariance = cov
        else:
            cov = self.running_covariance = (1 - self.momentum) * self.running_covariance.detach() + self.momentum * cov
        eigenvalues, eigenvectors = torch.symeig(cov.cpu(), eigenvectors=True, upper=True)
        S, U = eigenvalues.to(x.device), eigenvectors.to(x.device)
        self.eig = eigenvalues.min()
        whitening_transform = U.matmul(S.rsqrt().diag()).matmul(U.t())

        return x.matmul(whitening_transform)

if __name__ == "__main__":

    batch_size = 512
    num_features = 512

    dn = DecorrelatedNorm(num_features, memory_size=None)
    x = torch.randn((batch_size*2, num_features), requires_grad=True)
    print(get_corrcoef(x))
    
    '''
    # y = torch.cat([), ])
    # print(y)
    dn(x[:batch_size])
    y = dn(x[batch_size:])
    
    print(get_corrcoef(y))
    # print(dn.memory)
    '''

    # dn = DecorrelatedNorm(num_features, memory_size=None)
    # dn = torch.nn.BatchNorm1d(num_features)
    
    y = dn(x)
    print(y)
    print(get_corrcoef(y))
    y.mean().backward()
    x = torch.randn((batch_size*2, num_features), requires_grad=True)
    # dn.zero_grad()
    y = dn(x)
    y.mean().backward()
    # print(dn.memory)





