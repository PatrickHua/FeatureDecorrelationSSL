import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from whiten_norm import Whitening1d, whiten_tensor_svd
from decorrelated_batch_norm import DBN
# from norm_tests import get_corrcoef
def get_corrcoef(x):
    corr_mat = np.corrcoef(x, rowvar=False)
    # print(corr_mat)
    # print(np.cov(x, rowvar=False))
    # print(np.cov(x, rowvar=False))
    np.fill_diagonal(corr_mat, 0)
    return np.abs(corr_mat).mean()  
file = "/Users/tiany/Downloads/input.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)

# wn = Whitening1d(data.shape[1], eps=0)
dbn = DBN(data.shape[1], eps=0, num_channels=1, dim=2, affine=False)
# x = 
# data = torch.rand_like(torch.from_numpy(data)).numpy()
# data = torch.rand((64, 512)).numpy()
# print(np.abs(np.corrcoef(data, rowvar=False)).mean())
print(get_corrcoef(data))
# y = whiten_tensor_svd(torch.from_numpy(data)).numpy()
# y = wn(torch.from_numpy(data)).numpy()
breakpoint()
y = dbn(torch.from_numpy(data)).numpy()

# print(np.abs(np.corrcoef(y, rowvar=False)).mean())
print(get_corrcoef(y))
breakpoint()




x = np.array([
    [-1, 0],
    [1, 0]
])



print(get_corrcoef(x))
print(np.cov(x, rowvar=False))





# class DecorBatchNorm1d(nn.Module):
#     def __init__(self, num_features, num_groups=32, num_channels=0, ndim=2, eps=1e-5, momentum=0.1, gamma=True, beta=True):
#         super(DecorBatchNorm1d, self).__init__()
#         if num_channels > 0:
#             num_groups = num_features // num_channels
#         self.num_features = num_features
#         self.num_groups = num_groups
#         assert self.num_features % self.num_groups == 0
#         self.dim = dim
#         self.eps = eps
#         self.mmomentum = momentum
#         # self.affine = affine
#         self.gamma = gamma
#         self.beta = beta
#         self.mode = mode
#         self.ndim = ndim

#         # if self.affine:
#         #     self.weight = nn.Parameter(torch.Tensor(self.num_features))
#         #     self.bias = nn.Parameter(torch.Tensor(self.num_features))
#         self.register_parameter('weight', nn.Parameter(torch.ones(num_features)) if gamma else None)
#         self.register_parameter('bias', nn.Parameter(torch.zeros(num_features)) if beta else None)

#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_projection', torch.eye(num_features))

#         self.reset_parameter()

#     def reset_parameter(self):
#         if self.gamma: nn.init.ones_(self.weight)
#         if self.beta: nn.init.zeros_(self.bias)
        


#     def forward(self, x):
#         if self.training:
#             mean = x.mean(dim=1, keepdim=True)
#             self.running_mean = (1-self.momentum) * self.running_mean + self.mmomentum * mean
#             x = x - mean
#             cov = x.matmut(x.t()) / x.size(1) + self.eps * torch.eye()
#             u, eig, _ = cov.cpu().svd()
            