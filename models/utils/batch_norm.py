import torch
import torch.nn as nn
import math


class BatchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, gamma=False, beta=False, unbiased=False):
        super(BatchNorm1d, self).__init__()

        self.eps = eps
        # self.affine = affine
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum

        self.register_parameter('weight', nn.Parameter(torch.ones(1, num_features)) if gamma else None)
        self.register_parameter('bias', nn.Parameter(torch.zeros(1,num_features)) if beta else None)

        self.register_buffer('running_mean', torch.zeros(1,num_features))
        self.register_buffer('running_std', torch.ones(1,num_features))

        self.unbiased = unbiased
    def forward(self, x):

        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True, unbiased=self.unbiased).clamp(self.eps)
            x = (x - mean) / std
            self.running_mean = self.running_mean + self.momentum * (mean - self.running_mean)
            self.running_std = self.running_std + self.momentum * (std* 1 if self.unbiased else math.sqrt(x.shape[0]/(x.shape[0]-1) ) - self.running_std)
        else:
            x = (x - self.running_mean) / self.running_std

        if self.gamma:
            x = x * self.weight
        if self.beta:
            x = x + self.bias
        return x
        # return x * self.gamma + self.beta if self.affine else x 
