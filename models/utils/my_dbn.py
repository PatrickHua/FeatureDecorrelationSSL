import torch
import torch.nn as nn

import numpy as np

def get_corrcoef(x):
    if type(x) is torch.Tensor:
        x = x.detach().cpu().numpy()
    corr_mat = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(corr_mat, 0)
    return np.abs(corr_mat).mean()




class DecorrelatedNorm(nn.Module):
    def __init__(self, num_features, memory_size=None, momentum=0.01, eps=0, memory_bank=False, mode='svd'):
        super(DecorrelatedNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_covariance", torch.eye(self.num_features))
        self.eps = eps
        self.memory_size = self.num_features*2 if memory_size is None else memory_size
        self.register_buffer("memory", torch.randn(self.memory_size, self.num_features) if memory_bank else None)
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long) if memory_bank else None)
        self.memory_bank = memory_bank
        self.mode = mode

    @torch.no_grad()
    def recall(self, batch):
        N, M, P = batch.size(0), self.memory.size(0), int(self.memory_ptr)
        if N > M:
            print("Hey, your batch size is larger than the memory!")
            raise NotImplementedError
            self.memory = batch[N-M:N].detach()
            self.memory_ptr = 0
        elif N <= M:
            if P + N < M:
                self.memory[P:P+N] = batch.detach()
                self.memory_ptr = torch.tensor(P + N).to(self.memory_ptr)
                idx = list(range(P,P+N))
            elif P < M <= P + N:
                self.memory[P:M] = batch[:M-P].detach()
                self.memory[:P+N-M] = batch[M-P:N].detach()
                self.memory_ptr = torch.tensor(P + N - M).to(self.memory_ptr)
                idx = list(range(P,M)) + list(range(0,P+N-M))
            else:
                raise Exception
        else:
            raise Exception
        rest_idx = list(range(M))
        for i in idx:
            rest_idx.remove(i)
        return torch.cat([batch, self.memory[rest_idx].detach()])

    def forward(self, x):
        N = x.shape[0]
        if self.training:
            if self.memory_bank:
                x = self.recall(x)

            mean = x.mean(dim=0)
            x = x - mean
            cov = x.t().matmul(x) / (x.size(0) - 1)
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_covariance = self.momentum * cov + (1 - self.momentum) * self.running_covariance
        else:
            raise NotImplementedError
            mean = self.running_mean
            cov = self.running_covariance
            x = x - mean


        I = torch.eye(self.num_features).to(cov)
        cov = (1 - self.eps) * cov + self.eps * I

        if self.mode.startswith('svd'): # zca whitening
            if self.mode == 'svd_lowrank':
                U, S, _ = torch.svd_lowrank(cov.cpu())
            elif self.mode == 'svd':
                U, S, _ = cov.cpu().svd()
            U, S = U.to(x.device), S.to(x.device)
            whitening_transform = U.matmul(S.rsqrt().diag()).matmul(U.t())

        elif self.mode == 'cholesky':
            C = torch.cholesky(cov.cpu())
            whitening_transform = torch.triangular_solve(I.cpu(), C, upper=False)[0].t().to(x.device)

        elif self.mode.startswith('pca'):
            if self.mode == 'pca_lowrank':
                U, S, _ = torch.pca_lowrank(cov.cpu(), center=False)
                S, U = S.to(x.device), U.to(x.device)
            elif self.mode == 'pca':
                # eigenvalues, eigenvectors = torch.eig(cov.cpu(), eigenvectors=True) # S: the first element is the real part and the second element is the imaginary part, not necessary ordered
                eigenvalues, eigenvectors = torch.symeig(cov.cpu(), eigenvectors=True, upper=True)
                S, U = eigenvalues.to(x.device), eigenvectors.to(x.device)
                self.eig = eigenvalues.min()
                # breakpoint()
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





