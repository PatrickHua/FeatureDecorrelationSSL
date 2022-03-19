
import torch

def covariance(x):
    x = x - x.mean(dim=0) # It use to be x -= x.mean(dim=0), stupid! this will change the input tensor!!!
    return x.t().matmul(x) / (x.size(0) - 1)

def corrcoef(x=None, c=None):
    # breakpoint()
    c = covariance(x) if c is None else c
    std = c.diagonal(0).sqrt()
    # breakpoint()
    c /= std[:,None] * std[None,:]
    eps = 1e-5
    # eps=0.3
    return c # .clamp(-1+eps, 1-eps)

def mean_std_covariance(x):
    x = (x - x.mean(dim=0)) / x.std(dim=0)
    return covariance(x)


def bn_covariance(x):
    from batch_norm import BatchNorm1d
    bn = BatchNorm1d(x.shape[1], unbiased=False)
    x = bn(x)
    return covariance(x)

if __name__ == "__main__":
    x = torch.randn(5,7)
    print(covariance(x))
    
    print(corrcoef(x))

    print(mean_std_covariance(x))

    # print(bn_covariance(x))



