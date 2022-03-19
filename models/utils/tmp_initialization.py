import torch
import torch.nn.functional as F
import torch.nn as nn
import math
# from torch.nn.functional import relu
import numpy as np
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
def covariance(x):
    x = x - x.mean(dim=0) # It use to be x -= x.mean(dim=0), stupid! this will change the input tensor!!!
    return x.t().matmul(x) / (x.size(0) - 1)

def corrcoef(x=None, c=None):
    # breakpoint()
    c = covariance(x) if c is None else c
    # print(c)
    std = c.diagonal(0).sqrt().clamp(1e-5)
    # breakpoint()
    c /= std[:,None] * std[None,:]
    eps = 1e-5
    # eps=0.3
    return c.clamp(-1+eps, 1-eps)

def get_corrcoef(x):
    # if type(x) is torch.Tensor:
    if x.ndim == 4:
        x = x.permute(0,2,3,1).flatten(0,2)
        # x = x.detach().cpu().numpy()
    # breakpoint()
    corr_mat = corrcoef(x)
    # print(corr_mat)
    corr_mat.fill_diagonal_(0)
    # corr_mat = np.corrcoef(x, rowvar=False)
    # np.fill_diagonal(corr_mat, 0)
    return corr_mat.abs().mean().item()





'''
num_features = 2
batch_size = 200

corr = -0
std = 0.5
covs = [[std, std * corr], [std * corr, std]]
means = [1.5, 1.5]

x = torch.from_numpy(np.random.multivariate_normal(means, covs, batch_size)).float()

print(get_corrcoef(x))

# w = torch.randn((num_features, num_features))
theta = 1
w = torch.tensor([
    [math.cos(theta),-math.sin(theta)],
    [math.sin(theta),math.cos(theta)]
]).to(x)




y = x.mm(w)
print(get_corrcoef(y))

exit()
'''


def xavier_normal_(weight):
    weight.data.normal_(mean=0, std=math.sqrt(1/weight.size(1)))

class MLP(nn.Module):
    def __init__(self, sizes=[128]*10):
        super().__init__()
        self.sizes = sizes
        self.num_layers = len(sizes) - 1 
        self.layer_dims = list(zip(sizes[:-1],sizes[1:]))
        self.layers = nn.Sequential(*[self.block(in_features, out_features) for in_features, out_features in self.layer_dims])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight)
                xavier_normal_(m.weight)

                # nn.init.normal(m.weight)
                # U, scale, V = 
                m.weight.data = (lambda U,_,V: U.mm(V.t()))(*m.weight.svd())
                # breakpoint()
                
                # U, scale, V = m.weight.svd()
                # m.weight.data = U.mm((torch.zeros_like(scale).fill_(scale.mean())).diag()).mm(V.t())
                # m.weight.data = U.mm(torch.ones_like(scale).diag()).mm(V.t())

                # m.weight.data = U.mm(V.t())
                pass
                # m.weight.data = U.mm(torch.ones_like(scale).diag()).mm(V.t())
                # m.weight.data = (m.weight.data - m.weight.data.mean()) / m.weight.data.std()
                # print(m.weight.data.std().item())
                # for i in range(100):
                #     U, scale, V = m.weight.svd()
                #     m.weight.data = U.mm((torch.zeros_like(scale).fill_(scale.mean())).diag()).mm(V.t())
                #     print(m.weight.data.std().item(), scale.max().item(), scale.min().item(), scale.mean().item(), scale.std().item())
                # breakpoint()


        def operate(self, input, output):
            # print(input[0].shape)
            # print(input[0].mean(), input[0].std())
            print(get_corrcoef(input[0]))#, input[0][0].std().item())

        for layer in self.layers:
            layer.register_forward_hook(operate)

    def block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.ReLU(inplace=False),
        )

    
    def forward(self, x):
        x = self.layers(x)
        return x



batch_size = 256
feature_dim = 128*2
model = MLP(sizes=[feature_dim]*20)
x = torch.randn((batch_size, feature_dim))
# breakpoint()
y = model(x)
# breakpoint()














exit()



# resnet50.fc = torch.nn.Identity()

model = resnet50(num_classes=1000)
model = IntermediateLayerGetter(model, return_layers={"relu":0, "layer1":1, "layer2":2, "layer3":3, "layer4":4})
batch_size = 128
image_size = 64
num_channels = 3
x = torch.randn((batch_size, num_channels, image_size, image_size))
y = model(x)
print(get_corrcoef(x))
for i, feature in y.items():
    print(i)
    print(feature.mean().item(), feature.var().item())
    print(get_corrcoef(feature))
# breakpoint()
# print(get_corrcoef(y))


exit()
x_valid = torch.randn(2000, 784)
# random init
w1 = torch.randn(784, 50) * math.sqrt(2/784)
b1 = torch.randn(50) 
w2 = torch.randn(50, 10) * math.sqrt(2/50)
b2 = torch.randn(10)
w3 = torch.randn(10, 2) * math.sqrt(2/10)
b3 = torch.randn(2)
def linear(x, w, b):
    return x@w + b
def relu(x):
    return x.clamp_min(0.)
t1 = relu(linear(x_valid, w1, b1))
t2 = relu(linear(t1, w2, b2))
t3 = relu(linear(t2, w3, b3))
# breakpoint()
print(t1.mean(), t1.std(), get_corrcoef(t1))
print(t2.mean(), t2.std(), get_corrcoef(t2))
print(t3.mean(), t3.std(), get_corrcoef(t3))

# ############# output ##############
# tensor(13.0542) tensor(17.9457)
# tensor(93.5488) tensor(113.1659)
# tensor(336.6660) tensor(208.7496)