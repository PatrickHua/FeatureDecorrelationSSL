import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

def visualize_matrix(weight, dilation=16, title='', cmap='bone', mode='save', save_path=None):
    assert isinstance(weight, (np.ndarray, torch.Tensor)) and weight.ndim == 2
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()

    matplotlib.use( 'tkagg' if mode=='show' else 'Agg')
    plt.title(label=title)
    # or Agg
    # weight = torch.nn.functional.interpolate(mat, scale_factor=dilation, mode='nearest')
    # weight = mat
    # linear.weight.detach().cpu().numpy()
    weight = weight[::dilation,::dilation]
    # if linear.bias is not None:
    #     bias = linear.bias.detach().cpu().numpy()
    #     bias = bias[::dilation]
    #     data = np.concatenate((bias[None,:], weight.T), axis=0)
    # else:
    data = weight.T

    plt.tight_layout()
    

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8 ))
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = axes[0].imshow(data, cmap=cmap, aspect='auto')
    fig.colorbar(im, cax=cax, orientation='vertical')

    (n, bins, patches) = axes[1].hist(data.flatten(), range=(data.min(),data.max()), density=True, bins=10)
    for i, p in enumerate(patches):
        plt.setp(p, 'facecolor', eval(f'plt.cm.{cmap}')(i/10))

    axes[1].set_ylim(ymin=n.min()-1)

    cm = plt.cm.get_cmap('bone')

    if mode=='show':
        plt.show()
    elif mode=='save':
        assert save_path is not None
        plt.savefig(save_path)
    else:
        raise Exception
    plt.close()