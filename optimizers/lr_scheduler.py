import torch
import numpy as np
# from torch.optim.lr_scheduler import _LRScheduler


class LR_Scheduler(object):
    def __init__(self, optimizer, batch_size, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False, world_size=1):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        
        warmup_lr = world_size*warmup_lr*batch_size/256
        base_lr = world_size*base_lr*batch_size/256
        final_lr = world_size*final_lr*batch_size/256

        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        self.max_steps = num_epochs * iter_per_epoch
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
    def byol_stat(self):
        return self.iter, self.max_steps
    def get_lr(self):
        return self.current_lr

if __name__ == "__main__":
    import torchvision
    model = torchvision.models.resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    epochs = 100
    n_iter = 1000
    scheduler = LR_Scheduler(optimizer, 10, 1, epochs, 3, 0, n_iter)
    import matplotlib.pyplot as plt
    lrs = []
    for epoch in range(epochs):
        for it in range(n_iter):
            lr = scheduler.step()
            lrs.append(lr)
    plt.plot(lrs)
    plt.show()