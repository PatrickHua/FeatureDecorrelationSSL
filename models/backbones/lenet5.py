import torch
import torch.nn as nn

import torch.nn.functional as F


class LeNet5(nn.Module):

    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            # nn.Tanh(),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            # nn.Tanh(),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            # nn.Tanh()
            # nn.ReLU()
            nn.AdaptiveAvgPool2d((1,1))
            # import torchvision
            # torchvision.models.resnet50
        )

        self.fc = nn.Linear(in_features=120, out_features=n_classes)
        
        
        # nn.Sequential(
        #     nn.Linear(in_features=120, out_features=84),
        #     nn.Tanh(),
        #     nn.Linear(in_features=84, out_features=n_classes),
        # )


    def forward(self, x):
        x = self.feature_extractor(x)
        
        # breakpoint()
        logits = self.fc(x.flatten(1))
        # probs = F.softmax(logits, dim=1)
        return logits



if __name__ == '__main__':
    model = LeNet5(n_classes=10)
    x = torch.randn((1, 1, 28, 28))
    y = model(x)
    print(y.shape)


