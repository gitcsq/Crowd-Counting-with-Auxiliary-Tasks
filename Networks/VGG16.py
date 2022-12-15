import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.front_end1 = make_layers(cfg1, 3, batch_norm=True)
        self.front_end = make_layers(cfg, 3, batch_norm=True)
        self.back = nn.Sequential(
              nn.Conv2d(512, 128, 3, 1, 1),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 64, 3, 1, 1),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.Conv2d(64, 1, 1),
              nn.ReLU()
        )
        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print("loading pretrained vgg16_bn!")
        if os.path.exists("/home/wfs/PycharmProjects_wfs/DM-Count-master/DM-Count-master/Networks/SWINIR/vgg16_bn.pth"):
            print("find pretrained weights!")
            vgg16_bn = models.vgg16_bn(pretrained=False)
            vgg16_weights = torch.load("/home/wfs/PycharmProjects_wfs/DM-Count-master/DM-Count-master/Networks/SWINIR/vgg16_bn.pth")
            vgg16_bn.load_state_dict(vgg16_weights)
        else:
            vgg16_bn = models.vgg16_bn(pretrained=True)
        # the front conv block's parameter no training
        # for p in self.front_end1.parameters():
        #     p.requires_grad = False

        self.front_end.load_state_dict(vgg16_bn.features[:33].state_dict())

    def forward(self, x):
        x = self.front_end(x)
        x = self.back(x)
        return x

def make_layers(cfg, in_channels, batch_norm=True):
    layers = []
    # in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

if __name__ == '__main__':
    x = torch.ones(8, 3, 256, 256)
    net = VGG()
    y = net(x)
    print(y.size())