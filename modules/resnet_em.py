'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.resnet import BasicBlock, Bottleneck

class ResNet_EM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_EM, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.softmax1 = nn.Softmax(dim=1)
        self.linear2 = nn.Linear(num_classes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.softmax1(out)
        out = self.linear2(out)
        return out


def ResNet18_EM():
    return ResNet_EM(BasicBlock, [2,2,2,2])

def ResNet34_EM():
    return ResNet_EM(BasicBlock, [3,4,6,3])

def ResNet50_EM():
    return ResNet_EM(Bottleneck, [3,4,6,3])

def ResNet101_EM():
    return ResNet_EM(Bottleneck, [3,4,23,3])

def ResNet152_EM():
    return ResNet_EM(Bottleneck, [3,8,36,3])


#
