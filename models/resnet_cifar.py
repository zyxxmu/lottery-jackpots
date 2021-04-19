import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.builder import get_builder

norm_mean, norm_var = 0.0, 1.0

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, inplanes, planes, filter_num, stride=1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = builder.conv3x3(inplanes, filter_num, stride)
        self.bn1 = builder.batchnorm(filter_num)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(filter_num, planes)
        self.bn2 = builder.batchnorm(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if stride != 1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, builder, block, num_layers, num_classes=100):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.cfg_index = 0
        self.inplanes = 32
        self.conv1 = builder.conv3x3(3, self.inplanes, stride=1)
        self.bn1 = builder.batchnorm(self.inplanes)
        self.relu = builder.activation()

        self.layer1 = self._make_layer(builder, block, 32, blocks=n, stride=1)
        self.layer2 = self._make_layer(builder, block, 64, blocks=n, stride=2)
        self.layer3 = self._make_layer(builder, block, 128, blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = builder.conv1x1_fc(128 * block.expansion, num_classes)

    def _make_layer(self, builder, block, planes, blocks, stride):
        layers = []

        layers.append(block(builder, self.inplanes, planes, filter_num=planes, stride=stride))
        self.cfg_index += 1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, filter_num=planes))
            self.cfg_index += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.flatten(1)

def resnet32_cifar10(**kwargs):
    return ResNet(get_builder(), ResBasicBlock, 32, num_classes=10, **kwargs)

def resnet32_cifar100(**kwargs):
    return ResNet(get_builder(), ResBasicBlock, 32, num_classes=100, **kwargs)

