'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = F.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetPositionReflect(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_location=9):
        super(ResNetPositionReflect, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # print(block.expansion)

        self.linear_location = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # print()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[3])
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        # print("Class: ", out_class.shape)
        out_location = self.linear_location(out)
        return out_location


'''class ResNetPosition(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_location=9):
        super(ResNetPosition, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # print(block.expansion)

        self.linear = nn.Linear(512, num_classes)
        self.linear_location = nn.Linear(512, num_location)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # print()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[3])
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out_class = self.linear(out)
        # print("Class: ", out_class.shape)
        out_location = self.linear_location(out)
        return out_class, out_location'''


def upsample_bilinear(x, size):
    if float(torch.__version__[:3]) <= 0.3:
        out = F.upsample(x, size, mode='bilinear')
    else:
        out = F.interpolate(x, size, mode='bilinear', align_corners=True)
    return out


class ResNetSemanticPositionReflect(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_location=9):
        super(ResNetSemanticPositionReflect, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # print(block.expansion)

        '''self.semantic_location = nn.Sequential(
            # nn.AvgPool2d(kernel_size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True),
            nn.Conv2d(
                in_channels=2048,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        )'''

        # self.semantic_location = nn.Sequential(
        #     OrderedDict([
        #         ('conv5_4', nn.Conv2d(512, 256, 3, 1, 1, 1, padding_mode='reflect')),
        #         ('conv5_4_bn', nn.BatchNorm2d(256)),
        #         ('conv5_4_relu', nn.ReLU()),
        #         ('drop5_4', nn.Dropout2d(p=0.1)),
        #         ('conv6', nn.Conv2d(256, num_classes, 1, stride=1, padding=0)),
        #     ])
        # )

        self.semantic_location = nn.Conv2d(512*block.expansion, 11, kernel_size=1, padding=0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # print()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)
        # print(out.shape)
        # print("Class: ", out_class.shape)
        out_location = self.semantic_location(out)
        # print(out_location.shape)
        out_location = upsample_bilinear(out_location, x.size()[2:])

        return out_location


class ResNet18PositionShuffleReflect(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_location=9):
        super(ResNet18PositionShuffleReflect, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # print(block.expansion)

        # self.pre_classifier = nn.Conv2d(512, num_classes, kernel_size=3, padding=1, padding_mode='reflect')

        self.linear_location = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, shuffle=False, layer=4):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)

        if shuffle and layer == 1:
            rand_index = torch.randperm(64)
            out = out[:, rand_index]
            out = self.layer1(out)
        else:
            out = self.layer1(out)

        if shuffle and layer == 2:
            rand_index = torch.randperm(64)
            out = out[:, rand_index]
            out = self.layer2(out)
        else:
            out = self.layer2(out)

        if shuffle and layer == 3:
            rand_index = torch.randperm(128)
            out = out[:, rand_index]
            out = self.layer3(out)
        else:
            out = self.layer3(out)

        if shuffle and layer == 4:
            rand_index = torch.randperm(256)
            out = out[:, rand_index]
            out = self.layer4(out)
        else:
            out = self.layer4(out)

        # out = self.pre_classifier(out)
        out = F.avg_pool2d(out, out.shape[3])
        # print(out.shape)
        out = out.view(out.size(0), -1)

        if shuffle and layer == 5:
            rand_index = torch.randperm(512)
            out = out[:, rand_index]
            out = self.linear_location(out)
        # print(out.shape)
        return out


def ResNet18GAPShuffleReflect(grid_size=3, num_class=10):
    print("ResNet18 Shuffle Position")
    return ResNet18PositionShuffleReflect(BasicBlock, [2, 2, 2, 2], num_classes=num_class, num_location=grid_size*grid_size)

def ResNet18PositionReflect(grid_size=3, num_class=10):
    print('Reflect Padding!')
    return ResNetPositionReflect(BasicBlock, [2, 2, 2, 2], num_classes=num_class, num_location=grid_size*grid_size)


def ResNet18SemanticPositionReflect(grid_size=3, num_class=11):
    print('Reflect Padding!')
    return ResNetSemanticPositionReflect(BasicBlock, [2,2,2,2], num_classes=num_class, num_location=grid_size*grid_size)


