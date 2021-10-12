'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


cfg = {
    'VGG5': [32, 'M', 64, 'M', 128, 'M', 256],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def upsample_bilinear(x, size):
    if float(torch.__version__[:3]) <= 0.3:
        out = F.upsample(x, size, mode='bilinear')
    else:
        out = F.interpolate(x, size, mode='bilinear', align_corners=True)
    return out


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        # print(out.shape)
        # out = out.view(out.size(0), -1)
        # print(out.shape, x.shape)
        out = F.avg_pool2d(out, out.shape[3])

        out = out.view(out.size(0), -1)
        # print(out.shape)
        out_location = self.classifier(out)
        return out_location

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGGAP(nn.Module):
    def __init__(self, vgg_name, num_class=25):
        super(VGGGAP, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.pre_classifier = nn.Conv2d(512, num_class, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.features(x)
        out = self.pre_classifier(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)
        # print(out.shape)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGGAPReflection(nn.Module):
    def __init__(self, vgg_name, num_class=25):
        super(VGGGAPReflection, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.pre_classifier = nn.Conv2d(512, num_class, kernel_size=3, padding=1, padding_mode='reflect')

    def forward(self, x):
        out = self.features(x)
        out = self.pre_classifier(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)
        # print(out.shape)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, padding_mode='reflect'),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGGAPReplicate(nn.Module):
    def __init__(self, vgg_name, num_class=25):
        super(VGGGAPReplicate, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.pre_classifier = nn.Conv2d(512, num_class, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):
        out = self.features(x)
        out = self.pre_classifier(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)
        # print(out.shape)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, padding_mode='replicate'),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGGAPShuffle(nn.Module):
    def __init__(self, vgg_name, num_class=25, shuffle=False):
        super(VGGGAPShuffle, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.pre_classifier = nn.Conv2d(512, num_class, kernel_size=3, padding=1)
        self.classifier = nn.Linear(512, num_class)
        self.shuffle = shuffle

    def forward(self, x):
        out = self.features(x)
        # out = self.pre_classifier(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)

        if self.shuffle:
            rand_index = torch.randperm(512)
            out = out[:, rand_index]
            out = self.classifier(out)

        # print(out.shape)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, padding_mode='zeros'),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGGAPShuffleReflection(nn.Module):
    def __init__(self, vgg_name, num_class=25, shuffle=False):
        super(VGGGAPShuffleReflection, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.pre_classifier = nn.Conv2d(512, num_class, kernel_size=3, padding=1)
        self.classifier = nn.Linear(512, num_class)
        self.shuffle = shuffle

    def forward(self, x):
        out = self.features(x)
        # out = self.pre_classifier(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)

        if self.shuffle:
            rand_index = torch.randperm(512)
            out = out[:, rand_index]
            out = self.classifier(out)

        # print(out.shape)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, padding_mode='reflect'),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGGAPShuffleReplicate(nn.Module):
    def __init__(self, vgg_name, num_class=25, shuffle=False):
        super(VGGGAPShuffleReplicate, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.pre_classifier = nn.Conv2d(512, num_class, kernel_size=3, padding=1)
        self.classifier = nn.Linear(512, num_class)
        self.shuffle = shuffle

    def forward(self, x):
        out = self.features(x)
        # out = self.pre_classifier(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)

        if self.shuffle:
            rand_index = torch.randperm(512)
            out = out[:, rand_index]
            out = self.classifier(out)

        # print(out.shape)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, padding_mode='replicate'),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGSemantic(nn.Module):
    def __init__(self, vgg_name):
        super(VGGSemantic, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 10)
        # self.semantic_location = nn.Sequential(
        #     OrderedDict([
        #         ('conv5_4', nn.Conv2d(512, 256, 3, 1, 1, 1)),
        #         ('conv5_4_bn', nn.BatchNorm2d(256)),
        #         ('conv5_4_relu', nn.ReLU()),
        #         ('drop5_4', nn.Dropout2d(p=0.1)),
        #         ('conv6', nn.Conv2d(256, 11, 1, stride=1, padding=0)),
        #     ])
        # )

        self.semantic_location = nn.Conv2d(512, 11, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.features(x)
        # out = out.view(out.size(0), -1)
        out = self.semantic_location(out)
        # print(out.shape)
        out = upsample_bilinear(out, x.size()[2:])
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



