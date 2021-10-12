import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Conv2d_NoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nopad=False):
        super(Conv2d_NoPadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_ori):
        pad_val = (self.kernel_size[0] - 1) // 2
        if pad_val == 1:
            pad_val = pad_val + 1
        else:
            pad_val = pad_val
        # print(pad_val)
        # input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        size = []
        size.append(input_ori.shape[2] + pad_val)
        size.append(input_ori.shape[3] + pad_val)
        size = torch.Size(size)
        input_to_conv2d = F.interpolate(input_ori, size, mode='bilinear', align_corners=False)

        # print("Padding: ", self.padding)
        return F.conv2d(input_to_conv2d, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2d_downPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nopad=False):
        super(Conv2d_downPadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_ori):
        pad_val = (self.kernel_size[0] - 1) // 2
        if pad_val == 1:
            pad_val = pad_val + 1
        else:
            pad_val = pad_val

        size = []
        size.append(input_ori.shape[2] - pad_val)
        size.append(input_ori.shape[3] - pad_val)
        size = torch.Size(size)
        # print(self.padding, self.in_channels, self.out_channels)
        conv_output = F.conv2d(input_ori, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # print(conv_output.size())

        input_to_conv2d = F.interpolate(conv_output, size, mode='bilinear', align_corners=False)

        return input_to_conv2d