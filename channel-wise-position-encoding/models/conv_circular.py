import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class conv2d_circular(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nopad=False):
        super(conv2d_circular, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_ori):
        # print(self.padding[0])
        # expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                            # (self.padding[0] + 1) // 2, self.padding[0] // 2)
        input = F.pad(input_ori, (self.padding[0], self.padding[0], self.padding[0], self.padding[0]), mode="circular")
        # input = F.pad(input_ori, (1,1,1,1), mode="circular")
        # input = F.pad(input_ori, expanded_padding, mode="circular")

        return F.conv2d(input, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)

        # return F.conv2d(F.pad(input_ori, expanded_padding, mode='circular'),
        #                 self.weight, self.bias, self.stride,
        #                 _pair(0), self.dilation, self.groups)
