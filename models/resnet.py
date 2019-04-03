"""
ResNet model.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform, constant


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  dilation=dilation,
                  bias=False),
        nn.BatchNorm2d(out_channels),
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_bn(in_channels, channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_bn(channels, channels)
        self.downsample = downsample

    def forward(self, x):
        identify = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            identify = self.downsample(x)
        out += identify
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    num_blocks = [2, 2, 2, 2]
    strides = [1, 2, 2, 2]

    def __init__(self, label_number):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_channels = 64
        self.stages = []
        for i, num in enumerate(self.num_blocks):
            stride = self.strides[i]
            channels = 64 * 2 ** i
            stage = self.make_stage(in_channels, channels, stride, num)
            stage_name = "stage{}".format(str(i))
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)
            in_channels = channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(channels, label_number, bias=True)

    def make_stage(self, in_channels, channels, stride, num_block):
        downsample = None
        if stride != 1:
            downsample = conv_bn(in_channels, channels, kernel_size=1, stride=stride, padding=0)
        layers = []
        layers.append(BasicBlock(in_channels, channels, stride, downsample))
        for i in range(1, num_block):
            layers.append(BasicBlock(channels, channels))
        return nn.Sequential(*layers)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_uniform(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant(m, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for stage_name in self.stages:
            stage_layer = getattr(self, stage_name)
            x = stage_layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ResNet18()
    print(model)
