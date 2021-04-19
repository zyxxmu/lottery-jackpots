import math

import torch
import torch.nn as nn

from utils.options import args
import utils.conv_type


class Builder(object):
    def __init__(self, conv_layer, bn_layer):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer

    def conv(self, kernel_size, in_planes, out_planes, stride=1, bias=False):
        conv_layer = self.conv_layer

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=bias,
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=bias,
            )
        else:
            return None

        return conv

    def conv2d(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        return self.conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )


    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride)
        return c

    def conv1x1_fc(self, in_planes, out_planes, stride=1):
        """full connect layer"""
        c = self.conv(1, in_planes, out_planes, stride=stride, bias=True)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes, last_bn=False):
        return self.bn_layer(planes)

    def activation(self):
        return (lambda: nn.ReLU(inplace=True))()



def get_builder():

    print("==> Conv Type: {}".format(args.conv_type))
    print("==> BN Type: {}".format(args.bn_type))

    conv_layer = getattr(utils.conv_type, args.conv_type)
    bn_layer = getattr(utils.conv_type, args.bn_type)

    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer)

    return builder
