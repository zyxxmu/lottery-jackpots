from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from utils.options import args as parser_args
import numpy as np

LearnedBatchNorm = nn.BatchNorm2d


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

DenseConv = nn.Conv2d

class GetMask(autograd.Function):
    @staticmethod
    def forward(ctx, mask, prune_rate):
        out = mask.clone()
        _, idx = mask.flatten().sort()
        j = int(prune_rate * mask.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

# Not learning weights, finding lottery jackpots
class PretrainConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape))

    def forward(self, x):
        mask = GetMask.apply(self.clamped_masks, self.prune_rate)
        sparseWeight = mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
    
    @property
    def clamped_masks(self):
        return self.mask.abs()

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        w = self.weight.detach().cpu()
        w = w.view(-1) #c_out * (c_in * k * k) -> 4 * (c_out * c_in * k * k / 4)
        m = self.mask.detach().cpu()
        m = m.view(-1)
        _, indice = torch.topk(torch.abs(w), int(w.size(0)*prune_rate), largest=False)
        m[indice] = 0.95
        self.mask = nn.Parameter(m.view(self.weight.shape))
