

from multiprocessing import popen_spawn_posix
from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from utils.options import args as parser_args
import numpy as np
import pdb
LearnedBatchNorm = nn.BatchNorm2d


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

DenseConv = nn.Conv2d

class GetMask(autograd.Function):
    @staticmethod
    def forward(ctx, mask, b_mask):
        return b_mask

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

# Not learning weights, finding lottery jackpots
class PretrainConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape))
        self.b_mask = torch.ones(self.weight.shape, requires_grad=False)

    def forward(self, x):
        mask = GetMask.apply(self.mask, self.b_mask)
        sparseWeight = mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def pop_up(self, percent):
        m = self.mask.clone()
        pop_num = int(self.prune_rate * self.mask.numel())
        _, idx = m.flatten().sort()
        flat_bm = self.b_mask.flatten()
        flat_bm[idx[:pop_num]] = 0
        flat_bm[idx[pop_num:]] = 1
    
    def ls_pop_up(self, rate):
        m = self.mask.clone().flatten()
        prune_num = int(self.prune_rate * self.mask.numel())
        sorted, _ = m.sort()

        flat_bm = self.b_mask.flatten()
        idx0 = torch.squeeze(torch.nonzero(flat_bm==0))
        idx1 = torch.squeeze(torch.nonzero(flat_bm))
        pop_idx0 = torch.squeeze(torch.nonzero(m[idx0] > sorted[prune_num-1]))
        pop_idx1 = torch.squeeze(torch.nonzero(m[idx1] <= sorted[prune_num-1]))
       
        if pop_idx0.numel() != 0 and pop_idx1.numel() != 0 and pop_idx0.numel() == pop_idx1.numel():
            pop_num = pop_idx0.numel()
            _, sorted_pop_idx0 = torch.sort(m[idx0][pop_idx0], descending=True)
            _, sorted_pop_idx1 = torch.sort(m[idx1][pop_idx1], descending=False)
            pop_idx0 = [pop_idx0.tolist()]
            pop_idx1 = [pop_idx1.tolist()]
            sorted_pop_idx0 = [sorted_pop_idx0.tolist()]
            sorted_pop_idx1 = [sorted_pop_idx1.tolist()]
            if pop_num > 1:
                pop_num = math.ceil(len(sorted_pop_idx0[0])*rate)
                flat_bm[idx0[pop_idx0][sorted_pop_idx0[0][:pop_num]]] = 1
                flat_bm[idx1[pop_idx1][sorted_pop_idx1[0][:pop_num]]] = 0   
            else:
                flat_bm[idx0[pop_idx0]] = 1
                flat_bm[idx1[pop_idx1]] = 0   

            return pop_num 
        

        return 0

    def final_pop_up(self, rate):
        m = self.mask.clone().flatten()
        prune_num = int(self.prune_rate * self.mask.numel())
        sorted, _ = m.sort()

        flat_bm = self.b_mask.flatten()
        idx0 = torch.squeeze(torch.nonzero(flat_bm==0))
        idx1 = torch.squeeze(torch.nonzero(flat_bm))
        pop_idx0 = torch.squeeze(torch.nonzero(m[idx0] > sorted[prune_num-1]))
        pop_idx1 = torch.squeeze(torch.nonzero(m[idx1] <= sorted[prune_num-1]))
       
        if pop_idx0.numel() != 0 and pop_idx1.numel() != 0 and pop_idx0.numel() == pop_idx1.numel():
            #print(pop_idx0.numel(), pop_idx1.numel())
            pop_num = pop_idx0.numel()
            _, sorted_pop_idx0 = torch.sort(m[idx0][pop_idx0], descending=True)
            _, sorted_pop_idx1 = torch.sort(m[idx1][pop_idx1], descending=False)
            pop_idx0 = [pop_idx0.tolist()]
            pop_idx1 = [pop_idx1.tolist()]
            sorted_pop_idx0 = [sorted_pop_idx0.tolist()]
            sorted_pop_idx1 = [sorted_pop_idx1.tolist()]
            if pop_num > 1:
                pop_num = math.ceil(len(sorted_pop_idx0[0])*rate)
                #pop_num = 1
                flat_bm[idx0[pop_idx0][sorted_pop_idx0[0][:pop_num]]] = 1
                flat_bm[idx1[pop_idx1][sorted_pop_idx1[0][:pop_num]]] = 0   
            else:
                flat_bm[idx0[pop_idx0]] = 1
                flat_bm[idx1[pop_idx1]] = 0  
            #import pdb; pdb.set_trace()
            #self.mask.clamp(0,0)

            return pop_num 
        return 0

    def val_pop_up(self, number):
        m = self.mask.clone().flatten()
        prune_num = int(self.prune_rate * self.mask.numel())
        sorted, _ = m.sort()

        flat_bm = self.b_mask.flatten()
        idx0 = torch.squeeze(torch.nonzero(flat_bm==0))
        idx1 = torch.squeeze(torch.nonzero(flat_bm))
        pop_idx0 = torch.randint(0, idx0.size(0), number)
        pop_idx1 = torch.randint(0, idx1.size(0), number)

        flat_bm[idx0[pop_idx0]] = 1
        flat_bm[idx1[pop_idx1]] = 0  

        
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        w = self.weight.detach().cpu()
        w = w.view(-1) #c_out * (c_in * k * k) -> 4 * (c_out * c_in * k * k / 4)
        m = self.mask.detach().cpu()
        m = m.view(-1)
        b_m = self.b_mask.detach().cpu()
        b_m = b_m.view(-1)
        #import pdb; pdb.set_trace()
        _, indice = torch.topk(torch.abs(w), int(w.size(0)*prune_rate), largest=False)
        b_m[indice] = 0
        m[indice] = 0.99
        self.b_mask = nn.Parameter(b_m.view(self.weight.shape), requires_grad=False)
        self.mask = nn.Parameter(m.view(self.weight.shape))
