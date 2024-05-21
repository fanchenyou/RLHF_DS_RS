import os
import math
import torch
import numpy as np
import torch
from torch import nn

def mkdir_p(path):
    if os.path.isdir(path):
        print('%s exists' % (path,))
    else:
        os.mkdir(path)
        print('%s created' % (path,))

class ZeroSoftmax(nn.Module):

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x



def get_ID(s1, s2, device):
    identity_spatial = torch.ones((s1, s2, s2)) * torch.eye(s2)  # [obs_len N N]
    identity_temporal = torch.ones((s2, s1, s1)) * torch.eye(s1)  # [N obs_len obs_len]
    identity = [identity_spatial.to(device), identity_temporal.to(device)]
    return identity


def A_distance_to_energy_attention(t, two_sigma_square, zero_diagnoal=True):
    assert t.ndim==4
    assert t.size(1)==t.size(2)
    assert t.size(-1) == 2  # [8/12, N, N, 2=(dx,dy)]
    t_sq = torch.sum(t ** 2, dim=-1)
    t_kernel = torch.exp(-t_sq / two_sigma_square)
    assert t_kernel.ndim==3
    if zero_diagnoal:
        for i in range(t.size(0)):
            torch.diagonal(t_kernel[i,...]).fill_(-1e9)
    t_kernel = torch.softmax(t_kernel, dim=-1)

    return t_kernel