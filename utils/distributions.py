from __future__ import print_function
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + torch.pow((x - mean) * torch.exp(-log_var / 2), 2))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)