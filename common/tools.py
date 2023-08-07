import numpy
import torch

def get_np_size(x):
    return x.size * x.itemsize

def myabs(x):
    return torch.where(x==0, x, torch.abs(x))

def mysign(x):
    return torch.where(x == 0, torch.ones_like(x), torch.sign(x))

def to_np(x):
    return x.detach().cpu().numpy()