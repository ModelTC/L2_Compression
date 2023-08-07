import torch

def cal_centric(scale, clip_min, clip_max):
    '''
    infinite centric
    '''
    centric = torch.arange(-1, ((clip_max - clip_min)/scale).item() + 2, 1, device=scale.device) * scale + clip_min
    return centric

