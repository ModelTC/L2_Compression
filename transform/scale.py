import torch
from common.tools import mysign, myabs

class Scale_T():
    @staticmethod
    def encode_param(param, trans_param):
        scale = trans_param["scale"]
        return param / scale

    @staticmethod
    def decode_param(code, trans_param):
        scale = trans_param["scale"]
        return code * scale

    @staticmethod
    def get_init_trans_param(param):
        trans_param = {}
        device = param.device
        trans_param["scale"] = torch.tensor((param.max() - param.min()) / 256, device=device).requires_grad_(True)
        return trans_param

    @staticmethod
    def get_trainable_list(trans_param):
        return [trans_param["scale"],]