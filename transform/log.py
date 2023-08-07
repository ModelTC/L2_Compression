import torch
from common.tools import mysign, myabs

class Log_T():
    @staticmethod
    def encode_param(param, trans_param):
        shift = trans_param["shift"]
        scale = trans_param["scale"]
        inner_scale = trans_param["inner_scale"]
        return mysign(param) * torch.log(shift + myabs(param) / inner_scale) / scale
    @staticmethod
    def decode_param(code, trans_param):
        shift = trans_param["shift"]
        scale = trans_param["scale"]
        inner_scale = trans_param["inner_scale"]
        return mysign(code) * (torch.exp(myabs(code) * scale) - shift) * inner_scale

    @staticmethod
    def get_init_trans_param(param):
        trans_param = {}
        device = param.device
        trans_param["shift"] = torch.tensor(1.0, device=device).requires_grad_(True)
        trans_param["inner_scale"] = torch.tensor((param.abs().max() / 1.718281828459045), device=device).requires_grad_(True)
        trans_param["scale"] = torch.tensor(1.0 / 64, device=device).requires_grad_(True)
        return trans_param

    @staticmethod
    def get_trainable_list(trans_param):
        return [trans_param["inner_scale"], trans_param["scale"]]