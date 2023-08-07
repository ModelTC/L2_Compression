import torch
from common.tools import mysign, myabs

class EdgeScale_T():
    @staticmethod
    def encode_param(param, trans_param):
        edge = trans_param["edge"]
        scale = trans_param["scale"]
        param_sign = torch.sign(param)
        # res = torch.zeros_like(param)
        reserve_mask = torch.abs(param) > torch.abs(edge)
        sparse = (param / (2 * torch.abs(edge)))
        reserve = (param_sign * (0.5 + (torch.abs(param) - torch.abs(edge)) / torch.abs(scale)))
        # param = soft_sparse(param, edge)[0]
        # res = param / scale
        return torch.where(reserve_mask, reserve, sparse)
    @staticmethod
    def decode_param(code, trans_param):
        edge = trans_param["edge"]
        scale = trans_param["scale"]
        code_sign = torch.sign(code)
        # res = torch.zeros_like(code)
        reserve_mask = torch.abs(code) > 0.5
        sparse = (code * (2 * torch.abs(edge)))
        reserve = (code_sign * (torch.abs(edge) + (torch.abs(code) - 0.5) * torch.abs(scale)))
        return torch.where(reserve_mask, reserve, sparse)

    @staticmethod
    def get_init_trans_param(param):
        trans_param = {}
        device = param.device
        trans_param["edge"] = torch.tensor((param.max() - param.min()) / 256, device=device).requires_grad_(True)
        trans_param["scale"] = torch.tensor((param.max() - param.min()) / 256, device=device).requires_grad_(True)
        return trans_param

    @staticmethod
    def get_trainable_list(trans_param):
        return [trans_param["edge"], trans_param["scale"]]