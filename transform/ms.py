import torch
from common.tools import mysign, myabs

class MS_T():
    @staticmethod
    def encode_param(param, trans_param):
        param_range = trans_param["param_range"]
        scale = trans_param["scale"]

        assert param_range.shape[0] + 1 == scale.shape[0]
        param_sign = mysign(param)
        res = torch.zeros_like(param)
        filled = torch.zeros_like(param).bool()
        base_last = 0
        range_last = 0
        for i in range(len(param_range)):
            mask = (myabs(param) < param_range[i]) & (~filled)
            res[mask] = (base_last + (myabs(param) - range_last) / myabs(scale[i]))[mask]
            filled = filled | mask
            base_last += ((param_range[i] - range_last) / myabs(scale[i]))
            range_last = param_range[i]
        res[~filled] = (base_last + (myabs(param) - range_last) / myabs(scale[-1]))[~filled]
        return res * param_sign

    @staticmethod
    def decode_param(code, trans_param):
        param_range = trans_param["param_range"]
        scale = trans_param["scale"]

        assert param_range.shape[0] + 1 == scale.shape[0]
        code_sign = mysign(code)
        res = torch.zeros_like(code)
        filled = torch.zeros_like(code).bool()
        base_last = 0
        range_last = 0
        for i in range(len(param_range)):
            base_now = (base_last + (param_range[i] - range_last) / scale[i])
            mask = (myabs(code) < base_now) & (~filled)
            res[mask] = (range_last + (myabs(code) - base_last) * scale[i])[mask]
            filled = filled | mask
            base_last = base_now
            range_last = param_range[i]

        res[~filled] = (range_last + (myabs(code) - base_last) * scale[-1])[~filled]
        return res * code_sign

    @staticmethod
    def get_init_trans_param(param):
        trans_param = {}
        device = param.device
        NUM_LIN = 5
        trans_param["scale"] = torch.full((NUM_LIN,), (param.max() - param.min()) / 256, device=device).requires_grad_(True)
        trans_param["param_range"] = (torch.arange(1, NUM_LIN, device=device, dtype=torch.float32) * (param.abs().max() / NUM_LIN)).detach().requires_grad_(False)
        return trans_param

    @staticmethod
    def get_trainable_list(trans_param):
        return [trans_param["scale"],]