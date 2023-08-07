import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from quant.soft_quant import get_cdf, get_cdf_new

from torch.autograd import Function

def get_sigma(iter, std):
    MAX_ITER = 100
    MIN_ITER = 100
    range = (1e-1, 1e-3)
    if iter < MIN_ITER:
        sigma = range[0]
    else:
        iter -= MIN_ITER
        if iter < MAX_ITER:
            sigma = (1-iter/MAX_ITER) * range[1] + iter/MAX_ITER * range[0]
        # elif iter < 40:
        #     sigma = 1e-4
        # elif iter < 60:
        #     sigma = 1e-5
        else:
            sigma = range[0]

    return sigma * std


def annealed_temperature(t, r, ub, lb=1e-8, backend=np, scheme='exp', **kwargs):
    """
    Return the temperature at time step t, based on a chosen annealing schedule.
    :param t: step/iteration number
    :param r: decay strength
    :param ub: maximum/init temperature
    :param lb: small const like 1e-8 to prevent numerical issue when temperature gets too close to 0
    :param backend: np or tf
    :param scheme:
    :param kwargs:
    :return:
    """
    default_t0 = 700
    if scheme == 'exp':
        tau = backend.exp(-r * t)
    elif scheme == 'exp0':
        # Modified version of above that fixes temperature at ub for initial t0 iterations
        t0 = kwargs.get('t0', default_t0)
        tau = ub * backend.exp(-r * (t - t0))
    elif scheme == 'linear':
        # Cool temperature linearly from ub after the initial t0 iterations
        t0 = kwargs.get('t0', default_t0)
        tau = -r * (t - t0) + ub
    else:
        raise NotImplementedError

    if backend is None:
        return min(max(tau, lb), ub)
    else:
        return backend.minimum(backend.maximum(tau, lb), ub)


def soft_sparse1(matrix, edge, iter, std):
    sigma = get_sigma(iter, std)
    matrix_diff = torch.abs(matrix) - torch.abs(edge)
    matrix_dis = matrix_diff * matrix_diff
    # logits_one = torch.sign(matrix_diff) * matrix_dis / sigma
    logits_one = matrix_diff / sigma


    logits_zero = torch.zeros_like(logits_one)
    logits = torch.stack([logits_zero, logits_one], dim=-1)

    # # TDOO: input outside
    # annealing_scheme = 'linear'
    # annealing_rate = 1e-3  # default annealing_rate = 1e-3
    # # annealing_rate = 5e-2  # default annealing_rate = 1e-3
    # t0 = 0  # default t0 = 700 for 2000 iters
    # # t0 = 100  # default t0 = 700 for 2000 iters
    # T_ub = 1.0
    #
    #
    # temprature = annealed_temperature(iter, r=annealing_rate, ub=T_ub, scheme=annealing_scheme, t0=t0)
    # soft_one_hot = F.gumbel_softmax(logits, tau=temprature, hard=False, eps=1e-10, dim=-1)
    #
    # mask = soft_one_hot[..., 1]

    mask = torch.sigmoid(logits_one)
    return mask * matrix, mask


def ste(matrix):
    x_tile = torch.round(matrix) - matrix.detach() + matrix
    return x_tile


class BinaryQuantize_m(Function):
    @staticmethod
    def forward(ctx, input):
        out = (torch.sign(input) + 1) / 2
        ctx.save_for_backward(out)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.saved_tensors[0].long()
        grad_input = grad_output.clone()

        grad_input[out == 0] = -torch.abs(grad_input[out == 0])
        grad_input[out == 1] = 0 #TODO
        return grad_input


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = (torch.sign(input) + 1) / 2
        ctx.save_for_backward(out)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.saved_tensors[0].long()
        grad_input = grad_output.clone()
        return grad_input


# def soft_sparse(matrix, edge):
#     mask1 = BinaryQuantize_m.apply(torch.abs(matrix) - torch.abs(edge))
#     mask2 = BinaryQuantize.apply(torch.abs(matrix) - torch.abs(edge))
#     return mask1 * matrix, mask2


class FLIPGrad(Function):
    @staticmethod
    def forward(ctx, input, original):
        ctx.save_for_backward(original)

        return input

    @staticmethod
    def backward(ctx, grad_output):
        original = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input = -torch.sign(original) * torch.abs(grad_input)
        return grad_input, None


def soft_sparse(matrix, edge):
    x = matrix / (2 * edge)
    x_mask = torch.abs(x) > 0.5
    x = -x.detach() + x
    x_tile = x * (2 * edge)
    # x_tile = FLIPGrad.apply(x_tile, matrix)
    return torch.where(x_mask, matrix, x_tile), x_mask.float()
    # mask1 = BinaryQuantize_m.apply(torch.abs(matrix) - torch.abs(edge))
    # mask2 = BinaryQuantize.apply(torch.abs(matrix) - torch.abs(edge))
    # return mask1 * matrix, mask2


def sparse(matrix, edge):
    # matrix_diff = torch.abs(matrix) - torch.abs(edge)
    # matrix_dis = matrix_diff * matrix_diff
    # mask = torch.sigmoid(torch.sign(matrix_diff) * matrix_dis / 1e-7)
    mask_new = torch.abs(matrix) > torch.abs(edge)
    return matrix * mask_new, mask_new


# def get_bitrate(matrix, edge, resolution):
#     # x = torch.round(matrix * 512) / 512
#     # x[matrix.abs() <= edge.abs()] = 0
#     # num1 = matrix.numel() - x.count_nonzero()
#     # x = x.nonzero()
#     # x = torch.round(x * 512) / 512
#     # x[x.abs() <= edge.abs()] = 0
#     # num2 = x.numel() - x.count_nonzero()
#     # x = x.nonzero()

#     x = torch.round(matrix * 256) / 256
#     x[x.abs() <= edge.abs()] = 0
#     x = x.flatten()
#     idx = x.nonzero().flatten()
#     x = x.index_select(dim=0, index=idx)

#     if x.shape[0] != 0:
#         x_min = x.abs().min()
#         x_min_ = x.min()
#         probi1, probf1, corr1 = get_cdf(matrix, edge.abs(), resolution=resolution)
#         probi2, probf2, corr2 = get_cdf(matrix, -edge.abs(), resolution=resolution)

#         probi3, probf3, corr3 = get_cdf(matrix, x + 0.5/256, resolution=resolution)
#         probi4, probf4, corr4 = get_cdf(matrix, x - 0.5/256, resolution=resolution)

#         probi5, probf5, corr5 = get_cdf(matrix, x.abs().min(), resolution=resolution)
#         probi6, probf6, corr6 = get_cdf(matrix, edge.abs(), resolution=resolution)

#         probi7, probf7, corr7 = get_cdf(matrix, -edge.abs(), resolution=resolution)
#         probi8, probf8, corr8 = get_cdf(matrix, -x.abs().min(), resolution=resolution)

#         prob1 = (probi1 - probi2).float() + (probf1 - probf2)
#         prob2 = (probi3 - probi4).float() + (probf3 - probf4)
#         prob3 = (probi5 - probi6).float() + (probf5 - probf6)
#         prob4 = (probi7 - probi8).float() + (probf7 - probf8)

#         prob = (-(torch.log2(prob2[prob2 > 0]) - torch.log2(corr1.float()))).sum()[None]
#         prob += -((torch.log2(prob1[prob1 > 0]) - torch.log2(corr1.float()))) * (prob1 / corr1.float()) * matrix.numel()
#         if prob3.item() > 0:
#             prob += -((torch.log2(prob3[prob3 > 0]) - torch.log2(corr1.float()))) * (prob3 / corr1.float()) * matrix.numel()
#         if prob4.item() > 0:
#             prob += -((torch.log2(prob4[prob4 > 0]) - torch.log2(corr1.float()))) * (prob4 / corr1.float()) * matrix.numel()
#     else:
#         probi1, probf1, corr1 = get_cdf(matrix, edge.abs(), resolution=resolution)
#         probi2, probf2, corr2 = get_cdf(matrix, -edge.abs(), resolution=resolution)

#         prob1 = (probi1 - probi2).float() + (probf1 - probf2)
#         prob = -((torch.log2(prob1[prob1 > 0]) - torch.log2(corr1.float()))) * (prob1 / corr1.float()) * matrix.numel()

#     return prob


def get_bitrate(matrix, quant, scale, edge, resolution):
    x = quant.detach()
    zero_cnt = x.numel() - x.count_nonzero()
    x = x.flatten()
    idx = x.nonzero().flatten()
    x = x.index_select(dim=0, index=idx)
    # 找到离edge最近的整数值，并且去掉
    x_min = x.abs().min()
    x[x == x_min] = 0
    plus_cnt = x.numel() - x.count_nonzero()
    x[x == -x_min] = 0
    minus_cnt = x.numel() - x.count_nonzero() - plus_cnt
    idx = x.nonzero().flatten()
    x = x.index_select(dim=0, index=idx)

    if x.shape[0] != 0:
        # 0的cdf
        probi1, probf1, corr1 = get_cdf(matrix, edge.abs(), resolution=resolution)
        probi2, probf2, corr2 = get_cdf(matrix, -edge.abs(), resolution=resolution)

        # 非0非最小量化值的cdf
        probi3, probf3, corr3 = get_cdf(matrix, x + 0.5 * scale, resolution=resolution)
        probi4, probf4, corr4 = get_cdf(matrix, x - 0.5 * scale, resolution=resolution)

        # edge到最小量化值
        probi5, probf5, corr5 = get_cdf(matrix, x_min + 0.5 * scale, resolution=resolution)
        probi6, probf6, corr6 = get_cdf(matrix, edge.abs().detach(), resolution=resolution)

        probi7, probf7, corr7 = get_cdf(matrix, -edge.abs().detach(), resolution=resolution)
        probi8, probf8, corr8 = get_cdf(matrix, -x_min - 0.5 * scale, resolution=resolution)

        prob1 = (probi1 - probi2).float() + (probf1 - probf2)
        prob2 = (probi3 - probi4).float() + (probf3 - probf4)
        prob3 = (probi5 - probi6).float() + (probf5 - probf6)
        prob4 = (probi7 - probi8).float() + (probf7 - probf8)
        
        prob = -((torch.log2(prob1[prob1 > 0]) - torch.log2(corr1.float()))) * zero_cnt
        prob += (-(torch.log2(prob2[prob2 > 0]) - torch.log2(corr1.float()))).sum()[None]
        if prob3.item() > 0:
            prob += -((torch.log2(prob3[prob3 > 0]) - torch.log2(corr1.float()))) * plus_cnt
        if prob4.item() > 0:
            prob += -((torch.log2(prob4[prob4 > 0]) - torch.log2(corr1.float()))) * minus_cnt
        # print("zero_cnt {}, plus_cnt {}, minus cnt {}, matrix num {}".format(zero_cnt, plus_cnt, minus_cnt, matrix.numel()))
        # print("zero_prob {}, plus_prob {}, minus prob {}, matrix num{}".format(prob1, prob3, prob4, corr1))
    else:
        probi1, probf1, corr1 = get_cdf(matrix, edge.abs(), resolution=resolution)
        probi2, probf2, corr2 = get_cdf(matrix, -edge.abs(), resolution=resolution)

        prob1 = (probi1 - probi2).float() + (probf1 - probf2)
        prob = -((torch.log2(prob1[prob1 > 0]) - torch.log2(corr1.float()))) * (prob1 / corr1.float()) * matrix.numel()

    return prob


def get_bitrate_new(matrix, quant, scale, edge, resolution):
    if torch.all(quant==0):
        return torch.zeros_like(quant)
    
    sum = quant.numel()
    x_tilde = quant
    if matrix.numel() > 128 * 128:
        sample_num = matrix.numel() // 256
        sample_idx = torch.randperm(matrix.numel())[:sample_num]
        matrix = matrix.flatten()[sample_idx]
        x_tilde = quant.flatten()[sample_idx]

    x = quant.clone()
    zero_cnt = x.numel() - x.count_nonzero()
    x = x.flatten()
    idx = x.nonzero().flatten()
    x = x.index_select(dim=0, index=idx)
    # 找到离edge最近的整数值，并且去掉
    x_min_tmp = torch.abs(x.clone())
    x_min = x_min_tmp.min()
    x[x == x_min] = 0
    plus_cnt = x.numel() - x.count_nonzero()
    x[x == -x_min] = 0
    minus_cnt = x.numel() - x.count_nonzero() - plus_cnt
    # idx = x.nonzero().flatten()
    # x = x.index_select(dim=0, index=idx)
    
    x_tilde[x_tilde == x_min] = 0
    x_tilde[x_tilde == -x_min] = 0
    idx = x_tilde.nonzero().flatten()
    x = x_tilde.index_select(dim=0, index=idx)

    if x.shape[0] != 0:
        # 0的cdf
        probi1, probf1, corr1 = get_cdf_new(matrix, edge.abs(), resolution=resolution)
        probi2, probf2, corr2 = get_cdf_new(matrix, -edge.abs(), resolution=resolution)

        # 非0非最小量化值的cdf
        probi3, probf3, corr3 = get_cdf_new(matrix, x + 0.5 * scale, resolution=resolution)
        probi4, probf4, corr4 = get_cdf_new(matrix, x - 0.5 * scale, resolution=resolution)

        # edge到最小量化值
        probi5, probf5, corr5 = get_cdf_new(matrix, x_min + 0.5 * scale, resolution=resolution)
        probi6, probf6, corr6 = get_cdf_new(matrix, edge.abs().detach(), resolution=resolution)

        probi7, probf7, corr7 = get_cdf_new(matrix, -edge.abs().detach(), resolution=resolution)
        probi8, probf8, corr8 = get_cdf_new(matrix, -x_min - 0.5 * scale, resolution=resolution)

        prob1 = (probi1 - probi2).float() + (probf1 - probf2)
        prob2 = (probi3 - probi4).float() + (probf3 - probf4)
        prob3 = (probi5 - probi6).float() + (probf5 - probf6)
        prob4 = (probi7 - probi8).float() + (probf7 - probf8)
        
        prob = -((torch.log2(prob1[prob1 > 0]) - torch.log2(corr1.float()))) * zero_cnt
        # prob += (-(torch.log2(prob2[prob2 > 0]) - torch.log2(corr1.float()))).sum()[None]
        prob += (-(torch.log2(prob2[prob2 > 0]) - torch.log2(corr1.float()))).mean() * (sum - zero_cnt - plus_cnt -minus_cnt) 
        if prob3.item() > 0:
            prob += -((torch.log2(prob3[prob3 > 0]) - torch.log2(corr1.float()))) * plus_cnt
        if prob4.item() > 0:
            prob += -((torch.log2(prob4[prob4 > 0]) - torch.log2(corr1.float()))) * minus_cnt
        # print("zero_cnt {}, plus_cnt {}, minus cnt {}, matrix num {}".format(zero_cnt, plus_cnt, minus_cnt, matrix.numel()))
        # print("zero_prob {}, plus_prob {}, minus prob {}, matrix num{}".format(prob1, prob3, prob4, corr1))
    else:
        probi1, probf1, corr1 = get_cdf_new(matrix, edge.abs(), resolution=resolution)
        probi2, probf2, corr2 = get_cdf_new(matrix, -edge.abs(), resolution=resolution)

        prob1 = (probi1 - probi2).float() + (probf1 - probf2)
        prob = -((torch.log2(prob1[prob1 > 0]) - torch.log2(corr1.float()))) * (prob1 / corr1.float()) * matrix.numel()

    return prob


def get_bitrate_sparse(matrix, edge, resolution):
    # x = torch.round(matrix * 512) / 512
    # x[matrix.abs() <= edge.abs()] = 0
    # num1 = matrix.numel() - x.count_nonzero()
    # x = x.nonzero()
    # x = torch.round(x * 512) / 512
    # x[x.abs() <= edge.abs()] = 0
    # num2 = x.numel() - x.count_nonzero()
    # x = x.nonzero()

    # x = torch.round(matrix * 256) / 256
    # x[x.abs() <= edge.abs()] = 0
    # x = x.flatten()
    # idx = x.nonzero().flatten()
    # x = x.index_select(dim=0, index=idx)

    probi1, probf1, corr1 = get_cdf(matrix, edge.abs(), resolution=resolution)
    probi2, probf2, corr2 = get_cdf(matrix, -edge.abs(), resolution=resolution)

    prob1 = (probi1 - probi2).float() + (probf1 - probf2)
    prob = -((torch.log2(prob1[prob1 > 0]) - torch.log2(corr1.float())))

    return prob