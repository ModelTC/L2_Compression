import torch
import torch.nn.functional as F
from clib.lin_stat.lin_stat import Linear_Stat

linstat = Linear_Stat(resolution=10)
# from pytorch3d.ops import knn_points


def distance_metric(matrix, centric):
    return torch.abs(matrix[:, None] - centric[None, :])


def get_cdf(matrix, x, resolution):
    # sorted_matrix = torch.sort(matrix.flatten())
    interval = (matrix.max() - matrix.min())/ resolution
    quantized_matrix = torch.round(matrix / interval).flatten().long()
    # unique, inverse_indices, counts = torch.unique(quantized_matrix, return_inverse=True, return_counts=True)
    unique_offset = quantized_matrix.min()
    _ind = quantized_matrix - unique_offset
    zeros = torch.zeros(_ind.max()+2, device=matrix.device, dtype=torch.float32)
    spline_pmf = torch.scatter_add(zeros, 0, _ind+1, torch.ones_like(_ind).float()).long()
    # spline_pmf = spline_pmf
    # print("1")
    # print(_ind.max()+1)
    spline_x = (torch.arange(start=-1, end=_ind.max()+1, device=matrix.device, dtype=torch.float32) + unique_offset) * interval + 0.5 * interval
    # print("2")
    spline_cdf = torch.cumsum(spline_pmf, dim=0)

    query_x_idx = torch.searchsorted(spline_x, x)

    res_cdf_int = torch.ones_like(x).long()

    mask1 = query_x_idx - 1 < 0
    mask2 = (query_x_idx >= spline_cdf.shape[0])
    inter_mask = ~(mask1 | mask2)
    corr = spline_pmf.sum()
    res_cdf_int[mask1] = 0
    res_cdf_int[mask2] = corr

    barycentric = (x[inter_mask] - spline_x[query_x_idx[inter_mask] - 1]) / interval
    res_cdf_int[inter_mask] = spline_cdf[query_x_idx[inter_mask]-1]

    res_cdf_float = torch.zeros_like(x)
    res_cdf_float[inter_mask] = barycentric * spline_pmf[query_x_idx[inter_mask]].float()

    return res_cdf_int.view(x.shape), res_cdf_float.view(x.shape), corr


def get_cdf_new(matrix, x, resolution):
    linstat.resolution = resolution
    rescdf_i, rescdf_f = linstat(matrix, x)
    return rescdf_i, rescdf_f, torch.tensor(matrix.numel(), device=matrix.device)


def aun(matrix, delta):
    x_tilde = matrix + (torch.rand_like(matrix) - 0.5) * delta
    return x_tilde


def ste(matrix, scale):
    if torch.all(matrix==0):
        return torch.zeros_like(matrix)
    matrix = matrix / scale
    x_tile = torch.round(matrix) - matrix.detach() + matrix
    x_tile = x_tile * scale
    return x_tile


def quant(matrix, scale):
    if torch.all(matrix==0):
        return torch.zeros_like(matrix)
    return torch.round(matrix / scale) * scale


def get_bitrate_quant(matrix, x_tilde, delta, resolution):
    if torch.all(x_tilde==0):
        return torch.zeros_like(x_tilde)
    original_num = matrix.numel()
    if matrix.numel() > 128 * 128:
        sample_num = matrix.numel() // 256
        sample_idx = torch.randperm(matrix.numel())[:sample_num]
        matrix = matrix.flatten()[sample_idx]
        x_tilde = x_tilde.flatten()[sample_idx]
    probi1, probf1, corr1 = get_cdf(matrix, x_tilde+0.5*delta, resolution=resolution)
    probi2, probf2, corr2 = get_cdf(matrix, x_tilde-0.5*delta, resolution=resolution)
    prob = (probi1 - probi2).float() + (probf1 - probf2)
    if torch.sum(prob == 0):
        print("except", torch.sum(prob == 0).item())
        # x_tilde[prob == 0]
        # [0.0329, 0.0430]
        # 0.0367
        raise ValueError
        import pdb
        pdb.set_trace()
        # # get_cdf(matrix, torch.tensor(0.0573)[None].to(matrix), 10)
        # prob = get_cdf(matrix, x_tilde + 0.5 * delta, 10) - get_cdf(matrix, x_tilde - 0.5 * delta, 10)
    return (-(torch.log2(prob) - torch.log2(corr1.float()))).mean() * original_num


def get_bitrate_quant_new(matrix, x_tilde, delta, resolution):
    if torch.all(x_tilde==0):
        return torch.zeros_like(x_tilde)
    original_num = matrix.numel()
    if matrix.numel() > 128 * 128:
        sample_num = matrix.numel() // 128
        sample_idx = torch.randperm(matrix.numel())[:sample_num]
        matrix = matrix.flatten()[sample_idx]
        x_tilde = x_tilde.flatten()[sample_idx]
    probi1, probf1, corr1 = get_cdf_new(matrix, x_tilde+0.5*delta, resolution=resolution)
    probi2, probf2, corr2 = get_cdf_new(matrix, x_tilde-0.5*delta, resolution=resolution)
    prob = (probi1 - probi2).float() + (probf1 - probf2)
    if torch.sum(~torch.log2(prob).isfinite()):
        print("except", torch.sum(prob == 0).item())
        # x_tilde[prob == 0]
        # [0.0329, 0.0430]
        # 0.0367
        # raise ValueError
        import pdb
        pdb.set_trace()
        # # get_cdf(matrix, torch.tensor(0.0573)[None].to(matrix), 10)
        # prob = get_cdf(matrix, x_tilde + 0.5 * delta, 10) - get_cdf(matrix, x_tilde - 0.5 * delta, 10)
    return (-(torch.log2(prob) - torch.log2(corr1.float()))).mean() * original_num


def knn_soft_quant(matrix, centric, temprature=1, K=5):
    K = min(K, centric.shape[0])
    get_cdf(matrix, centric, resolution=10)
    # dists, idx, _ = knn_points(matrix.flatten()[None, :, None], centric[None, :, None], K=K)
    # dists = dists[0]
    # idx = idx[0]
    logits = -torch.abs(matrix.flatten()[:, None] - centric[idx])
    soft_one_hot = F.softmax(logits, dim=-1)
    # soft_one_hot = F.gumbel_softmax(logits, tau=temprature, hard=False, eps=1e-10, dim=-1)

    scatter_ind = idx.flatten()
    scatter_value = soft_one_hot.flatten()
    zero_soft_cnt = torch.zeros_like(centric)
    prob_per_centric = torch.scatter_add(zero_soft_cnt, 0, scatter_ind, scatter_value)
    prob_per_centric = prob_per_centric / prob_per_centric.sum()
    bitrate_per_centric = -(torch.log2(prob_per_centric))
    # scatter_value[scatter_ind==1].sum()
    quant_res = torch.sum(soft_one_hot * centric[idx], dim=1, keepdim=False).view(matrix.shape)
    bitrate = torch.sum(soft_one_hot * bitrate_per_centric[idx], dim=1, keepdim=False).view(matrix.shape)
    return quant_res, bitrate


def gumbel_soft_quant(matrix, centric, temprature=1):
    '''

    :param matrix: any shape
    :param centric: 1D tensor
    :return: quantization, entropy
    '''
    print(matrix.shape, centric.shape)
    logits = -distance_metric(matrix.flatten(), centric)
    soft_one_hot = F.gumbel_softmax(logits, tau=temprature, hard=False, eps=1e-10, dim=-1)
    prob_per_centric = torch.sum(soft_one_hot, dim=0, keepdim=True)
    prob_per_centric = prob_per_centric / prob_per_centric.sum()
    bitrate_per_centric = -(torch.log2(prob_per_centric))
    quant_res = torch.sum(soft_one_hot * centric[None, :], dim=1, keepdim=False).view(matrix.shape)
    return quant_res, soft_one_hot * bitrate_per_centric






