import os

import torch
from torch.utils.cpp_extension import load


stat = load(name="stat",
            sources=[os.path.join(os.path.split(os.path.abspath(__file__))[0], "cos_stat.cpp"),
                     os.path.join(os.path.split(os.path.abspath(__file__))[0], "cos_stat.cu")])
print("stat module loaded")
class CosearStatFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix_flatten, queries_flatten, delta):
        matrix_sort_ind = torch.argsort(matrix_flatten)
        queries_sort_ind = torch.argsort(queries_flatten)
        # matrix_sort_ind = torch.arange(len(matrix_flatten), device=queries_flatten.device)
        # queries_sort_ind = torch.arange(len(matrix_flatten), device=queries_flatten.device)
        # batchsize, n, _ = xyz1.size()
        # _, m, _ = xyz2.size()
        sorted_matrix_flatten = matrix_flatten[matrix_sort_ind].contiguous()
        queries_flatten = queries_flatten.contiguous()
        # dist1 = torch.zeros(batchsize, n)
        # dist2 = torch.zeros(batchsize, m)
        #
        # idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        # idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        rescdf_i = torch.zeros(queries_flatten.shape[0], dtype=torch.int, device=queries_flatten.device)
        rescdf_f = torch.zeros(queries_flatten.shape[0], dtype=torch.float, device=queries_flatten.device)
        respdf_f = torch.zeros(queries_flatten.shape[0], dtype=torch.float, device=queries_flatten.device)
        if not queries_flatten.is_cuda:
            stat.forward(sorted_matrix_flatten, queries_flatten, delta if isinstance(delta, float) else delta.item(), rescdf_i, rescdf_f, respdf_f)
        else:
            stat.forward_cuda(sorted_matrix_flatten, queries_flatten, delta if isinstance(delta, float) else delta.item(), rescdf_i, rescdf_f, respdf_f)
        ctx.save_for_backward(matrix_flatten, queries_flatten, torch.tensor(delta, device=queries_flatten.device) if isinstance(delta, float) else delta, queries_sort_ind, respdf_f)
        return rescdf_i, rescdf_f

    @staticmethod
    def backward(ctx, grad_rescdf_i, grad_rescdf_f):
        matrix_flatten, queries_flatten, delta, queries_sort_ind, respdf_f = ctx.saved_tensors
        delta = delta.item()
        grad_matrix_f = torch.zeros(matrix_flatten.shape[0], dtype=torch.float, device=matrix_flatten.device)
        if not grad_rescdf_f.is_cuda:
            stat.backward(matrix_flatten.contiguous(), queries_flatten[queries_sort_ind].contiguous(), delta, grad_rescdf_f[queries_sort_ind].contiguous(), grad_matrix_f)
        else:
            stat.backward_cuda(matrix_flatten.contiguous(), queries_flatten[queries_sort_ind].contiguous(), delta,
                          grad_rescdf_f[queries_sort_ind].contiguous(), grad_matrix_f)

        grad_queries = grad_rescdf_f.flatten() * respdf_f
        # print("+++++++++++++++++++++++")
        # print(grad_matrix_f.sum())
        # print(grad_queries.sum())
        # print("-----------------------")

        return grad_matrix_f, grad_queries, None


class Cosear_Stat(torch.nn.Module):
    def __init__(self, delta=None, resolution=None):
        super().__init__()
        self.delta = delta
        self.resolution = resolution
    def forward(self, matrix, queries):
        if self.resolution:
            delta = ((matrix.max() - matrix.min()) / self.resolution).detach()
        else:
            delta = self.delta
        matrix_flatten = matrix.flatten()
        queries_flatten = queries.flatten()
        rescdf_i, rescdf_f = CosearStatFun.apply(matrix_flatten, queries_flatten, delta)
        return rescdf_i.view(queries.shape), rescdf_f.view(queries.shape)

if __name__ == '__main__':
    import argparse

    tmp = (torch.arange(10, dtype=torch.float)/2).cuda().requires_grad_(True)
    test = (torch.tensor([5.1])/2).cuda().requires_grad_(True)
    rescdf_i, rescdf_f = CosearStatFun.apply(tmp, test, 1.0/2)
    rescdf_f.sum().backward()
    import pdb
    pdb.set_trace()
