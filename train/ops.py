from torch import Tensor, FloatTensor
from torch import nn
from torch.nn.common_types import _size_2_t


class QConv2d(nn.Module):
    def __init__(
        self,
        conv: nn.Module
    ):
        self.scale = nn.Parameter(FloatTensor(1), requires_grad=True)

        max = conv.weight.max
        self.scale.data.fill_()

    def forward(self, input: Tensor) -> Tensor:
        sparse_weight = self.weight_fake_sparse(self.weight)
        return self._conv_forward(input, sparse_weight, self.bias)
