import torch.nn as nn
import torch
from .resnet import BasicBlock, Bottleneck, resnet18, resnet50  # noqa: F401
from .regnet import ResBottleneckBlock, regnetx_600m, regnetx_3200m  # noqa: F401
from .mobilenetv2 import InvertedResidual, mobilenetv2  # noqa: F401
from .mnasnet import _InvertedResidual, mnasnet  # noqa: F401
from .vision_transformer import *  # noqa: F403, F401
from .vision_transformer import MultiHeadAttention, FeedForward, Encoder1DBlock, ViTEmbedding, ViTHead  # noqa: F401
# from qdrop.quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer   # noqa: F401


# class QuantBasicBlock(QuantizedBlock):
#     """
#     Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
#     """
#     def __init__(self, org_module: BasicBlock, w_qconfig, a_qconfig, qoutput=True):
#         super().__init__()
#         self.qoutput = qoutput
#         self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, w_qconfig, a_qconfig)
#         self.conv2 = QuantizedLayer(org_module.conv2, None, w_qconfig, a_qconfig, qoutput=False)
#         if org_module.downsample is None:
#             self.downsample = None
#         else:
#             self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
#         self.activation = org_module.relu2
#         if self.qoutput:
#             self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

#     def forward(self, x):
#         residual = x if self.downsample is None else self.downsample(x)
#         out = self.conv1_relu(x)
#         out = self.conv2(out)
#         out += residual
#         out = self.activation(out)
#         if self.qoutput:
#             out = self.block_post_act_fake_quantize(out)
#         return out


# class QuantBottleneck(QuantizedBlock):
#     """
#     Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
#     """
#     def __init__(self, org_module: Bottleneck, w_qconfig, a_qconfig, qoutput=True):
#         super().__init__()
#         self.qoutput = qoutput
#         self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, w_qconfig, a_qconfig)
#         self.conv2_relu = QuantizedLayer(org_module.conv2, org_module.relu2, w_qconfig, a_qconfig)
#         self.conv3 = QuantizedLayer(org_module.conv3, None, w_qconfig, a_qconfig, qoutput=False)

#         if org_module.downsample is None:
#             self.downsample = None
#         else:
#             self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
#         self.activation = org_module.relu3
#         if self.qoutput:
#             self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

#     def forward(self, x):
#         residual = x if self.downsample is None else self.downsample(x)
#         out = self.conv1_relu(x)
#         out = self.conv2_relu(out)
#         out = self.conv3(out)
#         out += residual
#         out = self.activation(out)
#         if self.qoutput:
#             out = self.block_post_act_fake_quantize(out)
#         return out


# class QuantResBottleneckBlock(QuantizedBlock):
#     """
#     Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
#     """
#     def __init__(self, org_module: ResBottleneckBlock, w_qconfig, a_qconfig, qoutput=True):
#         super().__init__()
#         self.qoutput = qoutput
#         self.conv1_relu = QuantizedLayer(org_module.f.a, org_module.f.a_relu, w_qconfig, a_qconfig)
#         self.conv2_relu = QuantizedLayer(org_module.f.b, org_module.f.b_relu, w_qconfig, a_qconfig)
#         self.conv3 = QuantizedLayer(org_module.f.c, None, w_qconfig, a_qconfig, qoutput=False)
#         if org_module.proj_block:
#             self.downsample = QuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig, qoutput=False)
#         else:
#             self.downsample = None
#         self.activation = org_module.relu
#         if self.qoutput:
#             self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

#     def forward(self, x):
#         residual = x if self.downsample is None else self.downsample(x)
#         out = self.conv1_relu(x)
#         out = self.conv2_relu(out)
#         out = self.conv3(out)
#         out += residual
#         out = self.activation(out)
#         if self.qoutput:
#             out = self.block_post_act_fake_quantize(out)
#         return out


# class QuantInvertedResidual(QuantizedBlock):
#     """
#     Implementation of Quantized Inverted Residual Block used in MobileNetV2.
#     Inverted Residual does not have activation function.
#     """
#     def __init__(self, org_module: InvertedResidual, w_qconfig, a_qconfig, qoutput=True):
#         super().__init__()
#         self.qoutput = qoutput
#         self.use_res_connect = org_module.use_res_connect
#         if org_module.expand_ratio == 1:
#             self.conv = nn.Sequential(
#                 QuantizedLayer(org_module.conv[0], org_module.conv[2], w_qconfig, a_qconfig),
#                 QuantizedLayer(org_module.conv[3], None, w_qconfig, a_qconfig, qoutput=False),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 QuantizedLayer(org_module.conv[0], org_module.conv[2], w_qconfig, a_qconfig),
#                 QuantizedLayer(org_module.conv[3], org_module.conv[5], w_qconfig, a_qconfig),
#                 QuantizedLayer(org_module.conv[6], None, w_qconfig, a_qconfig, qoutput=False),
#             )
#         if self.qoutput:
#             self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

#     def forward(self, x):
#         if self.use_res_connect:
#             out = x + self.conv(x)
#         else:
#             out = self.conv(x)
#         if self.qoutput:
#             out = self.block_post_act_fake_quantize(out)
#         return out


# class _QuantInvertedResidual(QuantizedBlock):
#     # mnasnet
#     def __init__(self, org_module: _InvertedResidual, w_qconfig, a_qconfig, qoutput=True):
#         super().__init__()
#         self.qoutput = qoutput
#         self.apply_residual = org_module.apply_residual
#         self.conv = nn.Sequential(
#             QuantizedLayer(org_module.layers[0], org_module.layers[2], w_qconfig, a_qconfig),
#             QuantizedLayer(org_module.layers[3], org_module.layers[5], w_qconfig, a_qconfig),
#             QuantizedLayer(org_module.layers[6], None, w_qconfig, a_qconfig, qoutput=False),
#         )
#         if self.qoutput:
#             self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

#     def forward(self, x):
#         if self.apply_residual:
#             out = x + self.conv(x)
#         else:
#             out = self.conv(x)
#         if self.qoutput:
#             out = self.block_post_act_fake_quantize(out)
#         return out


# class QuantViTEmbedding(QuantizedBlock):
#     # just remove the output quantization, because we do it after add and layernorm
#     def __init__(self, org_module: ViTEmbedding, w_qconfig, a_qconfig, qoutput=False):
#         super().__init__()
#         self.patch_embedding = QuantizedLayer(org_module.patch_embedding, None, w_qconfig, a_qconfig, qoutput=False)
#         self.cls_type = org_module.cls_type
#         if self.cls_type == 'token':
#             self.pos_embedding = org_module.pos_embedding
#             self.cls_token = org_module.cls_token
#         elif self.cls_type == 'gap':
#             self.pos_embedding = org_module.pos_embedding

#     def forward(self, x):
#         x = self.patch_embedding(x)
#         x = x.flatten(2).transpose(1, 2)
#         # x shape: [B, N, K]

#         if self.cls_type == 'token':
#             cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#             x = torch.cat((cls_tokens, x), dim=1)
#             # x shape: [B, N+1, K]

#         x += self.pos_embedding
#         return x


# class QuantViTHead(QuantizedBlock):
#     # just do the input quantization here, because there is layernorm and pool
#     def __init__(self, org_module: ViTEmbedding, w_qconfig, a_qconfig, qoutput=False):
#         super().__init__()
#         self.pool_post_act_fake_quantize = Quantizer(None, a_qconfig)
#         self.classifier = QuantizedLayer(org_module.classifier, None, w_qconfig, a_qconfig, qoutput=qoutput)
#         self.cls_type = org_module.cls_type

#     def forward(self, x):
#         if self.cls_type == 'token':
#             x = x[:, 0]
#         elif self.cls_type == 'gap':
#             x = torch.mean(x, dim=2, keepdim=False)
#         x = self.pool_post_act_fake_quantize(x)
#         return self.classifier(x)


# class QuantViTMHA(QuantizedBlock):

#     def __init__(self, org_module: MultiHeadAttention, w_qconfig, a_qconfig, qoutput=True):
#         super().__init__()
#         self.heads = org_module.heads
#         self.scale = org_module.scale
#         self.to_qkv = QuantizedLayer(org_module.to_qkv, None, w_qconfig, a_qconfig, qoutput=False)
#         self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
#         self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
#         self.attn_post_act_fake_quantize = Quantizer(None, a_qconfig)
#         self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)
#         self.context_post_act_fake_quantize = Quantizer(None, a_qconfig)
#         self.to_out = QuantizedLayer(org_module.to_out, None, w_qconfig, a_qconfig, qoutput=False)
#         self.qoutput = qoutput
#         if qoutput:
#             # in fact, not here, because there is also layernorm
#             self.attn_output_post_act_fake_quantize = Quantizer(None, a_qconfig)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         q = self.q_post_act_fake_quantize(q)
#         k = self.k_post_act_fake_quantize(k)
#         v = self.v_post_act_fake_quantize(v)
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_post_act_fake_quantize(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.context_post_act_fake_quantize(x)
#         x = self.to_out(x)
#         if self.qoutput:
#             x = self.attn_output_post_act_fake_quantize(x)
#         return x


# class QuantViTFFN(QuantizedBlock):
#     def __init__(self, org_module: FeedForward, w_qconfig, a_qconfig, qoutput=True):
#         super().__init__()
#         self.mlp1_act = QuantizedLayer(org_module.mlp1, org_module.act, w_qconfig, a_qconfig, True)
#         self.mlp2 = QuantizedLayer(org_module.mlp2, None, w_qconfig, a_qconfig, qoutput=False)
#         self.qoutput = qoutput
#         if qoutput:
#             self.mlp2_post_act_fake_quantize = Quantizer(None, a_qconfig)

#     def forward(self, x):
#         x = self.mlp1_act(x)
#         x = self.mlp2(x)
#         if self.qoutput:
#             x = self.mlp2_post_act_fake_quantize()
#         return x


# class QuantEncoder1DBlock(QuantizedBlock):
#     def __init__(self, org_module: Encoder1DBlock, w_qconfig, a_qconfig, qoutput=False):
#         super().__init__()
#         self.norm1 = org_module.norm1
#         self.norm1_post_act_fake_quantize = Quantizer(None, a_qconfig)
#         self.attention = QuantViTMHA(org_module.attention, w_qconfig, a_qconfig, False)
#         self.norm2 = org_module.norm2
#         self.norm2_post_act_fake_quantize = Quantizer(None, a_qconfig)
#         self.feedforward = QuantViTFFN(org_module.feedforward, w_qconfig, a_qconfig, False)

#     def forward(self, x):
#         residual = x
#         x = self.norm1(x)
#         x = self.norm1_post_act_fake_quantize(x)
#         x = self.attention(x)
#         x = x + residual
#         y = self.norm2(x)
#         y = self.norm2_post_act_fake_quantize(y)
#         y = self.feedforward(y)
#         y += x
#         return y


# specials = {
#     BasicBlock: QuantBasicBlock,
#     Bottleneck: QuantBottleneck,
#     ResBottleneckBlock: QuantResBottleneckBlock,
#     InvertedResidual: QuantInvertedResidual,
#     _InvertedResidual: _QuantInvertedResidual,
#     Encoder1DBlock: QuantEncoder1DBlock,
#     ViTHead: QuantViTHead,
#     ViTEmbedding: QuantViTEmbedding
# }


# def load_model(config):
#     config['kwargs'] = config.get('kwargs', dict())
#     model = eval(config['type'])(**config['kwargs'])
#     checkpoint = torch.load(config.path, map_location='cpu')
#     if config.type == 'mobilenetv2':
#         checkpoint = checkpoint['model']
#     model.load_state_dict(checkpoint)
#     return model
