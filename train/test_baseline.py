import os.path

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from compress.huffman import compress_matrix_flatten as huffman_coder
from compress.range_coder import compress_matrix_flatten as range_coder

from common.tools import get_np_size, to_np
from train.logger import Logger
import math
from transform.ms import MS_T
from transform.edge_scale import EdgeScale_T
from transform.scale import Scale_T
from transform.log import Log_T
from transform.exp import Exp_T
import pickle
import copy
import time
from clib.lin_stat.lin_stat import Linear_Stat
from clib.cos_stat.cos_stat import Cosear_Stat
from clib.tri_stat.tri_stat import Triear_Stat
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import resnet18, resnet50, mobilenetv2, mnasnet, regnetx_600m, regnetx_3200m

ROOTDIR = ""
METADIR = ""

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def get_correct(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

train_transform = T.Compose(
    [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
)

val_transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
)

logger = None
device = None
FLAGS = None

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "mobilenetv2": mobilenetv2,
    "mnasnet": mnasnet,
    "regnetx_600m": regnetx_600m,
    "regnetx_3200m": regnetx_3200m
}
state_path = {
    "resnet18": "",
    "resnet50": "",
    "mobilenetv2": "",
    "mnasnet": "",
    "regnetx_600m": "",
    "regnetx_3200m": ""
}
kd_layer = {
    "resnet18": ['conv1', 'layer2.1.conv2', 'layer4.1.conv2', 'fc'],
    "resnet50": ['conv1', 'layer2.1.conv2', 'layer4.1.conv2', 'fc'],
    "mobilenetv2": ['features.0.2', 'features.1.conv.4', 'features.2.conv.7', 'features.3.conv.3', 'features.17.conv.7', 'features.18.2', 'classifier.1'],
    "mnasnet": ['layers.8.2.layers.7', 'layers.9.2.layers.7', 'layers.10.1.layers.7', 'layers.12.3.layers.7', 'classifier.1'],
    "regnetx_600m": ['stem.conv', 's2.b1.proj', 's3.b1.proj', 's4.b1.proj', 'head.fc'],
    "regnetx_3200m": ['stem.conv', 's2.b1.proj', 's3.b1.proj', 's4.b1.proj', 'head.fc']
}

trans_map = {
    "edgescale": EdgeScale_T,
    "scale": Scale_T,
    "multiscale": MS_T,
    "log": Log_T,
    "exp": Exp_T,
}
diffkernel_map = {
    'lin': Linear_Stat,
    'cos': Cosear_Stat,
    'tri': Triear_Stat,
}
def init_model(name):
    net = model_dict[name]()
    if name == "mobilenetv2":
        net_weight = torch.load(state_path[name], map_location=torch.device('cpu'))["model"]
    else:
        net_weight = torch.load(state_path[name], map_location=torch.device('cpu'))
    net.load_state_dict(net_weight, strict=True)
    return net



def ste(matrix):
    if torch.all(matrix==0):
        return torch.zeros_like(matrix)
    x_tile = torch.round(matrix) - matrix.detach() + matrix
    return x_tile

def quant(matrix):
    if torch.all(matrix==0):
        return torch.zeros_like(matrix)
    return torch.round(matrix)

class DiffCounter():
    def __init__(self, resolution):
        self.linstat = diffkernel_map[FLAGS.diffkernel](resolution=resolution)

    def update(self):
        pass
        # if self.linstat.resolution < 64:
        #     self.linstat.resolution += 2

    def cal_bitrate(self, matrix, x_tilde):
        # if torch.all(x_tilde == 0):
        #     return torch.zeros_like(x_tilde)
        original_num = matrix.numel()

        if matrix.numel() > 128 * 128:
            sample_num = matrix.numel() // 128
            sample_idx = torch.randperm(matrix.numel())[:sample_num]
            matrix = matrix.flatten()[sample_idx]
            x_tilde = x_tilde.flatten()[sample_idx]

        probi1, probf1 = self.linstat(matrix, x_tilde + 0.5)
        probi2, probf2 = self.linstat(matrix, x_tilde - 0.5)
        corr = torch.tensor(matrix.numel(), device=matrix.device)
        prob = (probi1 - probi2).float() + (probf1 - probf2)
        valid_mask = torch.log2(prob).isfinite()
        if torch.sum(~valid_mask):
            print("except", torch.sum(prob == 0).item())
            # x_tilde[prob == 0]
            # [0.0329, 0.0430]
            # 0.0367
            # raise ValueError
            # import pdb
            # pdb.set_trace()
            # # get_cdf(matrix, torch.tensor(0.0573)[None].to(matrix), 10)
            # prob = get_cdf(matrix, x_tilde + 0.5 * delta, 10) - get_cdf(matrix, x_tilde - 0.5 * delta, 10)
        return (-(torch.log2(prob) - torch.log2(corr.float())))[valid_mask].mean() * original_num



class Trainer():
    def __init__(self, model, trainloader, testloader):
        self.state_weight = {}
        self.state_bias = {}
        self.model = model
        self.teacher_model = copy.deepcopy(model)
        self.trainloader = trainloader
        self.testloader = testloader
        self.sum_iter = 0
        self.diffcounter = DiffCounter(FLAGS.resolution)
        self.weight_t = trans_map[FLAGS.weight_transform]
        self.bias_t = trans_map[FLAGS.bias_transform]
        self.kd_loss_fun = torch.nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.loss_r_his = []

        self._init_transform()
        self.t_features_out = {}
        self.s_features_out = {}
        self.layer_name = kd_layer[FLAGS.model_name]
        for item in self.layer_name:
            for (name, module) in self.teacher_model.named_modules():
                if name == item:
                    self._register_hooks(self.t_features_out, item, module)
            for (name, module) in self.model.named_modules():
                if name == item:
                    self._register_hooks(self.s_features_out, item, module)
            logger.log("register hook {}".format(item))


    def _register_hooks(self, output_map, layer_name, module):
        def hook(module, input, output):
            output_map[layer_name] = output
        module.register_forward_hook(hook=hook)

    def _set_trainable(self, param_list, trainable=True):
        for param in param_list:
            param.requires_grad_(trainable)

    def _get_param_list(self):
        res = []
        for name, m in self.state_weight.items():
            res.append(m["param"])
        for name, m in self.state_bias.items():
            res.append(m["param"])
        return res

    def _get_trans_param_list(self):
        res = []
        for name, m in self.state_weight.items():
            res.extend(self.weight_t.get_trainable_list(m["trans_param"]))
        for name, m in self.state_bias.items():
            res.extend(self.bias_t.get_trainable_list(m["trans_param"]))
        return res

    def _init_transform(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    param = m.weight.clone().detach().requires_grad_(False)
                    trans_param = self.weight_t.get_init_trans_param(param)
                    ori_bit = get_np_size(to_np(param)) * 8
                    dict = {"param": param, "trans_param": trans_param, "ori_bit": ori_bit}
                    self.state_weight[name] = dict
                if m.bias is not None:
                    param = m.bias.clone().detach().requires_grad_(False)
                    trans_param = self.bias_t. get_init_trans_param(param)
                    ori_bit = get_np_size(to_np(param)) * 8
                    dict = {"param": param, "trans_param": trans_param, "ori_bit": ori_bit}
                    self.state_bias[name] = dict
        self.origin_bit = self._get_state_value_sum("ori_bit")

    def _cal_transform(self, quant_method, stat_mode=''):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if m.weight is not None and name in self.state_weight:
                    dict = self.state_weight[name]
                    param = dict["param"]
                    trans_param = dict["trans_param"]
                    code = self.weight_t.encode_param(param, trans_param)
                    quant = quant_method(code)
                    self._stat_code(dict, code, quant, stat_mode)
                    reconstruction = self.weight_t.decode_param(quant, trans_param)
                    m.__dict__['weight'] = reconstruction
                if m.bias is not None and name in self.state_bias:
                    dict = self.state_bias[name]
                    param = dict["param"]
                    trans_param = dict["trans_param"]
                    code = self.bias_t.encode_param(param, trans_param)
                    quant = quant_method(code)
                    self._stat_code(dict, code, quant, stat_mode)
                    reconstruction = self.bias_t.decode_param(quant, trans_param)
                    m.__dict__['bias'] = reconstruction

    def _stat_code(self, dict, code, quant, stat_mode):
        if stat_mode == 'test':
            np_quant = to_np(quant)
            dict["np_quant"] = np_quant
            dict["test_bitrate"] = self.diffcounter.cal_bitrate(code, quant)
            unique, unique_indices, unique_inverse, unique_counts = np.unique(np_quant, return_index=True,
                                                                              return_inverse=True, return_counts=True)
            quant_symbol = unique
            # bitrate, quant_symbol = huffman_coder(to_np(quant))
            # dict["test_bitrate"] = bitrate
            dict["test_bitrate_symbol"] = get_np_size(quant_symbol) * 8
            dict["test_cnt_symbol"] = quant_symbol.shape[0]
        elif stat_mode == "train":
            dict["train_bitrate"] = self.diffcounter.cal_bitrate(code, quant)
        elif stat_mode == "exclude":
            return
        else:
            raise NotImplementedError


    def _get_state_value_sum(self, name):
        sum = 0
        for key, value in self.state_weight.items():
            sum += value[name]
        for key, value in self.state_bias.items():
            sum += value[name]
        return sum

    def _test(self):
        self._cal_transform(quant, "test")
        bitrate = self._get_state_value_sum("test_bitrate")
        bitrate_symbol = self._get_state_value_sum("test_bitrate_symbol")
        logger.log("origin: {}KB, compress: {}KB, symbol: {}KB, CR: {}x".format(self.origin_bit / (8 * 1024),
                                                                    bitrate / (8 * 1024),
                                                                    bitrate_symbol / (8 * 1024),
                                                                    self.origin_bit / (bitrate + bitrate_symbol)))
        logger.record_test_state(self.state_weight, "weight")
        logger.record_test_state(self.state_bias, "bias")
        logger.record_test_value("origin_bit", self.origin_bit)
        logger.save_test_state()
        self._test_acc()

    def _test_acc(self):
        total = 0
        correct_top1 = 0
        correct_top5 = 0
        self.model.eval()
        # last_time = time.time()
        for i, data in enumerate(self.testloader, 0):
            # if i % 500 == 0:
            #     now = time.time()
            #     print("[{}it,{:.2f}s/500iters]".format(i, now-last_time))
            #     last_time = now
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = self.model(inputs)
            tmp_top1, tmp_top5 = get_correct(outputs.data, labels, topk=(1, 5))
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # correct += (predicted == labels).sum()
            correct_top1 += tmp_top1.item()
            correct_top5 += tmp_top5.item()
        logger.log("test_acc_top1: {}%, test_acc_top5: {}%".format(correct_top1 / total * 100, correct_top5 / total * 100))


    def _train(self, max_iteration, optimizer, scheduler=None, epoch=None, loss_mode=''):
        self.model.train()
        total = 0
        correct = 0
        last_time = time.time()
        for i, data in enumerate(self.trainloader, 0):
            now = time.time()
            # update
            self.sum_iter += 1
            self.diffcounter.update()
            optimizer.zero_grad()

            # main
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            self._cal_transform(ste, stat_mode="train" if loss_mode == 'transform' else 'exclude')
            outputs = self.model(inputs)
            with torch.no_grad():
                teacher_out = self.teacher_model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            if loss_mode == 'transform':
                out_student = torch.flatten(self.s_features_out[self.layer_name[-1]], start_dim=1)
                out_teacher = torch.flatten(self.t_features_out[self.layer_name[-1]], start_dim=1)
                loss_kd = self.kd_loss_fun(out_student, out_teacher)

                loss_r = self._get_state_value_sum("train_bitrate")
                if loss_r.item() < self.origin_bit / FLAGS.target_CR:
                    lambda_r = FLAGS.lambda_r / 100
                else:
                    lambda_r = FLAGS.lambda_r
                loss = lambda_r * loss_r + FLAGS.lambda_kd * loss_kd
                logger.log("[{}it,{:.2f}s/iter][transform] loss_kd {:.5f}, loss_r: {:.2f}KB, acc: {:.3f}%, target:{:.2f}KB".format(self.sum_iter, now-last_time, loss_kd.item(), loss_r.item() / (8 * 1024),
                                                                       correct / total * 100, self.origin_bit / FLAGS.target_CR / (8 * 1024)))
            elif loss_mode == 'reconstruct':
                loss_kd = 0
                for name in self.layer_name:
                    out_student = torch.flatten(self.s_features_out[name], start_dim=1)
                    out_teacher = torch.flatten(self.t_features_out[name], start_dim=1)
                    loss_kd += self.kd_loss_fun(out_student, out_teacher)
                loss_d = self.criterion(outputs, labels)
                loss = loss_d + 1e6 * loss_kd
                logger.log("[{}it,{:.2f}s/iter][reconstruct] loss_d: {:.5f}, loss_kd: {:.5f}, acc: {:.3f}%".format(self.sum_iter, now-last_time, loss_d.item(), loss_kd.item(), correct / total * 100))
            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()
            logger.record_train_state(self.state_weight, "weight")
            logger.record_train_state(self.state_bias, "bias")
            if self.sum_iter % 100 == 0:
                logger.save_train_state()
            if self.sum_iter >= max_iteration:
                break
            last_time = now

    def train(self):
        self._test_acc()


def main1(args):
    from data_new.imagenet import ImageNetDataset
    import torch

    global logger
    global device
    global FLAGS
    FLAGS = args

    assert FLAGS.model_name in model_dict
    for k, v in sorted(vars(FLAGS).items()):
        print(k, '=', v)

    logger = Logger(os.path.abspath(__file__), FLAGS.log_path, "{}".format(FLAGS.run_name))

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        logger.log("WARNING: CPU only, this will be slow!")

    train_dataset = ImageNetDataset(ROOTDIR + 'train/', 'train.txt', train_transform)
    trainloader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    test_dataset = ImageNetDataset(ROOTDIR + 'val/', METADIR + 'val.txt', val_transform)
    testloader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    net = init_model(FLAGS.model_name)
    net.to(device)
    for name, m in net.named_modules():
        print(name)

    trainer = Trainer(net, trainloader, testloader)
    trainer.train()

# def main2(args):
#     from cifar10_models.resnet import resnet18
#     from cifar10_models.data import CIFAR10Data
#
#     global logger
#     global device
#     global FLAGS
#     FLAGS = args
#
#     logger = Logger(os.path.abspath(__file__), "./log", "{}_{}_{}".format(FLAGS.run_name, FLAGS.lambda_r, FLAGS.lambda_kd))
#
#     # Set the device
#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#     else:
#         device = torch.device("cpu")
#         logger.log("WARNING: CPU only, this will be slow!")
#
#     data = CIFAR10Data()
#     trainloader = data.train_dataloader()
#     testloader = data.test_dataloader()
#
#     net = resnet18(pretrained=True)
#     net.to(device)
#     for name, m in net.named_modules():
#         print(name)
#
#     trainer = Trainer(net, trainloader, testloader)
#     trainer.train()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Compression")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--lambda_r", type=float)
    parser.add_argument("--lambda_kd", type=float)
    parser.add_argument("--weight_transform", type=str)
    parser.add_argument("--bias_transform", type=str)
    parser.add_argument("--transform_iter", type=int)
    parser.add_argument("--transform_lr", type=float)
    parser.add_argument("--reconstruct_iter", type=int)
    parser.add_argument("--reconstruct_lr", type=float)
    parser.add_argument("--target_CR", type=float)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--diffkernel", type=str)
    parser.add_argument("--resolution", type=int)
    args = parser.parse_args()

    main1(args)