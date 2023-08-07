import os
import pickle
from compress.range_coder import compress_matrix_flatten, compress_matrix_flatten_new
from common.tools import get_np_size
import numpy as np
import constriction
import time
import torch

log_dirs = [""]

def encode(pkl_path):
    pkl_file = pkl_path
    dict = pickle.load(open(pkl_file, "rb"))
    max_epoch = None
    
    bit_sum_epoch = None
    bit_quant_epoch = None
    bit_sym_epoch = None
    value_dict = dict["special"]
    res_str = ""
    time_cost = 0
    for name, state in dict.items():
        # print(name)
        if name=="special":
            continue
        if not max_epoch:
            max_epoch = len(state['np_quant'])
            bit_sum_epoch = [0.0 for _ in range(max_epoch)]
            bit_quant_epoch = [0.0 for _ in range(max_epoch)]
            bit_sym_epoch = [0.0 for _ in range(max_epoch)]
        for i in range(max_epoch):
            quant_compressed, quant_symbol, probabilities = compress_matrix_flatten_new(state['np_quant'][i])
            bit_sym_epoch[i] += get_np_size(quant_symbol) * 8
            bit_quant_epoch[i] += get_np_size(quant_compressed) * 8
            bit_sum_epoch[i] += get_np_size(quant_compressed) * 8 + 3 * get_np_size(quant_symbol) * 8
            quant_compressed.tofile("compressed_" + name + ".bin")
            quant_symbol.tofile("symbol_" + name + ".bin")
            probabilities.tofile("probabilities_" + name + ".bin")
            
    origin_bit = value_dict["origin_bit"]
    for i in range(max_epoch):
        tmp = "origin: {}KB, compress: {}KB, symbol: {}KB, CR: {}x \n".format(origin_bit / (8 * 1024),
                                                                                bit_quant_epoch[i] / (8 * 1024),
                                                                                3 * bit_sym_epoch[i] / (8 * 1024),
                                                                                origin_bit / (bit_quant_epoch[i] + 3 * bit_sym_epoch[i]))
        res_str += tmp
    print(res_str)


def decode(model_path):
    dict = torch.load(model_path)
    new_dict = {}
    for name, value in dict.items():
        quant_compressed = np.fromfile("compressed_" + name + ".bin")
        quant_symbol = np.fromfile("symbol_" + name + ".bin")
        probabilities = np.fromfile("probabilities_" + name + ".bin")
        decoder = constriction.stream.queue.RangeDecoder(quant_compressed)
        probabilities_model = constriction.stream.model.Categorical(probabilities)
        decoded = decoder.decode(probabilities_model, value.size)
        decoded = quant_symbol[decoded]
        decoded = quant_symbol[decoded].reshape(value.shape)
        new_dict[name] = torch.from_numpy(decoded)
        

