import os
import pickle
from compress.huffman import compress_matrix_flatten
from common.tools import get_np_size

log_dirs = [""]

for log_dir in log_dirs:
    for file in os.listdir(log_dir):
        try:
            if file.endswith(".log") and not file.endswith("range.log"):
                # print("start {}".format(file))
                dir_name = file[:-4]
                pkl_file = os.path.join(log_dir, dir_name, "test_state.pkl")
                dict = pickle.load(open(pkl_file, "rb"))
                max_epoch = None
                bit_sum_epoch = None
                bit_quant_epoch = None
                bit_sym_epoch = None
                value_dict = dict["special"]
                res_str = ""
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
                        bitrate, quant_symbol = compress_matrix_flatten(state['np_quant'][i])
                        bit_sym_epoch[i] += get_np_size(quant_symbol) * 8
                        bit_quant_epoch[i] += bitrate
                        bit_sum_epoch[i] += bitrate + get_np_size(quant_symbol) * 8

                origin_bit = value_dict["origin_bit"]
                for i in range(max_epoch):
                    tmp = "origin: {}KB, compress: {}KB, symbol: {}KB, CR: {}x \n".format(origin_bit / (8 * 1024),
                                                                                            bit_quant_epoch[i] / (8 * 1024),
                                                                                            bit_sym_epoch[i] / (8 * 1024),
                                                                                            origin_bit / (
                                                                                                        bit_quant_epoch[i] + bit_sym_epoch[i]))
                    res_str += tmp
                    # print(dir_name, tmp)
                with open(os.path.join(log_dir, dir_name + "_huffman.log"), "w") as f:
                    f.write(res_str)
                print("success", file)
        except:
            print("fail", file)
            continue