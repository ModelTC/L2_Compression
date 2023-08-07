import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from matplotlib import pyplot
import shutil
from common.tools import to_np
import pickle

import time


class Logger():
    def __init__(self, code_path, root, run_name=None):
        super().__init__()
        if run_name == None:
            run_name = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        else:
            run_name = run_name
        self.root = os.path.join(root, run_name)
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        # self.log_file = open(os.path.join(self.root, "log.txt"), "a")
        self.edge_dict = {}
        self.scale_dict = {}
        self.loss_dict = {}
        self.value_dict = {}
        self.history_state_train = {}
        self.history_state_test = {"special":{}}
        shutil.copy(code_path, self.root)

    def vis_hist(self, tensor_dict, title):
        '''

        Args:
                tensor_list: dict of (name: Tensors)
        '''
        np_list = [(t[0], t[1].detach().cpu().numpy().flatten()) for t in tensor_dict.items()]

        r_max = -99999999.0
        r_min = 99999999.0
        for name, ndarray in np_list:
            r_max = max(r_max, ndarray.max())
            r_min = min(r_min, ndarray.min())
        bins = np.linspace(r_min, r_max, 100)
        pyplot.cla()
        for name, ndarray in np_list:
            pyplot.hist(ndarray, bins, alpha=0.5, label=name)
        pyplot.legend(loc='upper right')
        pyplot.title(title)
        save_dir = os.path.join(self.root, "hist")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        pyplot.savefig(os.path.join(save_dir, "{}.png".format(title)))

    def record_edge(self, value, name):
        if name not in self.edge_dict:
            self.edge_dict[name] = []
        self.edge_dict[name].append(value.item())

    def record_scale(self, value, name):
        if name not in self.scale_dict:
            self.scale_dict[name] = []
        self.scale_dict[name].append(value.item())

    def record_value(self, value, name):
        if value is None:
            return
        if name not in self.value_dict:
            self.value_dict[name] = []
        self.value_dict[name].append(value.item())

    def record_loss(self, value, name):
        if name not in self.loss_dict:
            self.loss_dict[name] = []
        self.loss_dict[name].append(value.item())

    def log_curve_edge(self):
        save_dir = os.path.join(self.root, "curve")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        for name, data in self.edge_dict.items():
            pyplot.cla()
            x = np.arange(1, len(data) + 1)
            pyplot.plot(x, np.array(data), 'o-')
            pyplot.title(name)
            pyplot.savefig(os.path.join(save_dir, "{}_edge.png".format(name)))

    def log_curve_scale(self):
        save_dir = os.path.join(self.root, "curve")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        for name, data in self.scale_dict.items():
            pyplot.cla()
            x = np.arange(1, len(data) + 1)
            pyplot.plot(x, np.array(data), 'o-')
            pyplot.title(name)
            pyplot.savefig(os.path.join(save_dir, "{}_scale.png".format(name)))

    def log_curve_loss(self):
        save_dir = os.path.join(self.root, "curve")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        for name, data in self.loss_dict.items():
            pyplot.cla()
            x = np.arange(1, len(data) + 1)
            pyplot.plot(x, np.array(data), 'o-')
            pyplot.title(name)
            pyplot.savefig(os.path.join(save_dir, "{}_loss.png".format(name)))

    def log_curve_value(self):
        save_dir = os.path.join(self.root, "curve")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        for name, data in self.value_dict.items():
            pyplot.cla()
            x = np.arange(1, len(data) + 1)
            pyplot.plot(x, np.array(data), 'o-')
            pyplot.title(name)
            pyplot.savefig(os.path.join(save_dir, "{}_value.png".format(name)))

    def log(self, string):
        # self.log_file.write(string+"\n")
        # self.log_file.flush()
        print(string)

    def record_train_state(self, state, name):
        for key, substate in state.items():
            outkey = "{}_{}".format(name, key)
            if outkey not in self.history_state_train:
                self.history_state_train[outkey] = {}
            if "trans_param" not in self.history_state_train[outkey]:
                self.history_state_train[outkey]["trans_param"] = {}
            for tk, tv in substate["trans_param"].items():
                if tk not in self.history_state_train[outkey]["trans_param"]:
                    self.history_state_train[outkey]["trans_param"][tk] = []
                self.history_state_train[outkey]["trans_param"][tk].append(to_np(tv))

    def save_train_state(self):
        pickle.dump(self.history_state_train, open(os.path.join(self.root, "train_state.pkl"), "wb"))

    def save_test_state(self, ):
        pickle.dump(self.history_state_test, open(os.path.join(self.root, "test_state.pkl"), "wb"))

    def record_test_value(self, name, value):
        self.history_state_test["special"][name] = value

    def record_test_state(self, state, name):
        for key, substate in state.items():
            outkey = "{}_{}".format(name, key)
            if outkey not in self.history_state_test:
                self.history_state_test[outkey] = {}
            if "trans_param" not in self.history_state_test[outkey]:
                self.history_state_test[outkey]["trans_param"] = {}
            for tk, tv in substate["trans_param"].items():
                if tk not in self.history_state_test[outkey]["trans_param"]:
                    self.history_state_test[outkey]["trans_param"][tk] = []
                np_tv = to_np(tv)
                self.history_state_test[outkey]["trans_param"][tk].append(np_tv)
                self.log("[{}][{}]: {}".format(outkey, tk, np_tv))
            if "np_quant" not in self.history_state_test[outkey]:
                self.history_state_test[outkey]["np_quant"] = []
            self.history_state_test[outkey]["np_quant"].append(substate["np_quant"])
            self.log("[{}] symbols: {}, equal_bit: {}".format(outkey, substate["test_cnt_symbol"], math.log2(substate["test_cnt_symbol"])))



