from collections import ChainMap
import argparse
import os
import random
import sys
from collections import defaultdict
import shutil
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader, RandomSampler
import copy
import time
from sklearn.metrics import confusion_matrix
import model as mod
from core import Smooth
# from statsmodels.stats.proportion import proportion_confint

device = torch.device("cuda")

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def print_eval(name, scores, labels, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() #/ batch_size
    # print("{} accuracy: {:>5}".format(name, accuracy / batch_size))
    return accuracy.item()

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def get_ori_test_list(test_list):
    res = []
    for path in test_list:
        subdir = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        filename = os.path.join(subdir, filename) 
        res.append(filename)
    return res

def filter_test_list(test_list, label_list, ori_test_list):
    res = []
    res_label = []
    for i, path in enumerate(test_list):
        subdir = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        filename = os.path.join(subdir, filename) 
        if filename in ori_test_list:
            res.append(path)
            res_label.append(label_list[i])
    return res, res_label

words = {i+2: word for i, word in enumerate(["yes","no","left","right","up","down","on","off","stop","go"])}

def evaluate(config, ori_test_list, index, tf, alphas_final):
    _, _, test_set = mod.SpeechDataset.splits(config)
    test_set.audio_files, test_set.audio_labels = filter_test_list(test_set.audio_files, test_set.audio_labels, ori_test_list)
    test_set.audio_files_to_wav()
    
    test_loader = data.DataLoader(
        test_set,
        batch_size= 1,
        num_workers=8,
        # collate_fn=test_set.collate_fn
    )
    
    model = config["model_class"](config)
    model.load(config["output_file"])
    if not config["no_cuda"]:
        model.to(device)
    model.eval()

    smooth = Smooth(tf[index], alphas_final[index][:100], model, 12, config["sigma_test"])

    all_preds = []
    all_labels = []
    for model_in, labels in test_loader:
        if not config["no_cuda"]:
            model_in = model_in.to(device)
            labels = labels.to(device)

        preds = smooth.predict(model_in, 1000, 1000)
        # scores = model(model_in)
        # preds = torch.argmax(scores, dim=1)

        all_preds.append(int(preds))
        all_labels.extend(labels.cpu().numpy())
    
    conf_matrix = confusion_matrix(all_labels, all_preds)

    conf_matrix[0, :] = 0                                 
    accuracy = (np.trace(conf_matrix)) / (np.sum(conf_matrix))
    print(f"{accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    parser.add_argument("--sigma", default=0.5, type=float)
    parser.add_argument("--sigma_test", default=0.5, type=float)
    parser.add_argument("--id", default="main", type=str)
    config, _ = parser.parse_known_args()

    model_name = f"{config.id}_sigma={config.sigma}.pt"
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", model_name)

    global_config = dict(no_cuda=False, n_epochs=50, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10, seed=0,
        use_nesterov=False, input_file="", output_file=output_file, cache_size=32768, momentum=0.9, weight_decay=0.00001, 
        num_samples=100000, alpha = 0.001, id=config.id, sigma_test=config.sigma_test
    )
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()

    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)

    tf = mod.my_get_tf_phyaug(300)
    alphas_final = torch.load(f'model/alphas_final_{config["id"]}.pt', weights_only=False)

    config["data_folder"] = "/data2/phuc/speech/kws"
    _, _, test_set = mod.SpeechDataset.splits(config)
    ori_test_list = get_ori_test_list(test_set.audio_files)
    data_folder_list = [
        # "/data2/phuc/speech/kws",
        "data/recorded_dataset/ATR",
        "data/recorded_dataset/clipon",
        "data/recorded_dataset/maono",
        "data/recorded_dataset/USB",
        "data/recorded_dataset/USBplug",
    ]
    for index, data_folder in enumerate(data_folder_list):
        config["data_folder"] = data_folder
        evaluate(config, ori_test_list, index, tf, alphas_final)

if __name__ == "__main__":
    main()
