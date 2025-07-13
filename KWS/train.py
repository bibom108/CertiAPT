from collections import ChainMap
import argparse
import os
import random
import sys

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import copy
import time

from tqdm import tqdm
import model as mod
from manage_audio import AudioPreprocessor
from trans import Transformation
from fadro import FADRO
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device("cuda")
my_t = 300


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

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()
    # print("{} accuracy: {:>5}, loss: {}".format(name, accuracy, loss), end=end)
    return accuracy.item()

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def train(config):
    output_dir = os.path.dirname(os.path.abspath(config["output_file"]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])
    if not config["no_cuda"]:
        model.init_weights_glorot()
        model = nn.DataParallel(model)
        model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()

    # option to use TF from mic
    tf = mod.my_get_tf_phyaug(my_t)

    # transformation function
    trans = Transformation(tf=tf, sigma=config['sigma'])
    trans.to(device)
    dro = FADRO(writer=writer, sigma=config['sigma'])
    
    train_set, _, _ = mod.SpeechDataset.splits(config)
    train_set.audio_files_to_wav()

    target_loader = None
    source_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True,
        num_workers = 8,
        collate_fn=train_set.collate_fn
    )

    max_train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True,
        num_workers = 8,
        collate_fn=None
    )
    
    # populate main_set
    main_set = mod.TensorDataset()

    alphas_final = [[] for i in range(5)]

    step_no = 0
    train_start = time.time()
    for epoch_idx in range(config["n_epochs"]):
        print("Start training .....")
        train_loader = target_loader if target_loader is not None else source_loader
        train_bar = tqdm(train_loader, total=len(train_loader))
        running_results = {
            "samples": 0,
            "loss": 0,
            "acc": 0,
            "nan_loss": 0
        }
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            if epoch_idx == 0:
                main_set.add(list(torch.unbind(inputs, dim=0)), list(torch.unbind(labels, dim=0)))
            
            optimizer.zero_grad()
            inputs = inputs.to(device)  # [64, 101, 40]
            labels = labels.to(device)
            inputs = Variable(inputs, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

            model.train()
            requires_grad_(model, True)

            scores = model(inputs)
            loss = criterion(scores, labels)
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
            else:
                running_results["nan_loss"] += 1

            # writer.add_scalar("Loss/train", loss.item(), step_no)
            running_results["samples"] += 1
            running_results["loss"] += loss.item()
            batch_size = labels.size(0)
            accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
            running_results["acc"] += accuracy
            train_bar.set_description(
                desc="[%d/%d] Loss: %f Acc: %f Nan Loss: %d"
                % (
                    epoch_idx,
                    config["n_epochs"],
                    running_results["loss"] / running_results["samples"],
                    running_results["acc"] / running_results["samples"],
                    running_results["nan_loss"],
                )
            )

            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])

        # START FA-DRO
        if epoch_idx in config["dro_list"]:
            print("Start FA-DRO .....")
            aug_x, aug_y = [], []
            model.eval()
            requires_grad_(model, False)
            trans.train()
            requires_grad_(trans, True)
            max_bar = tqdm(max_train_loader, total=len(max_train_loader))
            for _, (inputs, labels) in enumerate(max_bar):
                for input, index in dro.forward(inputs, labels, model, trans):
                    if not torch.isnan(input).any(): 
                        aug_x.append(input.detach().clone().cpu())
                        aug_y.append(labels.cpu())
                    alphas_final[index].append(trans.alphas[index].detach().cpu().clone())
            
            aug_x = list(torch.unbind(torch.cat(aug_x)))
            aug_y = list(torch.unbind(torch.cat(aug_y)))
            main_set.add(aug_x, aug_y)
            target_loader = data.DataLoader(
                main_set,
                batch_size=config["batch_size"],
                shuffle=True, 
                # drop_last=True,
                num_workers = 8
            )
        # END FA-DRO
        
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.module.save(config["output_file"])

    torch.save(alphas_final, f'model/alphas_final_{config["id"]}.pt')
    train_end = time.time()
    print("train ended at ",epoch_idx, "total training time ",(train_end-train_start)/3600,"hours")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    parser.add_argument("--sigma", default=0.5, type=float)
    parser.add_argument("--id", default="main", type=str)
    config, _ = parser.parse_known_args()

    model_name = f"{config.id}_sigma={config.sigma}.pt"
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", model_name)

    global_config = dict(no_cuda=False, n_epochs=50, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10, seed=0,
        use_nesterov=False, input_file="", output_file=output_file, dro_list=[0,10,20,30,40],
        cache_size=32768, momentum=0.9, weight_decay=0.00001, sigma=config.sigma, id = config.id
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
    train(config)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
