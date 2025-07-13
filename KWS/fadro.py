from enum import Enum
import hashlib
import math
import os
import random
import re
import time

from chainmap import ChainMap
from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pcen
import torch.optim as optim

device = torch.device("cuda")

class FADRO():
    def __init__(self, N=20, writer=None, sigma=0.5):
        self.sigma = sigma
        self.N = N
        self.lr = 0.1
        self.writer = writer
        self.tau = -0.1
        self.gamma = 1000
        self.beta = -1.0

    def requires_grad_(self, model:torch.nn.Module, index:int) -> None:
        for i, param in enumerate(model.parameters()):
            if i == index:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def forward(self, inputs, labels, model, trans):
        trans.reset_alpha()
        inputs_prime = inputs.data.clone().to(device)
        optimizer_alpha = torch.optim.SGD(trans.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer_alpha, step_size=10, gamma=0.5)
        class_criterion = nn.CrossEntropyLoss()
        semantic_distance_criterion = nn.MSELoss()
        res = []
        random.seed(time.perf_counter())
        index = random.randint(0,4)
        init_feature = None
        for n in range(self.N):
            optimizer_alpha.zero_grad()
            after_spec = trans(inputs_prime, index, use_noise=True)
            last_features = model(after_spec, get_feat=True)
            if n == 0:
                init_feature = last_features.clone()
            rho = semantic_distance_criterion(last_features, init_feature)  #E[c(Z,Z0)]
            init_feature = last_features.clone()
            
            class_output = model(after_spec)
            
            loss_zt = class_criterion(class_output, labels.to(device))

            tmp = (torch.exp(trans.alphas[index]) * trans.tf[index]).mean()
            loss_fre = 0.5 * ((1 - tmp) ** 2)

            loss_phi = self.tau * loss_zt + self.gamma * rho + self.beta * loss_fre
            loss_phi.backward(retain_graph=True)
            
            optimizer_alpha.step()
            # scheduler.step()

        # self.writer.add_scalar("Loss/max", loss_phi.item(), idx)
        res.append((after_spec, index))
        return res
