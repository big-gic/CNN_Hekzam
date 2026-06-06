import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

import random

import time

from pathlib import Path

from datetime import timedelta











class Net(nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
        self.conf = conf
        self.net = nn.ModuleList()
        
        layers = self.conf["layers"]

        for layer in layers:
            if layer["type"] == "conv2d":
                dilation = 1
                padding = 0
                if (layer.get("dilation") != None):
                    dilation = layer["dilation"]
                if (layer.get("padding") != None):
                    padding = layer["padding"]
                self.net.append(nn.Conv2d(layer["input_channels"], layer["output_channels"], layer["kernel_size"], dilation=dilation, padding=padding).to(self.conf["device"]))

            elif layer["type"] == "fc":
                self.net.append(nn.Linear(layer["input_channels"], layer["output_channels"]).to(self.conf["device"]))
            
            elif layer["type"] == "dropout":
                self.net.append(nn.Dropout(layer["p"])).to(self.conf["device"])

            elif layer["type"] == "bn2d":
                self.net.append(nn.BatchNorm2d(layer["input_channels"]))

    def forward(self, x):
        layers = self.conf["layers"]
        i = 0
        for layer in layers:
            if layer["type"] == "conv2d":
                x = self.net[i](x)
                i = i + 1
            
            elif layer["type"] == "avg_pool2d":
                x = F.avg_pool2d(x, (2, 2))

            elif layer["type"] == "max_pool2d":
                x = F.max_pool2d(x, (2, 2))
            
            elif layer["type"] == "fc":
                # si x contient des images (n'est pas un vecteur)
                if len(x.shape) > 2:
                    x = x.view(-1, layer["input_channels"])
                x = self.net[i](x)
                i = i + 1

            elif layer["type"] == "dropout":
                x = self.net[i](x)
                i = i + 1

            elif layer["type"] == "bn2d":
                x = self.net[i](x)
                i = i + 1

            # activation function (optional)
            act = layer.get("activation")
            if act == "tanh":
                x = torch.tanh(x)
            elif act == "relu":
                x = F.relu(x)
            elif act == "softmax":
                x = F.softmax(x, dim=1)

        return x

    def display(self):
        print(self)
        return
    


class Model :

    def __init__(self, conf):
        self.conf = conf
        self.net = None
        self.results = {}


    def build_network_time(self):
        before_network = time.perf_counter()
        self.net = Net(self.conf)
        after_network = time.perf_counter()
        self.results["net_name"] = self.conf["net_name"]
        self.results["network_time"] = after_network - before_network