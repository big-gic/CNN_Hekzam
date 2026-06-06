import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
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




class TrainDataset:
    
    def __init__(self, conf):
        self.conf = conf
        self.train_loader = None
        self.val_loader = None


    def create_transforms(self, data_transforms):
        transforms_list = []
        for trans in data_transforms:
            if trans["type"] == "elastic":
                transforms_list.append(transforms.ElasticTransform())
                print("Elastic transformation applied.")
            if trans["type"] == "rotation":
                transforms_list.append(transforms.RandomRotation(trans["degree"]))
                print("Rotation transformation applied.")
            if trans["type"] == "affine":
                transforms_list.append(transforms.RandomAffine(trans["degree"], trans["translate"]))
                print("Affine transformation applied.")
            if trans["type"] == "pad":
                transforms_list.append(transforms.Pad(trans["padding"]))
                print("Padding transformation applied.")
            if trans["type"] == "resize":
                transforms_list.append(transforms.Resize(trans["size"]))
                print("Resize transformation applied.")
        return transforms_list
        

    def preprocess_dynamic(self):

        transforms_list = []
        transforms_list += self.create_transforms(self.conf["train_data_transforms"])
        transforms_list += self.create_transforms(self.conf["data_transforms"])

        transforms_list.append(transforms.ToTensor())
               
        if self.conf["normalization"] == True:
            transforms_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        transform = transforms.Compose(transforms_list)

        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        g = torch.Generator()
        g.manual_seed(self.conf["seed"])
        train_dataset, val_dataset = random_split(dataset, [54000, 6000], generator=g)
        self.train_loader = DataLoader(train_dataset, batch_size=self.conf["batch_size"], shuffle=True, generator=g)
        self.val_loader = DataLoader(val_dataset, batch_size=self.conf["batch_size"], shuffle=False)

        
    def display(self):
        _, axes = plt.subplots(1, 20, figsize=(36, 2))
        for i in range(20):
            image, label = self.train_loader.dataset[i]
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(f'Label: {label}')
        Path("figures").mkdir(exist_ok=True)
        plt.savefig("figures/train_samples.png", bbox_inches='tight')
        plt.close()
        return










class Trainer :

    def __init__(self, net, train_dataset, conf):
        self.conf = conf
        self.net = net.to(conf["device"])
        self.dataset = train_dataset
        self.accuracy = None


    def create_learning_param(self):
        criterion = None
        optimizer = None
        scheduler = None
        if self.conf["loss_function"]["type"] == "cross_entropy":
            criterion = nn.CrossEntropyLoss()

        if self.conf["optimizer"]["type"] == "sgd":
            optimizer = optim.SGD(self.net.parameters(), lr=self.conf["optimizer"]["lr"], momentum=self.conf["optimizer"]["momentum"])

        if self.conf["optimizer"]["type"] == "adam":
            optimizer = optim.Adam(self.net.parameters(), lr=self.conf["optimizer"]["lr"])

        if self.conf["scheduler"]["type"] == "steplr":
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.conf["scheduler"]["step_size"], gamma=self.conf["scheduler"]["gamma"])

        print(f'criterion : {self.conf["loss_function"]["type"]} | optimizer : {self.conf["optimizer"]["type"]} | scheduler : {self.conf["scheduler"]["type"]}')

        return criterion, optimizer, scheduler
    

    def train_one_epoch(self, criterion, optimizer, scheduler):

        self.net.train()

        total_loss = 0.0    

        # pour chaque batch
        for i, (inputs, labels) in enumerate(self.dataset.train_loader):
            # get the inputs
            inputs, labels = inputs.to(self.conf["device"]), labels.to(self.conf["device"])

            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            # calcul des gradients
            loss.backward()
            # ajustement des poids en fonction des gradients
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            if i % 500 == 0 :
                print(f"Batch {i + 1} | LR : {optimizer.param_groups[0]['lr']} | Loss: {loss.item():.4f} ")

        scheduler.step()

        return total_loss / len(self.dataset.train_loader.dataset)
    

    def train(self):

        Path("model_params").mkdir(exist_ok=True)
        torch.save(self.net.state_dict(), "best_model.pt")

        # initialisation des paramètres d'apprentissage
        criterion, optimizer, scheduler = self.create_learning_param()
        
        best_eval_loss = float("inf")
        patience_counter = 0
        num_epochs = self.conf["num_epochs"]
        # loop over the dataset multiple times
        for epoch in range(num_epochs):

            print(f"Epoch {epoch+1}/{num_epochs} | Patience_counter : {patience_counter}")

            train_loss = self.train_one_epoch(criterion, optimizer, scheduler)

            eval_loss, accuracy = self.evaluate()
            print(f"Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")

            # ====== EARLY STOPPING ======
            if (self.conf["early_stopping"]):

                if eval_loss - 10**(-4) < best_eval_loss :
                    best_eval_loss = eval_loss
                    patience_counter = 0

                    # sauvegarde du meilleur modèle
                    print(f"sauvegarde (accuracy sur dataset de validation : {accuracy:.4f} %)")
                    torch.save(self.net.state_dict(), "best_model.pt")
                else:
                    patience_counter += 1

                if patience_counter >= self.conf["patience"]:
                    print("Early stopping déclenché.")
                    print(f"Arrêt à Epoch {epoch + 1}")
                    break

        # recharger le meilleur modèle
        self.net.load_state_dict(torch.load("best_model.pt"))
        print(f"Finished Training on {len(self.dataset.train_loader.dataset)} samples")
        return epoch, optimizer.param_groups[0]['lr']


    def evaluate(self):

        # ensure model in evaluation mode (disables Dropout, sets BatchNorm to eval)
        self.net.eval()

        criterion,_,_ = self.create_learning_param()

        correct = 0
        total = 0
        eval_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.dataset.val_loader:
                inputs, labels = inputs.to(self.conf["device"]), labels.to(self.conf["device"])

                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

                eval_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        eval_loss /= len(self.dataset.train_loader.dataset)
        accuracy = 100 * correct / total
        self.accuracy = accuracy
        print(f"Finished Evaluating on {total} samples")
        return eval_loss, accuracy
   











class Train:

    def __init__(self, conf):
        self.conf = conf
        self.train_dataset = None
        self.net = None
        self.trainer = None
        self.results = {}


    def set_global_seed(self):
        torch.manual_seed(self.conf["seed"])


    def prepare_train_data_time(self):
        before_train_data = time.perf_counter()
        self.train_dataset = TrainDataset(self.conf)
        self.train_dataset.preprocess_dynamic()
        after_train_data = time.perf_counter()
        self.results["train_data_time"] = after_train_data - before_train_data


    def train_time(self):
        before_train = time.perf_counter()
        self.trainer = Trainer(self.net, self.train_dataset, self.conf)
        epoch, lr = self.trainer.train()
        self.results["epoch"], self.results["lr"] = epoch, str(self.conf["optimizer"]["lr"]) +"-"+ str(lr)
        after_train = time.perf_counter()
        self.results["train_time"] = after_train - before_train
    

    def display_train_data(self):
        self.train_dataset.display()
    
    def build_network_time(self):
        before_network = time.perf_counter()
        self.net = Net(self.conf)
        after_network = time.perf_counter()
        self.results["network_time"] = after_network - before_network


    def display_network(self):
        self.net.display()


    def save_net(self):
        Path("model_params").mkdir(exist_ok=True)
        torch.save(self.net.state_dict(), "model_params/"+self.conf["save_name"]+".pt")


    def load_params(self):
        params = pd.read_json(self.conf["config"])
        self.conf["data_transforms"] = params.at[0, "data_transformations"]
        self.conf["layers"] = params.at[0, "layer_list"]


    def add_results_to_csv(self):
        path = Path("results.csv")
        if path.exists():
            df = pd.read_csv(str(path))
        else:
            print("Création d'un nouveau fichier")
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([self.results])], ignore_index=True)
        df.to_csv(str(path), index=False)
        print("row added to "+str(path))


    def run(self):
        self.set_global_seed()
        self.load_params()
        self.prepare_train_data_time()
        self.display_train_data()
        self.build_network_time()
        self.display_network()
        self.train_time()
        self.save_net()
        self.add_results_to_csv()