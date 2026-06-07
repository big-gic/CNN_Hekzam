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















class TestDataset :

    def __init__(self, conf):
        self.conf = conf
        self.data_loader = None


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
        
        
    def transform_image(self, img, data_transforms):
        for trans in data_transforms:
            if trans["type"] == "elastic":
                img = transforms.ElasticTransform()(img)

            elif trans["type"] == "rotation":
                angle = random.uniform(-trans["degree"], trans["degree"])
                img = TF.rotate(img, angle)

            elif trans["type"] == "affine":
                angle = random.uniform(-trans["degree"], trans["degree"])
                dx = random.uniform(-trans["translate"][0], trans["translate"][0]) * img.size[0]
                dy = random.uniform(-trans["translate"][1], trans["translate"][1]) * img.size[1]
                img = TF.affine(img, angle=angle, translate=(int(dx), int(dy)), scale=1.0, shear=0)

            elif trans["type"] == "pad":
                img = TF.pad(img, trans["padding"])

            elif trans["type"] == "resize":
                img = TF.resize(img, trans["size"])
        
        return img


    def preprocess_static(self):

        transforms_list = []
        transforms_list += self.create_transforms(self.conf["data_transforms"])

        transforms_list.append(transforms.ToTensor())
               
        if self.conf["normalization"] == True:
            transforms_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        post_transform = transforms.Compose(transforms_list)

        test_data_transforms = self.conf["test_data_transforms"]
        for trans in test_data_transforms:
            if trans["type"] == "elastic":
                print("Elastic transformation applied.")
            if trans["type"] == "rotation":
                print("Rotation transformation applied.")
            if trans["type"] == "affine":
                print("Affine transformation applied.")
            if trans["type"] == "pad":
                print("Padding transformation applied.")
            if trans["type"] == "resize":
                print("Resize transformation applied.")

        dataset = datasets.MNIST(root='./data', train=False, download=True)

        transformed_images = []
        labels = []
        for img, label in dataset:
            transformed_img = self.transform_image(img, test_data_transforms)

            transformed_img = post_transform(transformed_img)

            transformed_images.append(transformed_img)
            labels.append(label)
        
        # TensorDataset fixe
        X_test_aug = torch.stack(transformed_images)
        y_test_aug = torch.tensor(labels)

        test_aug_dataset = TensorDataset(X_test_aug, y_test_aug)
        self.data_loader = DataLoader(test_aug_dataset, batch_size=64, shuffle=False)
    

    def display(self):
        _, axes = plt.subplots(1, 20, figsize=(36, 2))
        for i in range(20):
            image, label = self.data_loader.dataset[i]
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(f'Label: {label}')
        Path("figures").mkdir(exist_ok=True)
        plt.savefig("figures/test_samples.png", bbox_inches='tight')
        plt.close()
        return















class Evaluator:

    def __init__(self, net, test_dataset, conf):
        
        self.conf = conf
        self.net = net.to(conf["device"])
        self.dataset = test_dataset
        self.accuracy = None
        self.confusion_matrix = None
        

    def evaluate_time(self):

        # ensure model in evaluation mode (disables Dropout, sets BatchNorm to eval)
        self.net.eval()
        
        total = 0
        correct = 0
        y_true = []
        y_pred = []

        # accumulate separate timings
        total_load_time = 0.0
        total_forward_time = 0.0

        data_iter = iter(self.dataset.data_loader)
        
        # désactivation du calcul des gradients
        
        with torch.no_grad():
            while True:
                # --- DATA LOADING ---
                load_start = time.perf_counter()
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    break
                images = images.to(self.conf["device"])
                labels = labels.to(self.conf["device"])
                load_end = time.perf_counter()

                total_load_time += (load_end - load_start)

                # --- FORWARD ---
                forward_start = time.perf_counter()
                outputs = self.net(images)
                forward_end = time.perf_counter()

                total_forward_time += (forward_end - forward_start)

                total += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        print(f"Finished Testing on {total} samples")
        return total_forward_time, total_load_time

    
    def display(self):
        self.plot_matrix()


    def plot_matrix(self):
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=['0','1','2','3','4','5','6','7','8','9'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=15, pad=20)
        plt.xlabel('Prediction', fontsize=11)
        plt.ylabel('Actual', fontsize=11)
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()
        plt.gca().figure.subplots_adjust(bottom=0.2)
        plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
        Path("figures").mkdir(exist_ok=True)
        plt.savefig("figures/confusion_matrix.png")
        plt.close()
        return











class Test:

    def __init__(self, conf):
        self.conf = conf
        self.test_dataset = None
        self.net = None
        self.evaluator = None
        self.results = {}


    def set_global_seed(self):
        torch.manual_seed(self.conf["seed"])


    def build_network_time(self):
        before_network = time.perf_counter()
        self.net = Net(self.conf)
        after_network = time.perf_counter()
        self.results["network_time"] = after_network - before_network


    def prepare_test_data_time(self):
        before_test_data = time.perf_counter()
        self.test_dataset = TestDataset(self.conf)
        self.test_dataset.preprocess_static()
        after_test_data = time.perf_counter()
        self.results["test_data_time"] = after_test_data - before_test_data


    def test_time(self):
        self.evaluator = Evaluator(self.net, self.test_dataset, self.conf)
        forward_time, load_time = self.evaluator.evaluate_time()
        self.results["forward_time"] = forward_time
        self.results["load_time"] = load_time
        self.results["accuracy"] = self.evaluator.accuracy
        self.results["confusion_matrix"] = self.evaluator.confusion_matrix.tolist()
    

    def display_network(self):
        self.net.display()


    def display_test_data(self):
        self.test_dataset.display()


    def display_test(self):
        self.evaluator.display()

    
    def load_params(self):
        params = pd.read_json(self.conf["config_path"])
        self.conf["data_transforms"] = params.at[0, "data_transformations"]
        self.conf["layers"] = params.at[0, "layer_list"]

    
    def load_net(self):
        self.build_network_time()
        self.net.load_state_dict(torch.load(self.conf["params_path"]))


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
        self.load_net()
        self.display_network()
        self.prepare_test_data_time()
        self.display_test_data()
        self.test_time()
        self.display_test()
        self.add_results_to_csv()
