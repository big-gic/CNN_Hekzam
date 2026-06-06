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
        plt.show()




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
        for trans in data_transforms :
            if trans["type"] == "affine":
                trans["degree"], trans["translate"]
                angle = random.uniform(-trans["degree"], trans["degree"])
                dx = random.uniform(-trans["translate"][0], trans["translate"][0]) * img.size[0]
                dy = random.uniform(-trans["translate"][1], trans["translate"][1]) * img.size[1]
                # Transformation affine déterminée
                img = TF.affine(img, angle=angle, translate=(int(dx), int(dy)), scale=1.0, shear=0)
        
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
        plt.show()




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

        # initialisation des paramètres d'apprentissage
        criterion, optimizer, scheduler = self.create_learning_param()
        
        best_eval_loss = float("inf")
        patience_counter = 0
        num_epochs = self.conf["num_epochs"]
        # loop over the dataset multiple times
        for epoch in range(num_epochs):

            print(f"Epoch {epoch+1}/{num_epochs} | Patience : {patience_counter}")

            train_loss = self.train_one_epoch(criterion, optimizer, scheduler)

            # ====== EARLY STOPPING ======
            if (self.conf["early_stopping"]):
                eval_loss, accuracy = self.evaluate()
                print(f"Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")
                
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
   


class Evaluator:

    def __init__(self, net, test_dataset, conf):
        
        self.conf = conf
        self.net = net.to(conf["device"])
        self.dataset = test_dataset
        self.accuracy = None
        

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
        self.confusion_matrix = confusion_matrix(y_true, y_pred).tolist()
        print(f"Finished Testing on {total} samples")
        return total_forward_time, total_load_time

    
    def display(self):
        print(f"Accuracy : {self.accuracy} %")
        forward_time, load_time = self.evaluate_time()
        print(f"Forward time : {forward_time} | Load time : {load_time}")
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



class Experiment:

    def __init__(self, conf):
        self.conf = conf
        self.train_dataset = None
        self.test_dataset = None
        self.net = None
        self.trainer = None
        self.evaluator = None
        self.results = {}


    def set_global_seed(self):
        torch.manual_seed(self.conf["seed"])


    def prepare_train_data_time(self):
        before_train_data = time.perf_counter()
        self.train_dataset = TrainDataset(self.conf)
        self.train_dataset.preprocess_dynamic()
        after_train_data = time.perf_counter()
        self.results["train_data_time"] = after_train_data - before_train_data


    def build_network_time(self):
        before_network = time.perf_counter()
        self.net = Net(self.conf)
        after_network = time.perf_counter()
        self.results["net_name"] = self.conf["net_name"]
        self.results["network_time"] = after_network - before_network


    def train_time(self):
        before_train = time.perf_counter()
        self.trainer = Trainer(self.net, self.train_dataset, self.conf)
        epoch, lr = self.trainer.train()
        self.results["epoch"], self.results["lr"] = epoch, str(self.conf["optimizer"]["lr"]) +"-"+ str(lr)
        after_train = time.perf_counter()
        self.results["train_time"] = after_train - before_train


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
        self.results["confusion_matrix"] = self.evaluator.confusion_matrix
    

    def display_train_data(self):
        self.train_dataset.display()
    

    def display_network(self):
        self.net.display()


    def display_test_data(self):
        self.test_dataset.display()


    def display_test(self):
        self.evaluator.display()


    def save_net(self):
        torch.save(self.net.state_dict(), "created_files/"+self.conf["save_name"]+".pt")


    def load_params(self):
        params = pd.read_json("created_files/"+ self.conf["net_name"] +".json")
        self.conf["data_transforms"] = params.at[0, "data_transformations"]
        self.conf["layers"] = params.at[0, "layer_list"]
    
    def load_net(self):
        self.build_network_time()
        self.net.load_state_dict(torch.load("created_files/"+self.conf["save_name"]+".pt"))


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

    
    def save_params_to_json(self):
        path = Path("created_files/"+self.conf["save_name"]+".json")
        if path.exists():
            df = pd.read_json(str(path))
        else:
            print("Création d'un nouveau fichier")
            df = pd.DataFrame()
        df.at[0, "net_name"] = self.conf["net_name"]
        df.at[0, "layers"] = self.conf["layers"]
        df.at[0, "loss_function"] = self.conf["loss_function"]["type"]
        df.at[0, "optimizer"] = self.conf["optimizer"]["type"]
        df.at[0, "train_samples"] = len(self.train_dataset.train_loader.dataset)
        df.at[0, "test_samples"] = len(self.test_dataset.data_loader.dataset)
        df.at[0, "data_transformations"] = self.conf["data_transforms"]
        df.to_json(str(path), orient="records", indent=4)
        print("parameters saved in "+str(path))


    def save_results_to_json(self):
        path = Path("created_files/"+self.conf["save_name"]+".json")
        if path.exists():
            df = pd.read_json(str(path))
        else:
            print("Création d'un nouveau fichier")
            df = pd.DataFrame()
        dict = df.to_dict()
        dict = dict | [self.results]
        df = pd.Dataframe(dict)
        df.to_json(str(path), orient="records", indent=4)
        print("results saved in "+str(path))




device = torch.device('cpu')



conf = {

    # device configuration (CPU)
    "device": device,

    # seed configuration (for data shuffle and weights initialisation)
    "seed": 31,

    # json configuration
    "save_name": "",
    
    # data configuration
    "dataset": "mnist",
    "normalization": True,
    "data_transforms": [],

    # net configuration
    "net_name": "",
    "layers": [
        {"type": "conv2d", "input_channels": 1, "output_channels": 6, "kernel_size": 5, "activation": "relu"},
        {"type": "conv2d", "input_channels": 6, "output_channels": 12, "kernel_size": 5, "activation": "relu"},
        {"type": "max_pool2d", "kernel_size": 2},
        {"type": "dropout", "p": 0.25},
        {"type": "fc", "input_channels": 12*12*12, "output_channels": 90, "activation": "relu"},
        {"type": "dropout", "p": 0.25},
        {"type": "fc", "input_channels": 90, "output_channels": 10}
    ],
    
    # train configuration
    "train_data_transforms": [
        #{"type": "affine", "degree": 35, "translate" : (0.21, 0.21)}
    ],
    "loss_function": {"type": "cross_entropy"},
    "optimizer": {"type": "sgd", "lr": 0.005, "momentum": 0.9},
    "scheduler": {"type": "steplr", "step_size": 5, "gamma": 1},
    "num_epochs": 30,
    "batch_size": 32,
    "early_stopping": True,
    "patience": 5,
    
    # test configuration
    "test_data_transforms": [
        #{"type": "affine", "degree": 35, "translate" : (0.21, 0.21)}
    ],
}