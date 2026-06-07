import torch

import argparse

from train import Train
from test import Test
from network import Network









parser = argparse.ArgumentParser(
    description="Entraînement et test de réseaux de neurones sur MNIST"
)

subparsers = parser.add_subparsers(
    dest="command",
    required=True
)

# =====================
# INFO NEURAL NETWORK
# =====================

info_parser = subparsers.add_parser(
    "info",
    help="Afficher les informations sur un réseau"
)

info_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Nom du réseau"
)


# =====================
# TRAIN
# =====================

train_parser = subparsers.add_parser(
    "train",
    help="Entraîner un réseau"
)

train_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Nom du réseau"
)

train_parser.add_argument(
    "--params_path",
    type=str,
    required=True
)

train_parser.add_argument(
    "--seed",
    type=int,
    default=42
)

train_parser.add_argument(
    "--epochs",
    type=int,
    default=30
)

train_parser.add_argument(
    "--batch-size",
    type=int,
    default=32
)

train_parser.add_argument(
    "--normalization",
    action="store_true"
)

train_parser.add_argument(
    "--early-stopping",
    action="store_true"
)

train_parser.add_argument(
    "--patience",
    type=int,
    default=5
)

train_parser.add_argument(
    "--lr",
    type=float,
    default=0.005
)

train_parser.add_argument(
    "--momentum",
    type=float,
    default=0.9
)

# =====================
# TEST
# =====================

test_parser = subparsers.add_parser(
    "test",
    help="Tester un réseau"
)

test_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Nom du réseau"
)

test_parser.add_argument(
    "--params_path",
    type=str,
    required=True
)

test_parser.add_argument(
    "--seed",
    type=int,
    default=42
)


args = parser.parse_args()



# configuration du Train/Test


device = torch.device('cpu')

conf = {
    # device configuration (CPU)
    "device": device,

    "normalization": True,
    
    # train configuration
    "train_data_transforms": [],
    "loss_function": {"type": "cross_entropy"},
    "scheduler": {"type": "steplr", "step_size": 5, "gamma": 1},

    # test configuration
    "test_data_transforms": []
}


if args.command == "train":

    conf.update({
        "seed": args.seed,
        "params_path": args.params_path,
        "config_path": args.config_path,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping": args.early_stopping,
        "patience": args.patience,
        "optimizer": {
            "type": "sgd",
            "lr": args.lr,
            "momentum": args.momentum,
        }
    })


elif args.command == "test":

    conf.update({
        "seed": args.seed,
        "params_path": args.params_path,
        "config_path": args.config_path,
    })

elif args.command == "info":
    
    conf.update({
        "config_path": args.config_path
    })


# exécution du Train/Test

if args.command == "train" :

    train = Train(conf)
    train.run()

elif args.command == "test" :

    test = Test(conf)
    test.run()

elif args.command == "info":

    network = Network(conf)
    network.display()






"""
possibilités d'amélioration :

optimizer :
{"type": "sgd", "lr": 0.005, "momentum": 0.9}
{"type": "adam", "lr": 0.001}

scheduler :
{"type": "steplr", "step_size": 5, "gamma": 1}

train_data_transforms :
{"type": "elastic"}
{"type": "rotation", "degree": 35}
{"type": "affine", "degree": 35, "translate" : (0.21, 0.21)}
{"type": "pad", "padding": 2}
{"type": "resize", "size": (28, 28)}

test_data_transforms :
{"type": "elastic"}
{"type": "rotation", "degree": 35}
{"type": "affine", "degree": 35, "translate" : (0.21, 0.21)}
{"type": "pad", "padding": 2}
{"type": "resize", "size": (28, 28)}

"""