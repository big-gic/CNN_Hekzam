import torch
import numpy as np

import argparse

from train import Train
from test import Test









parser = argparse.ArgumentParser(
    description="Entraînement et test de réseaux de neurones sur MNIST"
)

subparsers = parser.add_subparsers(
    dest="command",
    required=True
)


# =====================
# TRAIN
# =====================

train_parser = subparsers.add_parser(
    "train",
    help="Entraîner un réseau"
)

train_parser.add_argument(
    "--config-path",
    type=str,
    required=True,
    help="Chemin vers la configuration du réseau"
)

train_parser.add_argument(
    "--params-path",
    type=str,
    required=True,
    help="Chemin de sauvegarde des paramètres du réseau entraîné"
)

train_parser.add_argument(
    "--output-path",
    type=str,
    required=True,
    help="Chemin du fichier CSV de sortie"
)

train_parser.add_argument(
    "--affine-transform",
    nargs=3,
    type=float,
    metavar=("DEGREE", "TX", "TY"),
    help="Appliquer des transformations affines aux données d'entraînement"
)

train_parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Graine aléatoire pour la reproductibilité des expériences"
)

train_parser.add_argument(
    "--epochs",
    type=int,
    default=30,
    help="Nombre d'itérations d'entraînement"
)

train_parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Taille du lot d'entraînement"
)

train_parser.add_argument(
    "--early-stopping",
    action="store_true",
    help="Activer la méthode d'arrêt anticipé"
)

train_parser.add_argument(
    "--patience",
    type=int,
    default=5,
    help="Nombre d'itérations sans amélioration avant l'arrêt anticipé"
)

train_parser.add_argument(
    "--lr",
    type=float,
    default=0.005,
    help="Taux d'apprentissage pour la méthode de descente de gradient stochastique"
)

train_parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="Momentum pour la méthode de descente de gradient stochastique"
)

# =====================
# TEST
# =====================

test_parser = subparsers.add_parser(
    "test",
    help="Tester un réseau"
)

test_parser.add_argument(
    "--config-path",
    type=str,
    required=True,
    help="Chemin vers la configuration du réseau"
)

test_parser.add_argument(
    "--params-path",
    type=str,
    required=True,
    help="Chemin vers les paramètres du réseau à tester"
)

test_parser.add_argument(
    "--output-path",
    type=str,
    required=True,
    help="Chemin du fichier CSV de sortie"
)

test_parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Graine aléatoire pour la reproductibilité des expériences"
)

test_parser.add_argument(
    "--affine-transform",
    nargs=3,
    type=float,
    metavar=("DEGREE", "TX", "TY"),
    help="Appliquer des transformations affines aux données de test"
)


# =====================
# TESTS-TIME
# =====================

tests_time_parser = subparsers.add_parser(
    "tests-time",
    help="Tester les réseaux de neurones dix fois dans un ordre différent"
)

tests_time_parser.add_argument(
    "--output-path",
    type=str,
    required=True,
    help="Chemin du fichier CSV de sortie"
)

tests_time_parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Graine aléatoire pour la reproductibilité des expériences"
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

    degree, tx, ty = args.affine_transform

    conf.update({
        "seed": args.seed,
        "params_path": args.params_path,
        "config_path": args.config_path,
        "output_path": args.output_path,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping": args.early_stopping,
        "patience": args.patience,
        "optimizer": {"type": "sgd", "lr": args.lr, "momentum": args.momentum},
        "train_data_transforms": [
            {"type": "affine", "degree": degree, "translate": (tx, ty)}
        ]
    })



elif args.command == "test":

    degree, tx, ty = args.affine_transform

    conf.update({
        "seed": args.seed,
        "params_path": args.params_path,
        "config_path": args.config_path,
        "output_path": args.output_path,
        "test_data_transforms": [
            {"type": "affine", "degree": degree, "translate": (tx, ty)}
        ]
    })


elif args.command == "tests-time":
    
    conf.update({
        "output_path": args.output_path,
        "seed": args.seed
    })


# =====================
# EXECUTION DES COMMANDES
# =====================

net_paths = np.array([
    {"config_path": "configs/network3.json", "params_path": "params/network3.pt"},
    {"config_path": "configs/CNN_with_BN.json", "params_path": "params/CNN_with_BN.pt"},
    {"config_path": "configs/dropout_CNN.json", "params_path": "params/dropout_CNN.pt"},
    {"config_path": "configs/dropout_lenet5.json", "params_path": "params/dropout_lenet5.pt"},
    {"config_path": "configs/lenet5.json", "params_path": "params/lenet5.pt"},
    {"config_path": "configs/lenet4.json", "params_path": "params/lenet4.pt"},
    {"config_path": "configs/simplified_lenet5.json", "params_path": "params/simplified_lenet5.pt"},
    {"config_path": "configs/dilated_lenet5.json", "params_path": "params/dilated_lenet5.pt"},
    {"config_path": "configs/lenet1.json", "params_path": "params/lenet1.pt"},
    {"config_path": "configs/400-300-10.json", "params_path": "params/400-300-10.pt"},
    {"config_path": "configs/400-10.json", "params_path": "params/400-10.pt"},
    {"config_path": "configs/128-10.json", "params_path": "params/128-10.pt"}
])

def run_tests_in_random_order(net_paths):

    for i in range(10):

        np.random.shuffle(net_paths)
        print(f"itération n°{i} :\n{[net_path['config_path'] for net_path in net_paths]}\n")

        for net_path in net_paths :
            conf.update(net_path)
            test = Test(conf)
            test.run()


if args.command == "train" :
    train = Train(conf)
    train.run()

elif args.command == "test" :

    test = Test(conf)
    test.run()

elif args.command == "tests-time" :

    run_tests_in_random_order(net_paths)


