# CNN_Hekzam

Un outil complet pour l'expérimentation de réseaux de neurones sur le dataset MNIST. Ce projet permet d'entraîner, tester et analyser différentes architectures de réseaux de neurones.

## 📋 Table des matières

- [Installation](#installation)
- [Utilisation](#utilisation)
  - [Afficher les informations d'un réseau](#afficher-les-informations-dun-réseau)
  - [Entraîner un réseau](#entraîner-un-réseau)
  - [Tester un réseau](#tester-un-réseau)
- [Arguments disponibles](#arguments-disponibles)
- [Exemples](#exemples)
- [Configuration avancée](#configuration-avancée)

## 🔧 Installation

### Prérequis
- Python 3.7+
- PyTorch

### Setup
```bash
# Cloner le repository
git clone https://github.com/big-gic/CNN_Hekzam.git
cd CNN_Hekzam

# Installer les dépendances
pip install torch torchvision
```

## 🚀 Utilisation

Le script `script.py` utilise un système de commandes pour gérer différentes opérations sur les réseaux de neurones.

### Afficher les informations d'un réseau

Affiche les détails et l'architecture d'un réseau de neurones.

```bash
python script.py info --config_path <chemin_vers_config>
```

**Arguments :**
- `--config_path` (requis) : Chemin vers le fichier de configuration du réseau

**Exemple :**
```bash
python script.py info --config_path configs/mon_reseau.json
```

---

### Entraîner un réseau

Entraîne un réseau de neurones sur le dataset MNIST.

```bash
python script.py train --config_path <chemin_vers_config> [options]
```

**Arguments requis :**
- `--config_path` : Chemin vers le fichier de configuration du réseau

**Arguments optionnels :**
- `--params_path` : Chemin où sauvegarder les paramètres du réseau entraîné
- `--seed` (par défaut: `42`) : Seed pour la reproductibilité
- `--epochs` (par défaut: `30`) : Nombre d'epochs d'entraînement
- `--batch-size` (par défaut: `32`) : Taille des batches
- `--normalization` : Activer la normalisation des données
- `--early-stopping` : Activer l'arrêt précoce de l'entraînement
- `--patience` (par défaut: `5`) : Nombre d'epochs sans amélioration avant arrêt précoce
- `--lr` (par défaut: `0.005`) : Learning rate (taux d'apprentissage)
- `--momentum` (par défaut: `0.9`) : Momentum pour SGD

**Exemple basique :**
```bash
python script.py train --config_path configs/mon_reseau.json
```

**Exemple avec options personnalisées :**
```bash
python script.py train \
    --config_path configs/mon_reseau.json \
    --params_path model_params/mon_reseau.pt \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.01 \
    --momentum 0.95 \
    --normalization \
    --early-stopping \
    --patience 10
```

---

### Tester un réseau

Évalue les performances d'un réseau entraîné sur l'ensemble de test.

```bash
python script.py test --config_path <chemin_vers_config> --params_path <chemin_vers_poids> [options]
```

**Arguments requis :**
- `--config_path` : Chemin vers le fichier de configuration du réseau
- `--params_path` : Chemin vers les paramètres du réseau entraîné

**Arguments optionnels :**
- `--seed` (par défaut: `42`) : Seed pour la reproductibilité

**Exemple :**
```bash
python script.py test --config_path configs/mon_reseau.json --params_path model_params/mon_reseau.pt
```


## 💡 Exemples

### Workflow complet : Créer, entraîner et tester un réseau

```bash
# 1. Afficher les informations du réseau
python script.py info --config_path configs/cnn_simple.json

# 2. Entraîner le réseau avec normalisation et early stopping
python script.py train \
    --config_path configs/cnn_simple.json \
    --params_path models/cnn_simple.pt \
    --epochs 100 \
    --batch-size 128 \
    --normalization \
    --early-stopping \
    --patience 15 \
    --lr 0.001

# 3. Tester le réseau entraîné
python script.py test \
    --config_path configs/cnn_simple.json \
    --params_path models/cnn_simple.pt
```

### Expérimentation avec différents hyperparamètres

```bash
# Test avec SGD
python script.py train --config_path configs/mon_reseau.json --lr 0.01 --momentum 0.95

# Test avec un batch size plus grand
python script.py train --config_path configs/mon_reseau.json --batch-size 64

# Test avec plus d'epochs
python script.py train --config_path configs/mon_reseau.json --epochs 50
```

---

## ⚙️ Configuration avancée

### Configuration du device

Par défaut, le script utilise le CPU. Pour utiliser un GPU (si disponible), modifiez la ligne dans `script.py` :

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Fichier de configuration du réseau

Créez un fichier JSON pour définir l'architecture de votre réseau (ex: `configs/mon_reseau.json`) avec les paramètres de votre réseau.

---

## 📊 À propos du projet

CNN_Hekzam est un projet L3 sur l'intelligence artificielle, permettant d'expérimenter avec des architectures de réseaux de neurones modernes sur le dataset MNIST classique.
