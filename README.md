# CNN_Hekzam

Un outil complet pour l'expérimentation de réseaux de neurones sur le dataset MNIST. Ce projet permet d'entraîner, tester et analyser différentes architectures de réseaux de neurones.

## 📋 Table des matières

- [Installation](#installation)
- [Utilisation](#utilisation)
  - [Entraîner un réseau](#entraîner-un-réseau)
  - [Tester un réseau](#tester-un-réseau)
  - [Tester plusieurs réseaux](#tester-plusieurs-réseaux)
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
pip install torch torchvision pandas scikit-learn matplotlib
```

## 🚀 Utilisation

Le script `script.py` utilise un système de commandes pour gérer différentes opérations sur les réseaux de neurones.

### Entraîner un réseau

Entraîne un réseau de neurones sur le dataset MNIST.

```bash
python script.py train --config-path <chemin_vers_config> --params-path <chemin_vers_poids> --output-path <chemin_csv> [options]
```

**Arguments requis :**
- `--config-path` : Chemin vers le fichier de configuration du réseau
- `--params-path` : Chemin où sauvegarder les paramètres du réseau entraîné
- `--output-path` : Chemin du fichier CSV où enregistrer les métriques d'entraînement

**Arguments optionnels :**
- `--seed` (par défaut: `42`) : Seed pour la reproductibilité
- `--epochs` (par défaut: `30`) : Nombre d'itérations d'entraînement
- `--batch-size` (par défaut: `32`) : Taille du lot d'entraînement
- `--early-stopping` : Activer l'arrêt précoce de l'entraînement
- `--patience` (par défaut: `5`) : Nombre d'itérations sans amélioration avant arrêt
- `--lr` (par défaut: `0.005`) : Taux d'apprentissage
- `--momentum` (par défaut: `0.9`) : Momentum pour SGD
- `--affine-transform DEGREE TX TY` : Appliquer une transformation affine sur les images d'entraînement

**Exemple basique :**
```bash
python script.py train --config-path configs/mon_reseau.json --params-path params/mon_reseau.pt --output-path results/train_results.csv
```

**Exemple avec options personnalisées :**
```bash
python script.py train \
    --config-path configs/mon_reseau.json \
    --params-path params/mon_reseau.pt \
    --output-path results/train_results.csv \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.01 \
    --momentum 0.95 \
    --early-stopping \
    --patience 10 \
    --affine-transform 15 0.1 0.1
```

---

### Tester un réseau

Évalue les performances d'un réseau entraîné sur l'ensemble de test.

```bash
python script.py test --config-path <chemin_vers_config> --params-path <chemin_vers_poids> --output-path <chemin_csv> [options]
```

**Arguments requis :**
- `--config-path` : Chemin vers le fichier de configuration du réseau
- `--params-path` : Chemin vers les paramètres du réseau entraîné
- `--output-path` : Chemin du fichier CSV où enregistrer les métriques de test

**Arguments optionnels :**
- `--seed` (par défaut: `42`) : Seed pour la reproductibilité
- `--affine-transform DEGREE TX TY` : Appliquer une transformation affine sur les images de test

**Exemple :**
```bash
python script.py test --config-path configs/mon_reseau.json --params-path params/mon_reseau.pt --output-path results/test_results.csv
```

---

### Tester plusieurs réseaux

Teste plusieurs réseaux dans un ordre aléatoire dix fois. C'est un processus utile pour évaluer les différentes métriques de temps ('network_time', 'load_time', 'forward_time') de ces réseaux. Cette commande exécute un ensemble de modèles en utilisant le dossier 'configs', pour récuperer les configuration de chaque modèle, et le dossier 'params', pour récupérer les poids de chaque modèle.



```bash
python script.py tests-time --output-path <chemin_csv> [--seed <seed>]
```

**Arguments requis :**
- `--output-path` : Chemin du fichier CSV où enregistrer les métriques de test

**Arguments optionnels :**
- `--seed` (par défaut: `42`) : Seed pour la reproductibilité

**Exemple :**
```bash
python script.py tests-time --output-path results/tests_time_results.csv --seed 123
```

## Arguments disponibles

Les commandes disponibles dans `script.py` sont :

- `train` : entraînement d'un réseau avec sauvegarde des poids et export des métriques dans un CSV
- `test` : évaluation d'un modèle entraîné et export des métriques dans un CSV
- `tests-time` : test de plusieurs modèles prédéfinis dans un ordre aléatoire sur 10 itérations

Chaque commande possède ses propres arguments requis et optionnels, décrits ci-dessus.

## 💡 Exemples

### Workflow complet : entraîner et tester un réseau

```bash
# 1. Entraîner le réseau
python script.py train \
    --config-path configs/lenet5.json \
    --params-path params/lenet5.pt \
    --output-path results/lenet5_train.csv \
    --epochs 20 \
    --batch-size 32 \
    --early-stopping \
    --patience 3 \
    --lr 0.001

# 2. Tester le réseau entraîné
python script.py test \
    --config-path configs/lenet5.json \
    --params-path params/lenet5.pt \
    --output-path results/lenet5_test.csv
```

---

## ⚙️ Configuration avancée

### Configuration du device

Par défaut, le script utilise le CPU. Pour utiliser un GPU (si disponible), modifiez la ligne dans `script.py` :

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Fichier de configuration du réseau

Les fichiers JSON de configuration contiennent une liste de couches et des transformations de données. Le script lit les clés `layer_list` et `data_transformations` pour construire le réseau et prétraiter les images.

---

**Sortie CSV — Colonnes et métriques**

Le script écrit des lignes dans un fichier CSV (`--output-path`) contenant les métriques et métadonnées calculées pour chaque exécution. Les colonnes actuellement produites sont :

- **model**: chemin vers le fichier de paramètres utilisé ou sauvegardé (`params/*.pt`).
- **train_data_time**: (train only) temps total (en secondes) passé à préparer les données d'entraînement avant l'entraînement.
- **test_data_time**: (test only) temps total (en secondes) passé à préparer les données de test avant l'évaluation.
- **network_time**: temps (en secondes) nécessaire à la construction/initialisation du réseau.
- **train_time**: (train only) temps total (en secondes) passé pendant la phase d'entraînement.
- **forward_time**: (test only) temps total (en secondes) passé dans les passes avant (inférence) lors de l'évaluation.
- **load_time**: (test only) temps total (en secondes) passé à charger/préparer les batches pendant l'évaluation.
- **accuracy**: (test only) précision en pourcentage calculée sur l'ensemble de test.
- **confusion_matrix**: (test only) matrice de confusion sous forme de liste imbriquée (10x10) sérialisée dans la cellule CSV.
- **epoch**: (train only) valeur retournée par la boucle d'entraînement (indice d'epoch retourné par `Trainer.train`).
- **lr**: (train only) chaîne de la forme `"<lr_initial>-<lr_final>"` indiquant le learning rate initial et le learning rate après entraînement.

Remarques :

- Les fichiers CSV peuvent contenir un mélange de lignes provenant de `train` et de `test` (ou `tests-time`). Certaines colonnes seront vides selon la commande exécutée.
- La colonne `confusion_matrix` contient une représentation JSON-like (liste de listes) : la lecture/visualisation exige de parser cette cellule avant utilisation.
- Les durées sont mesurées avec `time.perf_counter()` et sont exprimées en secondes.

---

## 📊 À propos du projet

CNN_Hekzam est un projet L3 sur l'intelligence artificielle, permettant d'expérimenter avec des architectures de réseaux de neurones modernes sur le dataset MNIST classique.
