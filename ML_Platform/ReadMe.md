# 🧠 AutoML App – Application de Machine Learning & Deep Learning avec Streamlit

Une application **Streamlit** complète, modulaire et intuitive permettant de **créer, entraîner, évaluer et déployer des modèles de Machine Learning et Deep Learning** en quelques clics.

---

## 🚀 Fonctionnalités principales

### 🔹 Chargement & préparation des données
- Upload de fichiers **CSV / Excel**
- Chargement de **datasets intégrés** (`iris`, `wine`, `breast_cancer`, `diabetes`, `california_housing`, `cifar10`)
- Affichage des infos principales : shape, types de colonnes, valeurs manquantes
- Sélection manuelle ou automatique des features et de la cible
- Split **train/test** avec `test_size` et `stratify`
- Nettoyage et **prétraitement automatique** : encodage, standardisation, imputation

---

### 🔹 Choix & configuration du modèle

#### Méthodes classiques (scikit-learn)
- Régression : `LinearRegression`, `RandomForestRegressor`, `SVR`, etc.  
- Classification : `LogisticRegression`, `RandomForestClassifier`, `SVM`, `KNN`, `NaiveBayes`, etc.
- Interface de réglage des **hyperparamètres avec explication de chaque paramètre**
- Visualisation automatique :
  - Matrice de confusion
  - Courbe ROC
  - Graphiques d’erreur pour la régression

#### Deep Learning (TensorFlow / Keras)
- Choix entre :
  - **MLP (Fully Connected)** pour données tabulaires
  - **CNN (Convolutional Neural Network)** pour images (ex. CIFAR-10)
- Deux modes :
  - **Auto-build** : architecture générée automatiquement
  - **Custom-build** : couches, activation, dropout, etc.
- Visualisation des **courbes de perte et d’exactitude**

---

### 🔹 Évaluation & Prédiction
- Métriques de performance :
  - Classification : `Accuracy`, `Precision`, `Recall`, `F1-score`
  - Régression : `MAE`, `MSE`, `R²`
- Courbes associées et rapport JSON téléchargeable
- Formulaire pour prédire sur un **exemple manuel**
- Upload d’un fichier CSV pour prédire en **batch**
- Prédiction sur **images (PNG/JPG)** pour les CNN

---

### 🔹 Gestion des modèles
- **Sauvegarde automatique** des modèles entraînés dans le dossier `/models`
  - `.pkl` → modèles scikit-learn
  - `.h5` → modèles Keras
- **Chargement ou upload d’un modèle existant** directement depuis l’interface
- Recompilation automatique des modèles Keras (`compile=False` → recompilés avant réentraînement)

---

## 📁 Structure du projet

```
project/
│
├── main_app.py                # Point d’entrée Streamlit
│
├── data_utils.py              # Chargement et prétraitement des données
├── model_utils.py             # Création, entraînement, sauvegarde, chargement de modèles
├── evaluation_utils.py        # Évaluation et visualisation des métriques
├── ui_components.py           # Interface Streamlit (blocs modulaires)
│
├── models/                    # Modèles sauvegardés (.pkl, .h5)
├── assets/                    # Ressources statiques (datasets, images, etc.)
│
└── requirements.txt           # Dépendances Python
```

---

## ⚙️ Installation & Lancement

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/Hamcase/ML_Platform.git
cd ML_Platform
```

### 2️⃣ Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4️⃣ Lancer l’application
```bash
streamlit run main_app.py
```

> L’application s’ouvrira automatiquement dans ton navigateur :
> [http://localhost:8501](http://localhost:8501)

---

## 🧩 Jeux de données disponibles

Tu peux utiliser :
- des **fichiers personnels** (upload CSV ou Excel),
- ou des **datasets intégrés** (`iris`, `wine`, `breast_cancer`, `diabetes`, `california_housing`, `cifar10`).

> Le dataset `cifar10` est le seul dataset image intégré (supporté par CNN).

---

## 💾 Sauvegarde et rechargement de modèles

Tous les modèles entraînés sont sauvegardés dans le dossier `models/` :
- `.pkl` → scikit-learn
- `.h5` → Keras

Depuis l’application :
- Ouvre **“📁 Charger un modèle existant”**
- Sélectionne un modèle déjà sauvegardé
- Ou **uploade** ton propre fichier `.pkl` / `.h5`  
  → Le modèle sera automatiquement importé et prêt à être utilisé.

---

## 🧠 Exemples de tests

| Cas d’usage | Dataset | Type de modèle |
|--------------|----------|----------------|
| Classification simple | `iris` | `RandomForestClassifier` |
| Régression | `california_housing` | `LinearRegression` |
| Image (Deep Learning) | `cifar10` | `CNN` |

---

## 📘 À propos du dossier `assets/`

Le dossier `assets/` contient toutes les **ressources statiques** :
- datasets de démonstration (ex : `sample_titanic.csv`)
- images ou logos pour l’interface
- templates de rapports ou de configuration  

Ce dossier est **optionnel** mais utile pour les démos locales sans upload.

---

## 🧰 Technologies utilisées

| Outil / Librairie | Rôle |
|--------------------|------|
| **Streamlit** | Interface web interactive |
| **scikit-learn** | Machine Learning classique |
| **TensorFlow / Keras** | Deep Learning |
| **Pandas / NumPy** | Traitement de données |
| **Matplotlib / Seaborn** | Visualisation |
| **Joblib** | Sauvegarde des modèles |
| **Pathlib / OS** | Gestion des fichiers |

---

## 💡 Conseils & bonnes pratiques

- Toujours vérifier les types de colonnes avant entraînement  
- Sauvegarder tes modèles entraînés dans `models/`  
- Pour les CNN, vérifier la taille d’entrée (`input_shape`)  
- Si tu charges un modèle `.h5` d’une autre version de TensorFlow, il sera automatiquement recompilé avant réentraînement  

---

## 🧾 Licence

Ce projet est open-source, sous licence MIT.  
Tu peux le modifier et le redistribuer librement en citant l’auteur original.

---

## ✨ Auteur

**Nom :**  Amcassou Hanane

**Email :**  amcassouhanane03@gmail.com

**Organisation / Études :**  Ecole nationale supérieure des arts et metiers - Meknès
