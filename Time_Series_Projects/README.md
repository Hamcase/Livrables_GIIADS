# ⏰ Time Series - Projets de Prévision et Analyse Temporelle

Ce dossier contient les projets d'analyse et de prévision de séries temporelles utilisant différentes techniques statistiques et de Deep Learning.

---

## 📂 Contenu

### 📘 Livrable_TimeSeries_1.ipynb - Fondamentaux des Séries Temporelles

**Objectif** : Maîtriser les concepts de base et les modèles classiques d'analyse de séries temporelles.

**Concepts couverts** :

#### 1️⃣ Analyse exploratoire
- ✅ Visualisation de séries temporelles
- ✅ Détection de tendances (trend)
- ✅ Identification de saisonnalité (seasonality)
- ✅ Analyse de la stationnarité
- ✅ Tests statistiques (ADF test, KPSS test)

#### 2️⃣ Décomposition de séries temporelles
- ✅ Décomposition additive vs multiplicative
- ✅ Extraction de tendance, saisonnalité et résidus
- ✅ Moving averages (moyennes mobiles)

#### 3️⃣ Modèles statistiques classiques
- 🔹 **AR (AutoRegressive)** : régression sur valeurs passées
- 🔹 **MA (Moving Average)** : moyenne des erreurs passées
- 🔹 **ARMA** : combinaison AR + MA
- 🔹 **ARIMA** : ARMA avec différenciation pour stationnarité
- 🔹 **SARIMA** : ARIMA avec composante saisonnière

#### 4️⃣ Préparation des données
- ✅ Windowing (fenêtres glissantes)
- ✅ Train/Test split temporel
- ✅ Normalisation des données
- ✅ Création de features lag

#### 5️⃣ Évaluation
- ✅ Métriques : MAE, RMSE, MAPE
- ✅ Visualisation des prédictions vs valeurs réelles
- ✅ Analyse des résidus

**Compétences** :
- Analyse de séries temporelles
- Stationnarité et transformations
- Modèles ARIMA/SARIMA
- Prévision statistique

---

### 📗 Livrable_TimeSeries_2.ipynb - Deep Learning pour Time Series

**Objectif** : Appliquer des architectures de Deep Learning avancées pour la prévision de séries temporelles.

**Architectures implémentées** :

#### 1️⃣ Réseaux de neurones récurrents
- 🔹 **RNN simple** (Simple Recurrent Neural Network)
  - Architecture de base pour séquences temporelles
  
- 🔹 **LSTM** (Long Short-Term Memory)
  - Meilleure capture des dépendances à long terme
  - Résolution du gradient vanishing
  
- 🔹 **GRU** (Gated Recurrent Unit)
  - Alternative plus légère au LSTM
  - Moins de paramètres, apprentissage plus rapide

- 🔹 **Bidirectional LSTM/GRU**
  - Traitement dans les deux sens temporels
  - Utile pour certains contextes

#### 2️⃣ Réseaux convolutionnels temporels
- 🔹 **1D CNN** : convolutions sur séquences temporelles
- 🔹 **Temporal Convolutional Networks (TCN)**

#### 3️⃣ Architectures hybrides
- ✅ CNN + LSTM : extraction de features + mémoire temporelle
- ✅ Attention mechanisms
- ✅ Encoder-Decoder architectures

#### 4️⃣ Techniques d'entraînement
- ✅ Callbacks : EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- ✅ Hyperparameter tuning
- ✅ Régularisation : Dropout, L1/L2
- ✅ Batch normalization

#### 5️⃣ Multi-step forecasting
- ✅ Prévision un pas dans le futur
- ✅ Prévision multi-horizons
- ✅ Strategies : Direct, Recursive, DirRec

**Compétences** :
- Deep Learning pour séries temporelles
- Architectures récurrentes (LSTM, GRU)
- Windowing et preprocessing temporel
- Multi-step forecasting
- Comparaison de modèles

---

### 📙 M5_project/ - Projet M5 Forecasting (Kaggle)

**Objectif** : Prévision de ventes à grande échelle sur le dataset M5 de Walmart.

#### Dataset M5
- **30,490 produits** vendus dans 10 magasins
- **3 états** : California, Texas, Wisconsin
- **1,913 jours** de données historiques
- **Features** :
  - Ventes quotidiennes par produit
  - Prix des produits
  - Événements spéciaux
  - Jours fériés

#### Structure du projet
```
M5_project/
├── M5_Projet.ipynb          # Notebook principal
├── m5-forecasting-data/     # Données Kaggle
│   ├── calendar.csv         # Calendrier et événements
│   ├── sales_train_validation.csv  # Historique des ventes
│   ├── sell_prices.csv      # Historique des prix
│   └── sample_submission.csv
├── models/                  # Modèles entraînés
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   ├── rnn_model.keras
│   └── mlp_model.keras
└── reduced_data/            # Données réduites pour prototypage
    ├── calendar_df_reduced.csv
    ├── prices_df_reduced.csv
    └── sales_df_reduced.csv
```

#### Approches utilisées
- ✅ Feature engineering : lags, rolling statistics, encodings
- ✅ Réduction de dimensionnalité (échantillonnage stratégié)
- ✅ Modèles Deep Learning :
  - MLP (Multi-Layer Perceptron)
  - RNN simple
  - LSTM
  - GRU
- ✅ Comparaison des performances
- ✅ Sauvegarde des modèles

**Challenges** :
- Volume de données massif
- Hiérarchie des séries (produit → catégorie → magasin → état)
- Saisonnalité multiple
- Événements spéciaux et promotions

---

## 🛠️ Technologies utilisées

- **Python 3.x**
- **TensorFlow / Keras** : modèles Deep Learning
- **NumPy / Pandas** : manipulation de données
- **Matplotlib / Seaborn** : visualisations
- **Statsmodels** : modèles ARIMA/SARIMA
- **Scikit-learn** : preprocessing, métriques

---

## 📊 Compétences développées

✔️ Analyse et décomposition de séries temporelles  
✔️ Modèles statistiques (ARIMA, SARIMA)  
✔️ Deep Learning : RNN, LSTM, GRU  
✔️ Windowing et feature engineering temporel  
✔️ Multi-step forecasting  
✔️ Gestion de grands volumes de données  
✔️ Feature engineering avancé (lags, rolling stats)  
✔️ Métriques de prévision (MAE, RMSE, MAPE)  
✔️ Sauvegarde et déploiement de modèles  

---

## 🎯 Applications pratiques

- Prévision de ventes (retail)
- Prévision de demande (supply chain)
- Prévision énergétique
- Analyse financière (prix actions, crypto)
- Prévision météorologique
- Analyse de trafic web

---

## 🚀 Comment utiliser

1. **Livrable 1** : Commencer par les fondamentaux et modèles statistiques
2. **Livrable 2** : Explorer les modèles Deep Learning
3. **M5 Project** : Projet complet sur données réelles à grande échelle

---

## 📝 Notes

Ces projets couvrent l'ensemble du pipeline de prévision de séries temporelles, des méthodes statistiques classiques aux architectures de Deep Learning modernes, avec application sur un cas réel d'envergure (M5 Kaggle Competition).
