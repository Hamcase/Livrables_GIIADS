# 🎓 Livrables GIIADS - Portfolio Machine Learning & Deep Learning

Bienvenue dans mon portfolio de projets de Data Science et d'Intelligence Artificielle développés dans le cadre de la formation GIIADS (Génie Informatique - Intelligence Artificielle et Data Science).

---

## 📚 Structure du projet

Ce repository est organisé en plusieurs dossiers thématiques contenant différents projets et exercices :

```
Livrables_GIIADS/
│
├── 📂 Machine_Learning/           # Apprentissage supervisé classique
│   ├── Livrable_1.ipynb          # Régression linéaire (gradient descent)
│   ├── Livrable_2.ipynb          # Classification & frontières de décision
│   ├── Livrable_3.ipynb          # CNNs et Deep Learning
│   └── README.md
│
├── 📂 NLP/                        # Traitement du Langage Naturel
│   ├── sarcasm.ipynb             # Détection de sarcasme avec RNN/LSTM
│   ├── RAG-Chatbot-UNO--main/    # Chatbot RAG sur les règles du jeu UNO
│   └── README.md
│
├── 📂 Time_Series_Projects/       # Séries Temporelles
│   ├── Livrable_TimeSeries_1.ipynb    # Modèles statistiques (ARIMA)
│   ├── Livrable_TimeSeries_2.ipynb    # Deep Learning (RNN/LSTM/GRU)
│   ├── M5_project/                    # Projet M5 Forecasting Kaggle
│   │   ├── M5_Projet.ipynb
│   │   ├── m5-forecasting-data/
│   │   ├── models/
│   │   └── reduced_data/
│   └── README.md
│
├── 📂 Reinforcement_Learning/     # Apprentissage par Renforcement
│   ├── prog1_random.py           # Agent aléatoire
│   ├── prog2_value_iteration.py  # Value Iteration
│   ├── prog3_goal_between_episodes.py  # Goal mobile (épisodes)
│   ├── prog4_goal_during_episode.py    # Goal mobile (temps réel)
│   ├── gym_environment.py        # Environnement Grid World
│   ├── config.py                 # Configurations
│   └── README.md
│
└── 📂 ML_Platform/                # Application AutoML Streamlit
    ├── main_app.py
    ├── model_utils.py
    ├── data_utils.py
    ├── evaluation_utils.py
    ├── ui_components.py
    ├── requirements.txt
    └── ReadMe.md
```

---

## 🎯 Projets principaux

### 1️⃣ Machine Learning Classique
**Objectif** : Maîtriser les fondamentaux du ML supervisé

- **Régression linéaire** : implémentation from scratch avec descente de gradient
- **Classification** : régression logistique, frontières de décision, transformations polynomiales
- **Deep Learning** : introduction aux CNNs avec TensorFlow/Keras

📖 [Voir détails →](./Machine_Learning/README.md)

---

### 2️⃣ Natural Language Processing (NLP)
**Objectif** : Traiter et analyser du texte avec Deep Learning

- **Détection de sarcasme** : classification de texte avec RNN/LSTM/GRU
- **RAG Chatbot** : système de questions-réponses sur les règles du jeu UNO
- **Techniques** : embeddings, tokenization, padding, RAG (Retrieval-Augmented Generation)

📖 [Voir détails →](./NLP/README.md)

---

### 3️⃣ Time Series Analysis & Forecasting
**Objectif** : Prédire l'évolution de séries temporelles

- **Modèles statistiques** : ARIMA, SARIMA
- **Deep Learning** : RNN, LSTM, GRU pour prévision
- **Projet M5** : prévision de ventes Walmart (30k+ produits)

📖 [Voir détails →](./Time_Series_Projects/README.md)

---

### 4️⃣ Reinforcement Learning (RL)
**Objectif** : Apprentissage par interaction avec l'environnement

- **Grid World** : environnement de navigation 2D
- **Value Iteration** : calcul de politique optimale
- **Adaptation dynamique** : goal mobile en temps réel
- **Visualisation** : trajectoires et value states

📖 [Voir détails →](./Reinforcement_Learning/README.md)

---

### 5️⃣ AutoML Platform (Streamlit App)
**Objectif** : Créer une application complète de Machine Learning

Application web interactive permettant de :
- Charger et explorer des datasets (CSV, Excel, datasets intégrés)
- Entraîner des modèles ML classiques (scikit-learn)
- Créer des modèles Deep Learning (MLP, CNN)
- Évaluer et comparer les performances
- Faire des prédictions (batch ou individuel)
- Visualiser les résultats

📖 [Voir détails →](./ML_Platform/ReadMe.md)

---

## 🛠️ Technologies & Frameworks

### Core
- **Python 3.x**
- **Jupyter Notebooks** / **VS Code**

### Data Science
- **NumPy** : calculs numériques
- **Pandas** : manipulation de données
- **Matplotlib / Seaborn** : visualisations

### Machine Learning
- **Scikit-learn** : modèles classiques, preprocessing, métriques
- **Statsmodels** : séries temporelles (ARIMA)

### Deep Learning
- **TensorFlow / Keras** : réseaux de neurones
- **RNN, LSTM, GRU** : séquences et séries temporelles
- **CNN** : vision et convolutions

### Reinforcement Learning
- **Gymnasium** : environnements RL (successeur d'OpenAI Gym)
- **Value Iteration** : programmation dynamique

### Applications
- **Streamlit** : applications web interactives

---

## 📊 Compétences développées

### Machine Learning
✅ Régression linéaire, logistique  
✅ Classification multi-classes  
✅ Feature engineering  
✅ Hyperparameter tuning  
✅ Cross-validation  
✅ Pipelines scikit-learn  

### Deep Learning
✅ Réseaux de neurones profonds (MLP)  
✅ CNNs pour images  
✅ RNNs pour séquences  
✅ LSTM/GRU pour mémoire à long terme  
✅ Architectures bidirectionnelles  
✅ Régularisation (Dropout, L1/L2)  

### Séries Temporelles
✅ Analyse exploratoire temporelle  
✅ Stationnarité et décomposition  
✅ Modèles ARIMA/SARIMA  
✅ Deep Learning pour forecast  
✅ Multi-step prediction  
✅ Feature engineering temporel  

### NLP
✅ Preprocessing de texte  
✅ Tokenization et embeddings  
✅ Classification de texte  
✅ Architectures récurrentes  
✅ RAG (Retrieval-Augmented Generation)  

### Reinforcement Learning
✅ Processus de Décision Markovien (MDP)  
✅ Value Iteration et programmation dynamique  
✅ Politique optimale  
✅ Environnements Gymnasium  
✅ Adaptation dynamique  

### Data Engineering
✅ Gestion de grands volumes de données  
✅ Preprocessing et cleaning  
✅ Pipelines de données  
✅ Sauvegarde de modèles  

### Déploiement
✅ Applications Streamlit  
✅ Interfaces utilisateur interactives  
✅ Visualisations dynamiques  

---

## 🎓 Contexte académique

**Formation** : GIIADS (Génie Informatique - Intelligence Artificielle et Data Science)  
**Étudiant** : Hanane AMCASSOU  
**Année** : 2024-2025

Ces projets représentent les livrables et travaux pratiques réalisés tout au long de la formation, couvrant les aspects théoriques et pratiques du Machine Learning, du Deep Learning et de l'Intelligence Artificielle.

---

## 🚀 Comment utiliser ce repository

### Prérequis
```bash
# Python 3.8+
python --version

# Installation des dépendances principales
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow statsmodels gymnasium streamlit
```

### Utiliser les notebooks
```bash
# Ouvrir avec Jupyter
jupyter notebook

# Ou avec VS Code (recommandé)
code .
```

### Lancer l'application AutoML
```bash
cd ML_Platform
pip install -r requirements.txt
streamlit run main_app.py
```

### Tester le Reinforcement Learning
```bash
cd Reinforcement_Learning
pip install -r requirements.txt
python prog2_value_iteration.py
```

---

## 📈 Progression et évolution

Ce repository est en constante évolution avec l'ajout de nouveaux projets et l'amélioration des notebooks existants au fur et à mesure de la formation.

### Projets récents
- ✅ Reinforcement Learning - Grid World avec Value Iteration
- ✅ RAG Chatbot sur les règles du jeu UNO
- ✅ Projet M5 Forecasting (Kaggle)

### Prochaines étapes
- [ ] Deep Reinforcement Learning (DQN, A3C)
- [ ] Projets de Computer Vision avancés
- [ ] Exploration de modèles Transformers
- [ ] Déploiement cloud des modèles

---

## 📧 Contact

Pour toute question ou collaboration :
- **Nom** : Hanane AMCASSOU
- **Formation** : GIIADS
- **GitHub** : [Hamcase/Livrables_GIIADS](https://github.com/Hamcase/Livrables_GIIADS)

---

## 📄 License

Ces projets sont réalisés dans un cadre académique. Veuillez créditer l'auteur en cas de réutilisation.

---

**⭐ Si ce repository vous aide, n'hésitez pas à le star !**
