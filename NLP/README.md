# 🗣️ Natural Language Processing (NLP)

Ce dossier contient les projets liés au traitement du langage naturel (NLP) avec TensorFlow et Keras.

---

## 📂 Contenu

### 📘 sarcasm.ipynb - Détection de Sarcasme avec RNNs

**Objectif** : Développer un modèle de Deep Learning capable de détecter le sarcasme dans des textes en utilisant des réseaux de neurones récurrents.

**Concepts couverts** :

#### 1️⃣ Prétraitement du texte
- ✅ **Tokenization** : conversion de phrases en séquences numériques
  - Utilisation de `Tokenizer` de Keras
  - Création d'un vocabulaire (word_index)
- ✅ **Padding** : uniformisation de la longueur des séquences
  - `pad_sequences` avec différentes stratégies (pre/post)
- ✅ **TextVectorization** : layer Keras pour vectorisation
- ✅ Gestion du vocabulaire et des mots hors vocabulaire (OOV)

#### 2️⃣ Embeddings
- ✅ **Word Embeddings** : représentation dense des mots
  - Embedding layer dans Keras
  - Dimensions d'embedding
  - Embeddings appris vs pré-entraînés
- ✅ Visualisation des embeddings

#### 3️⃣ Architectures de Deep Learning pour NLP

**Modèles implémentés** :
- 🔹 **RNN simple** (Recurrent Neural Network)
  - Architecture basique pour séquences
  - Problèmes de gradient vanishing
  
- 🔹 **LSTM** (Long Short-Term Memory)
  - Meilleure capture des dépendances à long terme
  - Gates : forget, input, output
  
- 🔹 **GRU** (Gated Recurrent Unit)
  - Version simplifiée du LSTM
  - Moins de paramètres
  
- 🔹 **Bidirectional RNN/LSTM**
  - Traitement dans les deux sens (forward + backward)
  - Meilleure compréhension du contexte

#### 4️⃣ Entraînement et évaluation
- ✅ Compilation des modèles (optimizer, loss, metrics)
- ✅ Entraînement avec callbacks
- ✅ Visualisation des courbes d'apprentissage
- ✅ Métriques de classification :
  - Accuracy
  - Precision, Recall, F1-score
  - Matrice de confusion
- ✅ Prédictions sur textes individuels
- ✅ Batch predictions

#### 5️⃣ Techniques avancées
- ✅ **Dropout** pour régularisation
- ✅ **Early Stopping** pour éviter le surapprentissage
- ✅ **Learning Rate Scheduling**
- ✅ Comparaison de différentes architectures

---

## 🎯 Dataset

**Sarcasm Detection Dataset** : phrases étiquetées comme sarcastiques ou non-sarcastiques

Structure typique :
- **Texte** : phrases/commentaires
- **Label** : 0 (non-sarcastique) ou 1 (sarcastique)
- **Contexte** : éventuellement des métadonnées

---

## 🛠️ Technologies utilisées

- **Python 3.x**
- **TensorFlow / Keras** : construction et entraînement des modèles
- **NumPy** : manipulation de données
- **Pandas** : chargement et exploration du dataset
- **Matplotlib / Seaborn** : visualisations
- **Scikit-learn** : métriques et preprocessing

---

## 📊 Compétences développées

✔️ Prétraitement de texte : tokenization, padding, embeddings  
✔️ Word embeddings et représentations vectorielles  
✔️ Réseaux de neurones récurrents (RNN, LSTM, GRU)  
✔️ Architecture bidirectionnelle  
✔️ Classification de texte  
✔️ Gestion de séquences de longueur variable  
✔️ Régularisation et prévention du surapprentissage  
✔️ Évaluation de modèles NLP  

---

## 🚀 Comment utiliser

1. Charger le dataset de détection de sarcasme
2. Exécuter le preprocessing du texte
3. Entraîner différents modèles (RNN, LSTM, GRU)
4. Comparer les performances
5. Faire des prédictions sur de nouveaux textes

---

## 💡 Cas d'usage

- Analyse de sentiment
- Détection d'ironie/sarcasme sur les réseaux sociaux
- Modération de contenu
- Chatbots et assistants conversationnels
- Analyse d'avis clients

---

## 📝 Notes

Ce projet illustre l'application du Deep Learning au traitement du langage naturel, en particulier pour des tâches de classification de texte nécessitant la compréhension du contexte et des subtilités linguistiques comme le sarcasme.
