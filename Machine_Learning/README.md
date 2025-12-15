# 🤖 Machine Learning - Projets et Exercices

Ce dossier contient les travaux pratiques et projets de Machine Learning classique couvrant différents aspects de l'apprentissage supervisé.

---

## 📂 Contenu

### 📘 Livrable_1.ipynb - Régression Linéaire par Descente de Gradient
**Objectif** : Implémentation from scratch d'un algorithme de régression linéaire utilisant la descente de gradient.

**Concepts couverts** :
- ✅ Génération de données synthétiques (relation linéaire avec bruit)
- ✅ Implémentation manuelle de la descente de gradient
- ✅ Calcul des gradients pour MSE (Mean Squared Error)
- ✅ Optimisation des paramètres (intercept et slope)
- ✅ Visualisation de la convergence
- ✅ Comparaison avec TensorFlow/Keras

**Compétences** :
- Compréhension mathématique de la régression linéaire
- Optimisation itérative
- Hyperparamètres : learning rate, epochs
- Visualisation avec matplotlib

---

### 📗 Livrable_2.ipynb - Classification et Frontières de Décision
**Objectif** : Étude de la régression logistique et des frontières de décision sur des données de classification.

**Concepts couverts** :
- ✅ Génération de clusters avec numpy
- ✅ Classification binaire avec régression logistique
- ✅ Visualisation des frontières de décision linéaires
- ✅ Transformation polynomiale des features (kernel trick)
- ✅ Frontières de décision non-linéaires
- ✅ Pipeline scikit-learn (preprocessing + model)
- ✅ Comparaison sur dataset `make_moons`

**Compétences** :
- Régression logistique
- Feature engineering (transformations polynomiales)
- Pipelines scikit-learn
- Visualisation 2D des décisions de classification

---

### 📙 Livrable_3.ipynb - Deep Learning avec CNNs
**Objectif** : Initiation au Deep Learning avec les réseaux de neurones convolutionnels (CNN).

**Contenu** :

#### Partie 1 : Convolutions manuelles
- Création d'images RGB 5×5 (3 canaux)
- Construction de couches Conv2D avec TensorFlow/Keras
- Analyse des dimensions : (5,5,3) → (3,3,2)
- Visualisation des feature maps
- Compréhension du nombre de paramètres

#### Partie 2 : CNN complet pour classification
- Architecture multi-couches :
  - Couches de convolution (Conv2D)
  - Couches de pooling (MaxPooling2D)
  - Couches denses (Dense)
  - Dropout pour régularisation
- Entraînement sur données synthétiques
- Courbes d'apprentissage (loss, accuracy)
- Prédictions et évaluation

**Compétences** :
- Réseaux de neurones convolutionnels
- Architecture CNN (Conv → Pool → Dense)
- TensorFlow/Keras API
- Feature extraction avec convolutions
- Visualisation des activations

---

## 🛠️ Technologies utilisées

- **Python 3.x**
- **NumPy** : manipulation de tableaux et calculs numériques
- **Pandas** : manipulation de données
- **Matplotlib** : visualisations
- **Scikit-learn** : modèles ML classiques, preprocessing
- **TensorFlow/Keras** : Deep Learning

---

## 📊 Compétences développées

✔️ Régression linéaire et optimisation par gradient  
✔️ Classification binaire et multi-classes  
✔️ Feature engineering et transformations  
✔️ Deep Learning : réseaux de neurones convolutionnels  
✔️ Visualisation des résultats et frontières de décision  
✔️ Pipelines de preprocessing  
✔️ Analyse des hyperparamètres  

---

## 🚀 Comment utiliser

1. Ouvrir les notebooks avec Jupyter ou VS Code
2. Exécuter les cellules séquentiellement
3. Observer les visualisations et les résultats
4. Expérimenter avec les hyperparamètres

---

## 📝 Notes

Ces travaux couvrent les fondamentaux du Machine Learning et du Deep Learning, de l'implémentation manuelle aux frameworks modernes.
