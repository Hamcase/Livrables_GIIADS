# 🎮 Reinforcement Learning - Grid World avec Value Iteration

Ce projet présente une implémentation progressive de l'apprentissage par renforcement (Reinforcement Learning) dans un environnement Grid World, avec 4 programmes de difficulté croissante démontrant différentes approches d'apprentissage et d'adaptation.

## 📋 Programmes

### Programme 1: Agent Random (`prog1_random.py`)
Agent qui explore aléatoirement jusqu'à trouver le goal.
- Exploration pure sans apprentissage
- Visualisation en temps réel

**Exécution:**
```bash
python prog1_random.py
```

### Programme 2: Value Iteration (`prog2_value_iteration.py`)
Agent intelligent utilisant Value Iteration pour apprendre la politique optimale.
- Algorithme de Value Iteration
- Affichage des Value States (couleurs)
- Affichage de la Politique Optimale (flèches)
- Chemin optimal garanti

**Exécution:**
```bash
python prog2_value_iteration.py
```

### Programme 3: Goal Mobile entre Épisodes (`prog3_goal_between_episodes.py`)
Value Iteration avec goal qui change de position entre chaque épisode.
- Ré-entraînement à chaque épisode
- Adaptation à différentes positions de goal
- Visualisation de l'apprentissage continu

**Exécution:**
```bash
python prog3_goal_between_episodes.py
```

### Programme 4: Goal Mobile en Temps Réel (`prog4_goal_during_episode.py`)
Value Iteration avec goal qui se déplace PENDANT l'épisode.
- Re-planning dynamique
- Goal mobile pendant que l'agent se déplace
- Défi d'apprentissage le plus complexe

**Exécution:**
```bash
python prog4_goal_during_episode.py
```

## 🚀 Installation

```bash
pip install numpy matplotlib gymnasium
```

## 📊 Configurations Disponibles

- **SMALL**: 5x5, 1 goal, 1 obstacle (recommandé pour démo)
- **DEFAULT**: 10x10, 1 goal, 2 obstacles
- **LARGE**: 15x15, 2 goals, 6 obstacles
- **COMPLEX**: 12x12, 3 goals, 10 obstacles

## 🎯 Concepts Clés

### Value Iteration
Algorithme de programmation dynamique qui calcule la valeur optimale de chaque état:
```
V(s) = max_a [R(s,a) + γ × V(s')]
```

### Politique Optimale
Meilleure action à prendre dans chaque état pour maximiser la récompense cumulative.

### Visualisation
- 🎨 **Couleurs**: Value States (rouge→vert = faible→élevé)
- ➡️ **Flèches**: Direction optimale
- 🔵 **Agent**: Position actuelle
- 🟡 **Goal**: Objectif
- ⬛ **Obstacles**: Cases bloquées

## 📈 Résultats Typiques

### Programme 1 (Random)
- Taux de succès: 20-40%
- Steps: 40-100 (aléatoire)

### Programme 2 (Value Iteration)
- Taux de succès: 100%
- Steps: Optimal (chemin le plus court)

### Programme 3 (Goal mobile - épisodes)
- Taux de succès: 100% par épisode
- Ré-apprentissage rapide

### Programme 4 (Goal mobile - temps réel)
- Comportement adaptatif
- Re-planning continu

## 🎓 À propos

Projet développé dans le cadre de la formation GIIADS (Génie Informatique - Intelligence Artificielle et Data Science) pour explorer les concepts fondamentaux du Reinforcement Learning.

**Auteur** : Hanane AMCASSOU  
**Formation** : GIIADS 2024-2025

---

## 🛠️ Technologies utilisées

- **Python 3.x**
- **NumPy** : calculs numériques et matrices
- **Matplotlib** : visualisation de l'environnement et des trajectoires
- **Gymnasium** : framework pour environnements de RL (successeur de OpenAI Gym)

---

## 📚 Concepts théoriques

### Apprentissage par Renforcement
L'agent apprend à prendre des décisions optimales en interagissant avec l'environnement et en recevant des récompenses.

### Équation de Bellman
```
V(s) = max_a [R(s,a) + γ × V(s')]
```
- `V(s)` : Valeur de l'état s
- `R(s,a)` : Récompense immédiate
- `γ` : Facteur d'actualisation
- `V(s')` : Valeur de l'état suivant

### Processus de Décision Markovien (MDP)
- **États** : Positions dans la grille
- **Actions** : {Haut, Bas, Gauche, Droite}
- **Récompenses** : Goal (+1), Obstacle (-1), Step (-0.01)
- **Transitions** : Déterministes (100% de succès)

---

## 🔗 Ressources

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
