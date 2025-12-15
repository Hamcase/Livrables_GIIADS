# 🎮 UNO RAG Chatbot

Un chatbot RAG (Retrieval-Augmented Generation) spécialisé dans les règles du jeu de société UNO. Le système utilise des modèles locaux légers pour répondre aux questions sur les règles du jeu en se basant sur des documents officiels.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Fonctionnalités

- 🤖 **LLM Local** : Utilise Mistral 7B via Ollama pour des réponses de qualité
- 📚 **RAG Pipeline** : Récupération de contexte pertinent avec FAISS vectorstore
- 🎯 **Embeddings légers** : Modèle sentence-transformers (all-MiniLM-L6-v2, ~80MB)
- 💻 **Interface Streamlit** : Interface web simple et épurée
- 👤 **Mode User** : Réponses simples et directes
- 👨‍💻 **Mode Developer** : Affiche le contexte RAG et les sources utilisées
- 🔒 **100% Local** : Aucune donnée envoyée à des API externes

## 🏗️ Architecture

```
Pipeline RAG
├── Documents (data/)
│   └── PDF/Markdown des règles UNO
├── Chunking & Embedding
│   └── RecursiveCharacterTextSplitter + all-MiniLM-L6-v2
├── Vectorstore
│   └── FAISS (index local)
├── Retrieval
│   └── Similarity search (top-k)
└── Generation
    └── Mistral 7B (via Ollama)
```

## 🚀 Installation

### Prérequis

- Python 3.8+
- [Ollama](https://ollama.ai/) installé
- ~5GB d'espace disque (modèles + dépendances)

### 1. Cloner le repository

```bash
git clone https://github.com/Hamcase/RAG-Chatbot-UNO-.git
cd uno-rag-chatbot
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Installer et lancer Ollama

#### Windows
```powershell
winget install Ollama.Ollama
```

#### Linux/Mac
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 5. Télécharger le modèle Mistral

```bash
ollama pull mistral
```

Vérifier que le modèle est bien installé :
```bash
ollama list
```

## 📂 Structure du projet

```
llm/
├── data/                      # Documents sources (règles UNO)
│   ├── uno_rules_1.pdf
│   ├── uno_rules_2.md
│   └── uno_rules_3.pdf
├── vectorstore/              # Généré automatiquement
│   ├── index.faiss
│   └── index.pkl
├── config.py                 # Configuration (modèles, paramètres)
├── rag_pipeline.py          # Logique RAG complète
├── app.py                   # Interface Streamlit
├── requirements.txt         # Dépendances Python
└── README.md               # Documentation
```

## 🎯 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application sera accessible sur : `http://localhost:8501`

### Premier lancement

Au premier démarrage, l'application va :
1. ✅ Télécharger le modèle d'embeddings (~80MB)
2. ✅ Créer le vectorstore à partir des documents UNO
3. ✅ Se connecter à Ollama/Mistral

Temps estimé : **2-3 minutes**

Les lancements suivants seront instantanés (vectorstore déjà créé).

### Modes d'utilisation

#### 👤 Mode User
Affiche uniquement la réponse générée, idéal pour un usage simple.

**Exemple :**
```
Question: Comment jouer un +4?
Réponse: La carte +4 sauvage peut être jouée à tout moment. 
Le joueur suivant doit piger 4 cartes et passer son tour.
```

#### 👨‍💻 Mode Developer
Affiche la réponse + le contexte RAG + les sources utilisées.

**Exemple :**
```
Question: Comment jouer un +4?
Réponse: [...]

📚 Contexte utilisé (RAG):
[Montre les chunks récupérés]

🔍 Sources récupérées:
Source 1: uno_rules_3.pdf (page 2)
Source 2: uno_rules_2.md
[...]
```

## ⚙️ Configuration

Modifiez `config.py` pour personnaliser le comportement :

```python
# Modèle d'embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Modèle LLM (doit être installé dans Ollama)
LLM_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"

# Paramètres de chunking
CHUNK_SIZE = 800          # Taille des chunks (en caractères)
CHUNK_OVERLAP = 100       # Chevauchement entre chunks

# Paramètres de retrieval
TOP_K_RESULTS = 3         # Nombre de chunks à récupérer

# Chemins
VECTORSTORE_PATH = "./vectorstore"
DATA_PATH = "./data"
```

### Changer de modèle LLM

Pour utiliser un autre modèle Ollama :

```bash
# Télécharger un modèle
ollama pull phi3
# ou
ollama pull llama2

# Modifier config.py
LLM_MODEL = "phi3"  # ou "llama2"
```

**Modèles recommandés :**
- `mistral` (7B) - ⭐ Recommandé - Excellent équilibre performance/taille
- `phi3:mini` (3.8B) - Plus léger, bonnes performances
- `llama2` (7B) - Alternative à Mistral
- `gemma:7b` (7B) - Très bon pour les instructions

## 🔄 Regénérer le vectorstore

Si vous modifiez les documents ou les paramètres de chunking :

```bash
# Windows
Remove-Item -Recurse -Force .\vectorstore

# Linux/Mac
rm -rf ./vectorstore
```

Le vectorstore sera recréé au prochain lancement.

## 🛠️ Troubleshooting

### Erreur : "Could not connect to Ollama"

**Solution :**
```bash
# Vérifier qu'Ollama est lancé
ollama list

# Si non lancé, démarrer Ollama
# Windows : Lancer l'application Ollama depuis le menu démarrer
# Linux/Mac : ollama serve
```

### Erreur : "Model not found: mistral"

**Solution :**
```bash
ollama pull mistral
```

### Vectorstore corrompu

**Solution :**
```bash
Remove-Item -Recurse -Force .\vectorstore
```

### Réponses lentes

**Solutions :**
- Utiliser un modèle plus léger (`phi3:mini`)
- Réduire `CHUNK_SIZE` dans `config.py`
- Réduire `TOP_K_RESULTS` à 2

## 📊 Performance

| Modèle | Taille | Vitesse | Qualité |
|--------|--------|---------|---------|
| Mistral 7B | ~4.1GB | 🟢 Moyenne | ⭐⭐⭐⭐⭐ Excellente |
| Phi3:mini | ~2.3GB | 🟢 Rapide | ⭐⭐⭐⭐ Très bonne |
| Llama2 7B | ~3.8GB | 🟡 Moyenne | ⭐⭐⭐⭐ Très bonne |

**Temps de réponse moyen :** 3-8 secondes (dépend du CPU)

## 🧪 Tests

Pour tester le chatbot avec différentes questions :

```
✅ Comment jouer un +4 ?
✅ Quand dois-je crier UNO ?
✅ Combien de cartes au début ?
✅ Comment gagner au UNO ?
✅ Que fait la carte Reverse ?
```

## 📦 Dépendances principales

- **streamlit** : Interface web
- **langchain** : Framework RAG
- **sentence-transformers** : Embeddings
- **faiss-cpu** : Vectorstore
- **pypdf** : Parsing PDF

Voir `requirements.txt` pour la liste complète.

## 🎯 Cas d'usage

- 📖 **Apprentissage des règles** : Nouveau joueur découvrant UNO
- 🎲 **Résolution de disputes** : Vérifier une règle pendant une partie
- 🧑‍💻 **Exemple RAG** : Template pour créer d'autres chatbots spécialisés
- 🏫 **Education** : Démonstration de RAG et LLM locaux

## 🔮 Améliorations futures

- [ ] Support de plus de formats (DOCX, TXT)
- [ ] Interface multi-langues
- [ ] Export de l'historique des questions
- [ ] API REST pour intégration
- [ ] Support GPU pour inférence plus rapide
- [ ] Mode conversation (mémoire du contexte)

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 👨‍💻 Auteur

Votre nom - [@votre-username](https://github.com/votre-username)

## 🙏 Remerciements

- [Ollama](https://ollama.ai/) pour l'infrastructure LLM locale
- [LangChain](https://www.langchain.com/) pour le framework RAG
- [Streamlit](https://streamlit.io/) pour l'interface web
- Mattel pour les règles officielles du jeu UNO

---

⭐ Si ce projet vous a aidé, n'hésitez pas à mettre une étoile !
