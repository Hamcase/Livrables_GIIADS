# Configuration pour le chatbot RAG UNO

# Modèle d'embeddings léger (80MB)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM via Ollama (doit être pullé avec: ollama pull mistral)
LLM_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"

# Paramètres de chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Paramètres de retrieval
TOP_K_RESULTS = 3

# Chemins
VECTORSTORE_PATH = "./vectorstore"
DATA_PATH = "./data"
