"""
Pipeline RAG pour le chatbot UNO
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import os
import config


class UNOChatbot:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        
    def load_documents(self):
        """Charge les documents du dossier data"""
        documents = []
        for file in os.listdir(config.DATA_PATH):
            file_path = os.path.join(config.DATA_PATH, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(('.txt', '.md')):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        print(f"✓ {len(documents)} documents chargés")
        return documents
    
    def create_vectorstore(self):
        """Crée ou charge le vectorstore"""
        print("Initialisation des embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Vérifier si le fichier index.faiss existe
        index_file = os.path.join(config.VECTORSTORE_PATH, "index.faiss")
        if os.path.exists(index_file):
            print("Chargement du vectorstore existant...")
            self.vectorstore = FAISS.load_local(
                config.VECTORSTORE_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✓ Vectorstore chargé")
        else:
            print("Création du vectorstore...")
            documents = self.load_documents()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            print(f"✓ {len(splits)} chunks créés")
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            os.makedirs(config.VECTORSTORE_PATH, exist_ok=True)
            self.vectorstore.save_local(config.VECTORSTORE_PATH)
            print("✓ Vectorstore sauvegardé")
        
    def load_llm(self):
        """Charge le LLM via Ollama"""
        print(f"Connexion à Ollama avec le modèle {config.LLM_MODEL}...")
        self.llm = Ollama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.3,
            num_predict=200
        )
        print("✓ LLM Ollama connecté")
    
    def query(self, question: str, dev_mode: bool = False):
        """Effectue une requête RAG"""
        # Récupération des documents pertinents
        docs = self.vectorstore.similarity_search(question, k=config.TOP_K_RESULTS)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Prompt optimisé pour Mistral
        prompt = f"""[INST] Tu es un assistant expert du jeu UNO. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.

Contexte (règles UNO):
{context}

Question: {question}

Réponds de manière concise et précise (2-3 phrases maximum). Si l'information n'est pas dans le contexte, dis-le clairement. [/INST]"""
        
        # Génération avec paramètres plus stricts
        response = self.llm.invoke(prompt)
        
        # Ollama retourne du texte propre, nettoyage minimal
        answer = response.strip() if isinstance(response, str) else str(response)
        
        if dev_mode:
            return {
                "answer": answer,
                "context": context,
                "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            }
        return answer
