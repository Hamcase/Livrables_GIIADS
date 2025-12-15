"""
Interface Streamlit pour le chatbot UNO RAG
"""
import streamlit as st
from rag_pipeline import UNOChatbot

# Configuration de la page
st.set_page_config(
    page_title="UNO RAG Chatbot", 
    page_icon="🎮", 
    layout="centered"
)

# Initialisation du chatbot (mise en cache)
@st.cache_resource
def init_chatbot():
    chatbot = UNOChatbot()
    chatbot.create_vectorstore()
    chatbot.load_llm()
    return chatbot

# Interface principale
st.title("🎮 UNO Chatbot RAG")
st.markdown("*Posez vos questions sur les règles du jeu UNO*")

# Sélection du mode
mode = st.radio(
    "Mode d'affichage:",
    ["👤 User", "👨‍💻 Developer"],
    horizontal=True
)
dev_mode = mode == "👨‍💻 Developer"

# Chargement du chatbot
with st.spinner("⏳ Chargement des modèles... (peut prendre 1-2 minutes au premier lancement)"):
    chatbot = init_chatbot()

st.success("✅ Chatbot prêt !")

# Zone de saisie
question = st.text_input(
    "Votre question:",
    placeholder="Ex: Comment jouer un +4 ? Peut-on contester un +4 ?"
)

# Bouton d'envoi
if st.button("🚀 Envoyer", type="primary") and question:
    with st.spinner("🤔 Recherche et génération de la réponse..."):
        result = chatbot.query(question, dev_mode=dev_mode)
        
        if dev_mode:
            # Mode développeur : affichage détaillé
            st.subheader("💬 Réponse:")
            st.write(result["answer"])
            
            with st.expander("📚 Contexte utilisé (RAG)"):
                st.text_area("", result["context"], height=200)
            
            with st.expander("🔍 Sources récupérées"):
                for i, source in enumerate(result["sources"], 1):
                    st.markdown(f"**📄 Source {i}:**")
                    st.text_area(f"source_{i}", source["content"], height=100, key=f"src_{i}")
                    st.json(source["metadata"])
        else:
            # Mode utilisateur : réponse simple
            st.success("💬 Réponse:")
            st.write(result)

# Sidebar avec informations
with st.sidebar:
    st.header("ℹ️ Informations")
    st.markdown("""
    **Modèles utilisés:**
    - 🧠 LLM: Mistral 7B (via Ollama)
    - 📊 Embeddings: all-MiniLM-L6-v2
    
    **Mode User:** Réponses simples
    
    **Mode Developer:** 
    - Réponse complète
    - Contexte RAG utilisé
    - Sources avec métadonnées
    
    **Performance:**
    - 🔒 100% local
    - ⚡ Temps de réponse: 3-8s
    """)
    
    st.markdown("---")
    st.caption("🎮 Chatbot RAG local - UNO")
