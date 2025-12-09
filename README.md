# 🤖 Chatbot Documentaire RAG — Retrieval-Augmented Generation

Ce projet implémente un **chatbot documentaire intelligent**, capable de répondre à des questions en langage naturel en utilisant vos propres documents (PDF, textes, rapports métier…).

Il repose sur une architecture **RAG (Retrieval-Augmented Generation)** combinant :

- embeddings HuggingFace,
- une base vectorielle ChromaDB,
- et un modèle de génération OpenAI (GPT-4.1-mini par défaut).

Ce type de pipeline est aujourd’hui utilisé en entreprise pour :  
- automatiser du support,  
- analyser des documents internes,  
- interroger des bases documentaires métier,  
- créer des assistants LLM privés.

---

# 🎯 Objectif

Permettre à un utilisateur de poser des questions naturelles sur ses documents et d’obtenir des réponses :

- précises  
- contextualisées  
- sourcées par des extraits réels  

---

# 🧱 Architecture du projet

Voici le pipeline complet :




# 🛠️ Stack technique

### **Langages & Frameworks**
- Python 3.10+
- Streamlit (application web)

### **LLM & NLP**
- HuggingFace SentenceTransformers → `all-MiniLM-L6-v2` pour les embeddings  
- OpenAI GPT-4.1-mini (ou tout autre modèle compatible) pour la génération

### **Vector Database**
- ChromaDB (persistant)

### **Autres bibliothèques**
- `pandas`, `numpy`
- `openai`
- `chromadb`
- `sentence-transformers`



# 📂 Structure du projet


01-llm-rag-chatbot/
 ├── data/
 │   ├── raw/                  # Fichiers d’entrée (PDF, TXT…)
 │   └── processed/
 │        └── chroma_db/       # Base vectorielle persistante
 ├── src/
 │   ├── ingest.py             # Extraction texte + création des chunks
 │   ├── build_index.py        # Embeddings + insertion dans Chroma
 │   ├── app.py                # Version console du chatbot RAG
 │   └── app_streamlit.py      # Application Streamlit
 ├── requirements.txt
 ├── .gitignore
 └── README.md                 # (ce fichier)


## 🚀 Lancer le projet
bash
# Cloner le repo
git clone https://github.com/Gwaldyso/01-llm-rag-chatbot.git
cd 01-llm-rag-chatbot

# Créer et activer un environnement virtuel (optionnel mais recommandé)

# Installer les dépendances
pip install -r requirements.txt

# Lancer le script principal (exemple)
python src/app.py
