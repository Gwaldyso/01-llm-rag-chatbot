# Chatbot documentaire avec LLM & RAG

Ce projet montre comment utiliser un **LLM** et une approche **RAG (Retrieval-Augmented Generation)** pour interroger intelligemment un corpus de documents (PDF, textes métier).

##  Objectif

Permettre à un utilisateur métier de poser des questions en langage naturel sur une base documentaire et d’obtenir des réponses : précises,sourcées et contextualisées par les documents d’origine.
##  Architecture

1. **Ingestion** des documents (`data/raw/`)
2. **Vectorisation** (embeddings) et création d’un index (`data/processed/`)
3. **RAG** : récupération des passages pertinents
4. **Génération de la réponse** par le LLM à partir du contexte

_Un schéma de l’architecture est disponible dans `assets/schema.png`._

##  Stack technique

- Python
- Langage de modèle : LLM type GPT / open-source (selon dispo)
- Bibliothèques :
  - `langchain` ou équivalent
  - `faiss` / `chromadb` / autre vecteur store
  - `pandas`, `numpy`
  - `streamlit` (optionnel si interface web)

##  Structure du projet

Voir l’arborescence détaillée dans le repo.

##  Lancer le projet


# Cloner le repo
git clone https://github.com/Gwaldyso/01-llm-rag-chatbot.git
cd 01-llm-rag-chatbot

# Créer et activer un environnement virtuel (optionnel mais recommandé)

#  Chatbot Documentaire RAG — Retrieval-Augmented Generation

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


#  Structure du projet


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


##  Lancer le projet
bash
# Cloner le repo
- git clone https://github.com/Gwaldyso/01-llm-rag-chatbot.git
- cd 01-llm-rag-chatbot

# Créer et activer un environnement virtuel (optionnel mais recommandé)

- python -m venv .venv
- .\.venv\Scripts\activate    # Windows


>>>>>>> 3409a5748b4ccdf7bafb835a8a1594003e8564af
# Installer les dépendances
pip install -r requirements.txt

# Lancer le script principal (exemple)
python src/app.py

## Pipeline d’utilisation
# Étape 1 — Déposer vos documents

Placez vos fichiers PDF/TXT dans :

data/raw/

---

# Étape 2 — Ingestion & chunking
- Exécutez le script d’ingestion pour extraire le texte et créer les chunks :
- Le script génère automatiquement :

---

# Étape 3 — Construction de l’index vectoriel
- Construisez l’index (embeddings + stockage ChromaDB) :
- L’index vectoriel persistant sera créé dans :

---

# Étape 4 — Version console (CLI)
- Configurez votre clé OpenAI : 
- Lancez le chatbot en mode console :

---

# Étape 5 — Version web (Streamlit)
- Exportez votre clé OpenAI :
- Lancez l’interface web :


- Accédez ensuite à :  
   http://localhost:8501

- Vous verrez :
  - une zone pour poser vos questions  
  - une réponse générée par le LLM  
  - un mode debug pour voir les chunks utilisés  


## ⚙️ Fonctionnalités

###  Chatbot documentaire intelligent
- Posez des questions en langage naturel sur vos documents.
- Obtenez des réponses contextualisées et structurées.

###  Pipeline RAG complet
- Ingestion et découpage des documents en chunks.
- Vectorisation (embeddings) et indexation dans une base vectorielle.
- Recherche sémantique des passages les plus pertinents.
- Génération de la réponse en s’appuyant sur le contexte.

###  Deux modes d’utilisation
- Mode console (CLI) : interaction dans le terminal.
- Interface web Streamlit : chatbot accessible via navigateur.

###  Architecture modulaire
- Scripts séparés pour :
  - l’ingestion (`ingest.py`)
  - la construction de l’index (`build_index.py`)
  - le chatbot console (`app.py`)
  - le chatbot web (`app_streamlit.py`)


##  Fonctionnement détaillé

###  Embeddings — HuggingFace MiniLM-L6-v2
- Modèle : `sentence-transformers/all-MiniLM-L6-v2`
- Caractéristiques :
  - rapide
  - léger
  - excellent rapport qualité / vitesse
- Utilisation :
  - transforme les questions et les chunks de texte en vecteurs numériques.
  - permet de mesurer la similarité entre la question et les passages de documents.

---

###  Vectorisation & Recherche — ChromaDB
- Base vectorielle : **ChromaDB** (mode persistant).
- Rôle :
  - stocker les embeddings des chunks.
  - faire de la recherche de similarité (kNN) pour retrouver les passages les plus proches de la question.
- Avantages :
  - simple à utiliser en Python.
  - adaptée aux projets RAG de petite et moyenne taille.

---

###  Génération — OpenAI GPT-4.1-mini
- Modèle utilisé : `gpt-4.1-mini` (configurable).
- Rôle :
  - recevoir un **prompt** contenant :
    - le contexte (chunks retrouvés)
    - la question utilisateur
    - des consignes de réponse (en français, strictement basée sur le contexte).
  - générer une réponse claire, synthétique et structurée.
- Avantages :
  - bonne qualité de langage.
  - adapté à des tâches de résumé, réponse à des questions, reformulation.

---

##  Points forts du projet

###  Architecture RAG complète
- Couverture de toutes les étapes :
  - ingestion
  - embeddings
  - indexation
  - retrieval
  - génération

###  Combinaison de technologies modernes
- Embeddings HuggingFace + ChromaDB + OpenAI :
  - montre une bonne compréhension des outils actuels de l’IA générative.
  - facilement transposable dans un contexte entreprise.

###  Code structuré et pédagogique
- Séparation claire des responsabilités :
  - `ingest.py` pour la préparation des données.
  - `build_index.py` pour la construction de l’index.
  - `app.py` pour la version console.
  - `app_streamlit.py` pour la version web.
- Facile à lire, à maintenir et à étendre.







