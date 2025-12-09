# 🧠 Chatbot documentaire avec LLM & RAG

Ce projet montre comment utiliser un **LLM** et une approche **RAG (Retrieval-Augmented Generation)** pour interroger intelligemment un corpus de documents (PDF, textes métier).

## 🎯 Objectif

Permettre à un utilisateur métier de poser des questions en langage naturel sur une base documentaire et d’obtenir des réponses :
- précises,
- sourcées,
- contextualisées par les documents d’origine.

## 🧱 Architecture

1. **Ingestion** des documents (`data/raw/`)
2. **Vectorisation** (embeddings) et création d’un index (`data/processed/`)
3. **RAG** : récupération des passages pertinents
4. **Génération de la réponse** par le LLM à partir du contexte

_Un schéma de l’architecture est disponible dans `assets/schema.png`._

## 🛠️ Stack technique

- Python
- Langage de modèle : LLM type GPT / open-source (selon dispo)
- Bibliothèques :
  - `langchain` ou équivalent
  - `faiss` / `chromadb` / autre vecteur store
  - `pandas`, `numpy`
  - `streamlit` (optionnel si interface web)

## 📁 Structure du projet

Voir l’arborescence détaillée dans le repo.

## 🚀 Lancer le projet

```bash
# Cloner le repo
git clone https://github.com/Gwaldyso/01-llm-rag-chatbot.git
cd 01-llm-rag-chatbot

# Créer et activer un environnement virtuel (optionnel mais recommandé)

# Installer les dépendances
pip install -r requirements.txt

# Lancer le script principal (exemple)
python src/app.py

