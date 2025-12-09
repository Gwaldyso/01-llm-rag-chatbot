from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from openai import OpenAI
import os

# -----------------------------
# 0. Constantes & chemins
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "rag_documents"

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4.1-mini"


# -----------------------------
# 1. Utils : RAG + OpenAI
# -----------------------------

@st.cache_resource
def init_chroma_and_model():
    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier de la base vectorielle n'existe pas : {DB_DIR}.\n"
            f"Lance d'abord src/ingest.py puis src/build_index.py."
        )

    chroma_client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    model = SentenceTransformer(HF_EMBEDDING_MODEL)
    return collection, model


@st.cache_resource
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "La variable d'environnement OPENAI_API_KEY n'est pas définie.\n\n"
            "Sous PowerShell, avant de lancer Streamlit, fais par ex. :\n"
            '$env:OPENAI_API_KEY="sk_ta_cle_ici"'
        )
    client = OpenAI(api_key=api_key)
    return client


def retrieve_context(
    question: str,
    collection,
    model: SentenceTransformer,
    k: int = 3,
    max_chars_per_chunk: int = 500,
) -> str:
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return "Aucun contexte trouvé dans la base de documents."

    context_parts = []
    for doc_text, meta in zip(docs, metadatas):
        src = meta.get("source", "inconnu")

        if len(doc_text) > max_chars_per_chunk:
            display_text = doc_text[:max_chars_per_chunk] + " [...]"
        else:
            display_text = doc_text

        context_parts.append(f"[Source: {src}]\n{display_text}")

    context = "\n\n---\n\n".join(context_parts)
    return context


def call_openai_llm(prompt: str) -> str:
    client = get_openai_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant qui répond STRICTEMENT à partir du contexte fourni. "
                    "Si l'information n'est pas dans le contexte, tu dis que tu ne sais pas."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()


def generate_answer(question: str, context: str) -> str:
    if not context or "Aucun contexte trouvé" in context:
        return (
            "Je n'ai pas trouvé d'information pertinente dans tes documents pour répondre "
            "à cette question. Essaie de reformuler ou d'ajouter plus de documents."
        )

    prompt = f"""
Voici un contexte extrait de documents (résultats d'un RAG) :

{context}

Question de l'utilisateur :
{question}

Consignes :
- Réponds en FRANÇAIS.
- Appuie-toi STRICTEMENT sur les informations présentes dans le contexte.
- Si une information n'est pas dans le contexte, dis que tu ne sais pas.
- Sois clair, structuré et synthétique.
"""

    return call_openai_llm(prompt)


# -----------------------------
# 2. Interface Streamlit
# -----------------------------

def main():
    st.set_page_config(page_title="Chatbot RAG - Docs perso", page_icon="🤖")
    st.title(" Chatbot RAG sur tes documents")
    st.write(
        """
Ce chatbot utilise un pipeline **RAG (Retrieval-Augmented Generation)** :

1. Recherche de passages pertinents dans tes documents (Chroma + embeddings HuggingFace)
2. Génération d'une réponse avec **OpenAI**, basée sur ces passages.

Tu peux lui poser des questions en français sur le contenu de tes documents indexés.
"""
    )

    with st.spinner("Initialisation du RAG (Chroma + embeddings)..."):
        collection, model = init_chroma_and_model()

    question = st.text_input(" Pose ta question sur tes documents :")

    show_context = st.checkbox("Afficher le contexte RAG (debug)", value=False)

    if question:
        with st.spinner("Recherche de contexte pertinent..."):
            context = retrieve_context(question, collection, model)

        with st.spinner("Génération de la réponse avec OpenAI..."):
            answer = generate_answer(question, context)

        st.markdown("###  Réponse du chatbot")
        st.write(answer)

        if show_context:
            st.markdown("###  Contexte utilisé (chunks retrouvés)")
            st.text(context)


if __name__ == "__main__":
    main()
