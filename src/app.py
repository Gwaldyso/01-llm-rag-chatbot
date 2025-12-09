import os
from pathlib import Path
from typing import Tuple


from openai import OpenAI
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests


# -----------------------------
# 0. Constantes & chemins
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "rag_documents"

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Active ou non le mode debug (affichage du contexte brut)
DEBUG_SHOW_CONTEXT = False

# Modèle de chat OpenAI (tu peux utiliser gpt-4.1-mini ou gpt-4o-mini par ex.)
OPENAI_MODEL = "gpt-4.1-mini"
_openai_client = None


def get_openai_client() -> OpenAI:
    """
    Initialise un client OpenAI en utilisant la variable d'environnement OPENAI_API_KEY.
    """
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "La variable d'environnement OPENAI_API_KEY n'est pas définie.\n"
            "Dans ton PowerShell :\n"
            '$env:OPENAI_API_KEY="sk_ta_cle_ici"'
        )

    _openai_client = OpenAI(api_key=api_key)
    return _openai_client

# -----------------------------
# 1. Initialisation Chroma + modèle embeddings
# -----------------------------

def init_chroma_and_model() -> Tuple[object, SentenceTransformer]:
    """
    Initialise :
    - le client Chroma persistant (base vectorielle)
    - la collection qui contient tes chunks
    - le modèle d'embeddings HuggingFace (pour la question)
    """
    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier de la base vectorielle n'existe pas : {DB_DIR}.\n"
            f"Lance d'abord src/ingest.py puis src/build_index.py."
        )

    print(f"💾 Connexion à ChromaDB dans : {DB_DIR}")
    chroma_client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Impossible de récupérer la collection {COLLECTION_NAME} : {e}"
        )

    print(f"🧠 Chargement du modèle d'embedding : {HF_EMBEDDING_MODEL}")
    model = SentenceTransformer(HF_EMBEDDING_MODEL)
    print("✅ Modèle d'embedding chargé.")

    return collection, model


# -----------------------------
# 2. Récupération de contexte (RAG)
# -----------------------------

def retrieve_context(
    question: str,
    collection,
    model: SentenceTransformer,
    k: int = 3,
    max_chars_per_chunk: int = 500,
) -> str:
    """
    - Encode la question
    - Récupère les k chunks les plus pertinents
    - Tronque chaque chunk à max_chars_per_chunk
    - Concatène le tout dans une seule chaîne de contexte
    """
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

        # Tronque le texte pour éviter de passer des pavés immenses au LLM
        if len(doc_text) > max_chars_per_chunk:
            display_text = doc_text[:max_chars_per_chunk] + " [...]"
        else:
            display_text = doc_text

        context_parts.append(f"[Source: {src}]\n{display_text}")

    context = "\n\n---\n\n".join(context_parts)
    return context


# -----------------------------
# 3. Appel LLM 
# -----------------------------

def call_openai_llm(prompt: str) -> str:
    """
    Appelle le modèle de chat OpenAI avec un prompt texte.
    """
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


# -----------------------------
# 4. Génération de réponse finale
# -----------------------------

def generate_answer(question: str, context: str) -> str:
    """
    Version avec vrai LLM OpenAI.

    - Si le contexte est vide : on le signale.
    - Sinon : on construit un prompt clair et on appelle le LLM.
    """
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
    try:
        llm_answer = call_openai_llm(prompt)
        return llm_answer
    except Exception as e:
        return (
            "Une erreur est survenue lors de l'appel à l'API OpenAI : "
            f"{e}\n\nLe RAG (recherche de contexte) fonctionne, "
            "mais la génération de texte n'a pas abouti."
        )





# -----------------------------
# 6. Boucle principale (console)
# -----------------------------

def main():
    print("🚀 Démarrage du chatbot RAG (version console + LLM OpenAI)")
    collection, model = init_chroma_and_model()

    print("\nTu peux poser des questions sur tes documents.")
    print("Tape 'exit' ou 'quit' pour quitter.\n")

    while True:
        question = input("❓ Question : ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Fin du chatbot. À bientôt !")
            break

        if not question:
            continue

        print("🔎 Recherche de contexte pertinent...")
        context = retrieve_context(question, collection, model)

        print("🧠 Appel du LLM OpenAI pour générer la réponse...")
        answer = generate_answer(question, context)

        print("\n================= RÉPONSE =================\n")
        print(answer)
        print("\n===========================================\n")

        if DEBUG_SHOW_CONTEXT:
            print("=========== CONTEXTE (DEBUG) ===========\n")
            print(context)
            print("\n=======================================\n")


if __name__ == "__main__":
    main()
