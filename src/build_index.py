import json
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# -----------------------------
# 0. Chemins & constantes
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # dossier racine du projet
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

DB_DIR = PROCESSED_DIR / "chroma_db"
COLLECTION_NAME = "rag_documents"

# Modèle d'embedding HuggingFace (sentence-transformers)
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# 1. Utilitaires
# -----------------------------

def load_chunks(path: Path) -> List[Dict]:
    """Charge la liste des chunks depuis un fichier JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier chunks non trouvé : {path}")

    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("Le fichier chunks.json doit contenir une liste.")
    return chunks


def init_embedding_model() -> SentenceTransformer:
    """
    Initialise le modèle d'embedding HuggingFace (sentence-transformers).
    Le modèle sera téléchargé la première fois, puis mis en cache.
    """
    print(f"Chargement du modèle d'embedding HuggingFace : {HF_EMBEDDING_MODEL}")
    model = SentenceTransformer(HF_EMBEDDING_MODEL)
    print(" Modèle chargé.")
    return model


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Calcule les embeddings pour une liste de textes avec sentence-transformers.
    Retourne une liste de listes de floats (compatible avec Chroma).
    """
    print(f" Calcul des embeddings pour {len(texts)} texte(s)...")
    # encode renvoie un array numpy -> on convertit en liste Python
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def init_chroma(db_dir: Path) -> "chromadb.api.client.Client":
    """Initialise un client ChromaDB persistant."""
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(
            anonymized_telemetry=False,
        ),
    )
    return client


# -----------------------------
# 2. Construction de l'index
# -----------------------------

def build_index():
    print(f" Chargement des chunks depuis : {CHUNKS_FILE}")
    chunks = load_chunks(CHUNKS_FILE)
    print(f" {len(chunks)} chunk(s) chargé(s).")

    if not chunks:
        print("Aucun chunk à indexer. Lance d'abord src/ingest.py avec des documents.")
        return

    print(" Initialisation du modèle d'embedding (HuggingFace)...")
    model = init_embedding_model()

    print(" Préparation des données pour la base vectorielle...")
    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        ids.append(str(c["chunk_global_id"]))
        documents.append(c["text"])
        metadatas.append(
            {
                "source": c.get("source"),
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
            }
        )

    print(" Calcul des embeddings (local, avec HuggingFace)...")
    embeddings = embed_texts(model, documents)
    print(f"{len(embeddings)} embedding(s) généré(s).")

    print(f"Initialisation de ChromaDB dans : {DB_DIR}")
    chroma_client = init_chroma(DB_DIR)

    print(f"Création / récupération de la collection : {COLLECTION_NAME}")
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
    )

    print("Ajout des documents dans la collection...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(" Index vectoriel construit et sauvegardé avec succès (HuggingFace + Chroma) !")


# -----------------------------
# 3. Point d'entrée
# -----------------------------

if __name__ == "__main__":
    build_index()
