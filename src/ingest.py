import os
import json
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


# Dossiers de travail
BASE_DIR = Path(__file__).resolve().parent.parent  # dossier 01-llm-rag-chatbot
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"


# -----------------------------
# 1. Chargement des documents
# -----------------------------

def load_txt_file(path: Path) -> str:
    """Lit un fichier .txt et renvoie son contenu en une seule chaîne."""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(path: Path) -> str:
    """Lit un PDF et concatène le texte de toutes les pages."""
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def load_all_documents(raw_dir: Path) -> List[Dict]:
    """
    Parcourt le dossier data/raw et charge tous les fichiers .txt et .pdf.
    Retourne une liste de dictionnaires :
    [{"id": ..., "source": ..., "text": ...}, ...]
    """
    documents = []
    doc_id = 0

    for path in raw_dir.glob("*"):
        if path.suffix.lower() == ".txt":
            text = load_txt_file(path)
        elif path.suffix.lower() == ".pdf":
            text = load_pdf_file(path)
        else:
            # On ignore les autres types pour l'instant
            continue

        if not text.strip():
            # On ignore les fichiers vides
            continue

        documents.append(
            {
                "id": doc_id,
                "source": path.name,
                "text": text,
            }
        )
        doc_id += 1

    return documents


# -----------------------------
# 2. Découpage en chunks
# -----------------------------

def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[str]:
    """
    Découpe un long texte en morceaux (chunks) de longueur ~chunk_size
    avec un recouvrement (overlap) entre les chunks.

    Si le texte est plus court que chunk_size, on renvoie simplement
    un seul chunk.
    """
    if not text:
        return []

    words = text.split()
    n = len(words)
    chunks = []

    # Cas simple : texte plus court qu'un chunk
    if n <= chunk_size:
        chunk = " ".join(words).strip()
        if chunk:
            chunks.append(chunk)
        return chunks

    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)

        # Si on est arrivé à la fin du texte, on sort
        if end == n:
            break

        # Sinon on avance en tenant compte de l'overlap
        start = end - overlap
        if start < 0:
            start = 0

    return chunks



def create_chunks_from_documents(documents: List[Dict]) -> List[Dict]:
    """
    Prend la liste de documents et renvoie une liste de chunks
    avec métadonnées :
    [
      {
        "doc_id": ...,
        "chunk_id": ...,
        "source": ...,
        "text": ...,
      },
      ...
    ]
    """
    all_chunks = []
    chunk_global_id = 0

    for doc in documents:
        doc_id = doc["id"]
        source = doc["source"]
        text = doc["text"]

        chunks = split_text_into_chunks(text)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_global_id": chunk_global_id,
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "source": source,
                    "text": chunk_text,
                }
            )
            chunk_global_id += 1

    return all_chunks


# -----------------------------
# 3. Sauvegarde des chunks
# -----------------------------

def save_chunks(chunks: List[Dict], out_path: Path) -> None:
    """Sauvegarde la liste de chunks au format JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


# -----------------------------
# 4. Point d'entrée
# -----------------------------

def main():
    print(f" Dossier RAW : {RAW_DIR}")
    print(" Chargement des documents...")

    documents = load_all_documents(RAW_DIR)
    print(f" {len(documents)} document(s) chargé(s).")

    if not documents:
        print(" Aucun document trouvé dans data/raw. Ajoute un .txt ou .pdf et relance.")
        return

    print(" Découpage en chunks...")
    chunks = create_chunks_from_documents(documents)
    print(f"{len(chunks)} chunk(s) généré(s).")

    print(f" Sauvegarde dans {CHUNKS_FILE} ...")
    save_chunks(chunks, CHUNKS_FILE)
    print("Terminé ! Les chunks sont prêts pour la création de l'index.")


if __name__ == "__main__":
    main()
