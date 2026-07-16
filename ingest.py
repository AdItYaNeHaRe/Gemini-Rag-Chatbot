import glob
import os
import sys

from pypdf import PdfReader

from chunking import chunk_text
from embeddings import embed_texts
from vectorstore import add_chunks, reset

DOCS_DIR = "documents"


def load_text_from_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def ingest_paths(paths: list[str]) -> dict:
    """
    Chunks, embeds, and indexes the given file paths. Shared by both the
    CLI (`ingest.py`) and the Streamlit UI's file uploader, so there's one
    place that owns "what does ingestion actually do."

    Returns a summary dict: {"files": {basename: chunk_count}, "chunk_count": int}
    """
    all_ids, all_texts, all_metas = [], [], []
    per_file_counts = {}

    for path in paths:
        text = load_text_from_file(path)
        chunks = chunk_text(text)
        basename = os.path.basename(path)
        for i, chunk in enumerate(chunks):
            all_ids.append(f"{basename}::{i}")
            all_texts.append(chunk)
            all_metas.append({"source": basename, "chunk_index": i})
        per_file_counts[basename] = len(chunks)

    if not all_texts:
        return {"files": per_file_counts, "chunk_count": 0}

    vectors = embed_texts(all_texts, task_type="RETRIEVAL_DOCUMENT")
    add_chunks(all_ids, all_texts, vectors, all_metas)

    return {"files": per_file_counts, "chunk_count": len(all_texts)}


def discover_document_paths() -> list[str]:
    return [
        f for f in glob.glob(os.path.join(DOCS_DIR, "*"))
        if f.lower().endswith((".txt", ".pdf", ".md"))
    ]


def main():
    fresh = "--fresh" in sys.argv
    if fresh:
        print("Wiping existing vector store (--fresh)...")
        reset()

    files = discover_document_paths()
    if not files:
        print(f"No .txt/.md/.pdf files found in {DOCS_DIR}/. Add some and re-run.")
        return

    print(f"Found {len(files)} file(s) in {DOCS_DIR}/:")
    summary = ingest_paths(files)

    for basename, count in summary["files"].items():
        print(f"  {basename}: {count} chunks")

    if summary["chunk_count"] == 0:
        print("No extractable text found in the provided files.")
        return

    print(f"\nIngestion complete. {summary['chunk_count']} chunks indexed from {len(files)} file(s).")


if __name__ == "__main__":
    main()
