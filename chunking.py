from config import settings


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Splits text into overlapping fixed-size character chunks.

    Overlap exists so a fact that happens to fall right at a chunk boundary
    isn't lost entirely from both sides -- each chunk shares `overlap`
    characters with its neighbor.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks
