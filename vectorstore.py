import chromadb

from config import settings

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        _collection = _client.get_or_create_collection("documents")
    return _collection


def add_chunks(ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
    """
    Uses upsert rather than add: if a chunk ID already exists (e.g. the
    same filename is re-ingested after being edited), add() would silently
    keep the OLD content and report success anyway -- upsert correctly
    replaces it with the new content instead.
    """
    collection = get_collection()
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def query(embedding: list[float], top_k: int = None):
    top_k = top_k or settings.TOP_K
    collection = get_collection()
    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    return collection.query(query_embeddings=[embedding], n_results=min(top_k, collection.count()))


def count() -> int:
    return get_collection().count()


def reset():
    """Wipes the collection -- useful when re-ingesting from scratch."""
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    try:
        _client.delete_collection("documents")
    except Exception:
        pass
    _collection = _client.get_or_create_collection("documents")
