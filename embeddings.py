from google import genai
from google.genai import types

from config import settings

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Embeds a batch of texts. task_type matters for retrieval quality:
    use RETRIEVAL_DOCUMENT when indexing chunks, RETRIEVAL_QUERY when
    embedding a user's question -- Gemini's embedding model optimizes the
    vector space differently for each role.
    """
    if not texts:
        return []
    client = _get_client()
    response = client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in response.embeddings]


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
