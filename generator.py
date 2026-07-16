from google import genai

from config import settings

_client: genai.Client | None = None

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context below. If the answer is not contained in the "
    "context, say you don't know rather than guessing. Keep answers "
    "concise and mention which source(s) the answer came from."
)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    client = _get_client()

    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    interaction = client.interactions.create(model=settings.MODEL_NAME, input=prompt)
    return interaction.output_text
