from embeddings import embed_query
from generator import generate_answer
from vectorstore import count, query
from config import settings


def retrieve(question: str) -> list[dict]:
    q_embedding = embed_query(question)
    results = query(q_embedding, top_k=settings.TOP_K)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[None] * len(docs)])[0]

    return [
        {"text": text, "source": meta.get("source"), "distance": dist}
        for text, meta, dist in zip(docs, metas, distances)
    ]


def main():
    if count() == 0:
        print("No documents indexed yet. Run `python ingest.py` first.\n")
        return

    print("RAG chatbot ready. Ask a question about your indexed documents.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        chunks = retrieve(question)
        if not chunks:
            print("Bot: I don't have any relevant indexed content for that.\n")
            continue

        answer = generate_answer(question, chunks)

        print(f"\nBot: {answer}\n")
        print("Sources used:")
        for c in chunks:
            dist_str = f" (distance={c['distance']:.4f})" if c["distance"] is not None else ""
            print(f"  - {c['source']}{dist_str}")
        print()


if __name__ == "__main__":
    main()
