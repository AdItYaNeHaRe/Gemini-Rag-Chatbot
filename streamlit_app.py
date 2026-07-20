import os

import streamlit as st

from config import settings
from embeddings import embed_query
from generator import generate_answer
from ingest import discover_document_paths, ingest_paths
from vectorstore import count as vs_count
from vectorstore import query as vs_query
from vectorstore import reset as vs_reset

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
    :root {
        --bg-deep: #0f1117;
        --bg-panel: #161925;
        --bg-panel-soft: #1b1f2d;
        --border-soft: #2a2f3e;
        --accent: #3b82f6;
        --accent-strong: #2563eb;
        --accent-soft: rgba(59, 130, 246, 0.14);
        --text-primary: #e6e9f0;
        --text-secondary: #a8b0c0;
        --text-muted: #7a8296;
        --success: #4ade80;
        --danger: #f87171;
    }

    html, body, .stApp, [class*="css"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(59, 130, 246, 0.08), transparent 45%),
            radial-gradient(circle at 85% 15%, rgba(139, 92, 246, 0.06), transparent 40%),
            var(--bg-deep);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.45); }
        70% { box-shadow: 0 0 0 6px rgba(74, 222, 128, 0); }
    }

    /* ---------- Hero header ---------- */
    .hero {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 4px 0 2px;
        animation: fadeIn 0.5s ease;
    }
    .hero-icon {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        background: linear-gradient(135deg, var(--accent), #8b5cf6);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35);
        flex-shrink: 0;
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff, #b9c3d9);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero-caption {
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-top: 2px;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--bg-panel);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] h3 {
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .config-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid var(--border-soft);
        animation: fadeIn 0.4s ease;
    }
    .config-pill.ok { background: #103d24; color: var(--success); border-color: rgba(74, 222, 128, 0.25); }
    .config-pill.warn { background: #3d1010; color: var(--danger); border-color: rgba(248, 113, 113, 0.25); }
    .config-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
    .config-pill.ok .config-dot { animation: glowPulse 2s infinite; }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid var(--border-soft) !important;
        background: var(--bg-panel-soft) !important;
        color: var(--text-primary) !important;
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button:hover:not(:disabled) {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
        transform: translateY(-1px);
    }
    .stButton > button:disabled {
        opacity: 0.45 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] section {
        background: var(--bg-panel-soft) !important;
        border: 1px dashed var(--border-soft) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent) !important;
    }

    /* ---------- Metric card (kept, restyled) ---------- */
    .metric-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
        animation: fadeIn 0.4s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent);
    }
    .metric-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 2px;
        color: var(--text-primary);
    }

    /* ---------- Source chips (kept, restyled) ---------- */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #1e3a5f;
        color: #7ec8ff;
        padding: 4px 13px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin: 2px 6px 2px 0;
        border: 1px solid rgba(126, 200, 255, 0.2);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .source-chip:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }

    /* ---------- Retrieved chunk box (kept, restyled) ---------- */
    .chunk-box {
        background: var(--bg-panel);
        border: 1px solid var(--border-soft);
        border-left: 3px solid var(--accent);
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 8px;
        font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
        line-height: 1.5;
        transition: background 0.15s ease;
    }
    .chunk-box:hover { background: var(--bg-panel-soft); }
    .chunk-box b { color: var(--text-primary); font-family: "Inter", sans-serif; }

    /* ---------- Native chat bubbles ---------- */
    [data-testid="stChatMessage"] {
        background: var(--bg-panel);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 4px 6px;
        margin-bottom: 10px;
        animation: fadeIn 0.35s ease;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
    }

    /* Chat input box */
    [data-testid="stChatInput"] textarea {
        background: var(--bg-panel-soft) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    /* ---------- Empty state ---------- */
    .empty-state {
        text-align: center;
        padding: 48px 20px;
        color: var(--text-muted);
        border: 1px dashed var(--border-soft);
        border-radius: 16px;
        margin-top: 18px;
        animation: fadeIn 0.5s ease;
    }
    .empty-state .empty-icon { font-size: 2rem; margin-bottom: 8px; }
    .empty-state h4 { color: var(--text-secondary); margin: 6px 0 4px; }

    hr { border-color: var(--border-soft) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

DOCS_DIR = "documents"
os.makedirs(DOCS_DIR, exist_ok=True)


def do_retrieve(question: str) -> list[dict]:
    q_embedding = embed_query(question)
    results = vs_query(q_embedding, top_k=settings.TOP_K)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[None] * len(docs)])[0]
    return [
        {"text": text, "source": meta.get("source"), "distance": dist}
        for text, meta, dist in zip(docs, metas, distances)
    ]



with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    key_set = bool(settings.GEMINI_API_KEY)
    if key_set:
        st.markdown(
            '<div class="config-pill ok"><span class="config-dot"></span> GEMINI_API_KEY is set</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="config-pill warn"><span class="config-dot"></span> GEMINI_API_KEY is missing — add it to your .env file</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"**Chat model:** `{settings.MODEL_NAME}`")
    st.markdown(f"**Embedding model:** `{settings.EMBEDDING_MODEL}`")
    st.markdown(f"**Top-k retrieved:** `{settings.TOP_K}`")

    st.divider()
    st.markdown("### 📄 Documents")

    indexed_count = vs_count()
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Chunks indexed</div>'
        f'<div class="metric-value">{indexed_count}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload documents (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("📥 Ingest uploaded files", use_container_width=True, disabled=not (uploaded_files and key_set)):
        saved_paths = []
        for uf in uploaded_files:
            dest = os.path.join(DOCS_DIR, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(dest)

        with st.spinner(f"Embedding and indexing {len(saved_paths)} file(s)…"):
            try:
                summary = ingest_paths(saved_paths)
                st.success(f"Indexed {summary['chunk_count']} chunks from {len(saved_paths)} file(s).")
                for name, cnt in summary["files"].items():
                    st.caption(f"• {name}: {cnt} chunks")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    existing_files = [os.path.basename(p) for p in discover_document_paths()]
    if existing_files:
        st.caption("Files currently in `documents/`:")
        for name in existing_files:
            st.caption(f"  📄 {name}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Re-index all", use_container_width=True, disabled=not key_set):
            paths = discover_document_paths()
            if not paths:
                st.warning("No files in documents/ to index.")
            else:
                with st.spinner("Re-indexing all documents…"):
                    vs_reset()
                    summary = ingest_paths(paths)
                    st.success(f"Re-indexed {summary['chunk_count']} chunks.")
    with col_b:
        if st.button("🗑️ Reset store", use_container_width=True):
            vs_reset()
            st.success("Vector store cleared.")

    st.divider()
    if st.button("💬 Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">📚</div>
        <div>
            <p class="hero-title">RAG Chatbot</p>
            <p class="hero-caption">Retrieval-Augmented Generation · Gemini + Chroma</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not key_set:
    st.info("Add your Gemini API key to `.env` and restart to enable chat.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not st.session_state["messages"]:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">📚</div>
            <h4>No messages yet</h4>
            <p>Index some documents in the sidebar, then ask a question below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">{s["source"]} · {s["distance"]:.3f}</span>'
                for s in msg["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)
            with st.expander("🔍 Retrieved chunks used for this answer"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="chunk-box"><b>{s["source"]}</b> '
                        f'(distance={s["distance"]:.4f})<br><br>{s["text"]}</div>',
                        unsafe_allow_html=True,
                    )

question = st.chat_input(
    "Ask a question about your indexed documents…",
    disabled=not key_set,
)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if vs_count() == 0:
            answer = "No documents are indexed yet — upload files in the sidebar and click **Ingest uploaded files** first."
            st.warning(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": []})
        else:
            with st.spinner("Retrieving relevant chunks and generating an answer…"):
                try:
                    chunks = do_retrieve(question)
                    if not chunks:
                        answer = "I couldn't find anything relevant in the indexed documents for that question."
                        st.markdown(answer)
                        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": []})
                    else:
                        answer = generate_answer(question, chunks)
                        st.markdown(answer)
                        chips = "".join(
                            f'<span class="source-chip">{c["source"]} · {c["distance"]:.3f}</span>'
                            for c in chunks
                        )
                        st.markdown(chips, unsafe_allow_html=True)
                        with st.expander("🔍 Retrieved chunks used for this answer"):
                            for c in chunks:
                                st.markdown(
                                    f'<div class="chunk-box"><b>{c["source"]}</b> '
                                    f'(distance={c["distance"]:.4f})<br><br>{c["text"]}</div>',
                                    unsafe_allow_html=True,
                                )
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": answer, "sources": chunks}
                        )
                except Exception as e:
                    err = f"Something went wrong: {e}"
                    st.error(err)
                    st.session_state["messages"].append({"role": "assistant", "content": err, "sources": []})
