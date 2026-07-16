# 📚 L2-01 — RAG Chatbot with Gemini & ChromaDB

A Retrieval-Augmented Generation (RAG) chatbot built with **Google Gemini**, **ChromaDB**, and **Streamlit** that answers questions using your own documents instead of relying solely on an LLM's general knowledge.

The application indexes PDF, Markdown, and text documents into a vector database using Gemini embeddings. When a user asks a question, the system retrieves the most relevant document chunks and uses them as context for Gemini to generate accurate, grounded responses.

This project demonstrates the complete RAG pipeline, including document ingestion, chunking, embeddings, vector search, prompt grounding, and conversational querying through both a command-line interface and an interactive Streamlit application.

---

# ✨ Features

- 📄 Upload PDF, TXT, and Markdown documents
- 🧠 Gemini Embedding API integration
- 🔍 Semantic search using ChromaDB
- 💬 Grounded AI responses with Gemini
- 📚 Source citation for every answer
- 🛡 Hallucination prevention through prompt grounding
- ⚡ Persistent vector database
- 🖥 Interactive Streamlit interface
- 📊 Live indexed document statistics
- 🔄 Re-index and reset functionality
- 💻 Command Line Interface (CLI)

---

# 🏗️ Architecture

The project consists of two independent pipelines sharing the same vector database.

## 1. Document Ingestion Pipeline

```
PDF / TXT / Markdown Files
            │
            ▼
      Text Extraction
            │
            ▼
      Chunking Engine
            │
            ▼
Gemini Embeddings
(RETRIEVAL_DOCUMENT)
            │
            ▼
        ChromaDB
```

Documents are processed only once during ingestion.

---

## 2. Query Pipeline

```
User Question
      │
      ▼
Gemini Embeddings
(RETRIEVAL_QUERY)
      │
      ▼
Semantic Search
 (ChromaDB)
      │
      ▼
Top-K Relevant Chunks
      │
      ▼
Grounded Prompt
      │
      ▼
Gemini LLM
      │
      ▼
Final Answer
```

Instead of asking Gemini directly, the chatbot first retrieves the most relevant document chunks and injects them into the prompt, ensuring responses remain grounded in the indexed documents.

---

# 📂 Project Structure

```
rag-chatbot/
│
├── documents/
│   └── company_policy.txt
│
├── config.py
├── chunking.py
├── embeddings.py
├── vectorstore.py
├── ingest.py
├── generator.py
├── chat.py
├── streamlit_app.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-chatbot
```

---

## 2. Create a Virtual Environment

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file.

```env
GEMINI_API_KEY=your-api-key
MODEL_NAME=gemini-3.5-flash
EMBEDDING_MODEL=gemini-embedding-001

CHROMA_DIR=chroma_db

TOP_K=4
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

Generate an API key from:

https://aistudio.google.com/apikey

---

# 🚀 Running the Application

## Option 1 — Streamlit Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Open:

```
http://localhost:8501
```

### Streamlit Features

- 📂 Upload PDF, TXT, and Markdown files
- 📊 Live indexed chunk count
- 🔄 Re-index all documents
- 🗑 Reset vector database
- 💬 Modern chat interface
- 📚 Retrieved source references
- 📄 Expandable retrieved chunk viewer
- ⚙ Live configuration status

Every answer displays exactly which document chunks were used, making the retrieval process completely transparent.

---

## Option 2 — Command Line Interface

### Step 1

Place your documents inside:

```
documents/
```

Supported formats:

- PDF
- TXT
- Markdown

---

### Step 2

Index the documents

```bash
python ingest.py
```

To rebuild the vector database:

```bash
python ingest.py --fresh
```

---

### Step 3

Start chatting

```bash
python chat.py
```

---

# 🔄 RAG Workflow

```
User Question
      │
      ▼
Embedding Generation
      │
      ▼
Vector Search
      │
      ▼
Top-K Chunks Retrieved
      │
      ▼
Prompt Construction
      │
      ▼
Gemini Response
      │
      ▼
Answer + Sources
```

The chatbot never answers directly without consulting the indexed knowledge base.

---

# 🧠 Retrieval Strategy

## Intelligent Chunking

Documents are split into fixed-size chunks.

```
Chunk Size      : 800 characters
Chunk Overlap   : 100 characters
```

Chunk overlap helps preserve context that spans chunk boundaries.

---

## Separate Embeddings for Storage and Search

Gemini supports two embedding modes:

### Document Embeddings

```
RETRIEVAL_DOCUMENT
```

Used while indexing documents.

### Query Embeddings

```
RETRIEVAL_QUERY
```

Used when searching.

Using different embedding modes significantly improves retrieval quality.

---

## Top-K Retrieval

The chatbot retrieves:

```
TOP_K = 4
```

document chunks for every query.

This value can easily be tuned through the `.env` configuration.

---

## Grounded Prompt Engineering

The retrieved chunks are inserted into a carefully designed prompt instructing Gemini to:

- Answer only from the provided context
- Never fabricate information
- Respond with **"I don't know"** when the answer is absent

This provides an additional safeguard against hallucinations.

---

## Persistent Vector Store

ChromaDB stores embeddings on disk, allowing documents to remain indexed across application restarts.

---

# 📊 Example Conversation

### User

```
How much is the home office reimbursement?
```

### Assistant

```
Employees working remotely can claim up to INR 15,000 for home office setup.

Source:
company_policy.txt
```

---

### User

```
How many sick days do employees receive?
```

### Assistant

```
Full-time employees receive 12 paid sick days each year.

Source:
company_policy.txt
```

---

### User

```
What is the parental leave policy?
```

### Assistant

```
I don't know.

The indexed documents do not contain information about parental leave.
```

This demonstrates one of the most important characteristics of a RAG system: refusing to hallucinate when the required information is unavailable.

---

# 🎯 Design Decisions

## ChromaDB

ChromaDB was selected because it:

- persists automatically
- stores metadata
- requires minimal setup
- integrates easily with Python

---

## Upsert Instead of Add

The vector store uses **upsert()** rather than **add()**.

This ensures updated files replace previous embeddings instead of creating duplicate entries.

---

## Independent Queries

Each question is processed independently.

The chatbot does not currently maintain conversational memory, keeping retrieval simple and predictable.

---

## Transparent Source Attribution

Every generated answer includes:

- source filename
- similarity score
- retrieved text chunks

allowing users to verify why the model produced a particular response.

---

# ⚠️ Current Limitations

- No reranking after retrieval
- No conversational memory
- Best-effort PDF extraction
- No OCR for scanned PDFs
- Duplicate content across renamed files is not automatically removed

---

# 🧪 Example Questions

## Company Policy

```
How many sick days do employees receive?
```

```
How much is the home office reimbursement?
```

---

## CloudSync Guide

```
What encryption is used for data at rest?
```

```
What 2FA methods does CloudSync support?
```

---

## Cross-Document Demonstration

Ask these back-to-back:

```
How many sick days do employees receive?
```

```
What 2FA methods does CloudSync support?
```

Notice how the retrieved sources switch between different documents without mixing unrelated information.

---

## Hallucination Test

Ask:

```
Does CloudSync support real-time collaborative editing?
```

The chatbot should respond:

```
I don't know.
```

rather than inventing an answer, demonstrating proper grounding.

---

# 🚀 Future Enhancements

- Hybrid search (Keyword + Vector)
- Cross-encoder reranking
- Conversation memory
- Metadata filtering
- OCR for scanned PDFs
- Streaming responses
- Multi-document summarization
- Citation highlighting
- Docker deployment
- Kubernetes support
- Authentication
- API endpoints for external applications

---

# 🛠 Technologies Used

- Python
- Google Gemini API
- Gemini Embeddings
- ChromaDB
- Streamlit
- PyPDF
- dotenv

---

# 📄 License

This project was developed for learning and demonstration purposes to showcase Retrieval-Augmented Generation (RAG) using Google Gemini and ChromaDB.
