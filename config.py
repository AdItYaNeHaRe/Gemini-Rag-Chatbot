import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
    TOP_K = int(os.getenv("TOP_K", "4"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


settings = Settings()
