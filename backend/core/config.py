
import os
from dotenv import load_dotenv

load_dotenv()
print("GROQ key", os.getenv("GROQ_API_KEY"))

# Core model configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ API KEY MISSING IN .ENV")

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
MAX_RESEARCH_ITERS = int(os.getenv("MAX_RESEARCH_ITERS", "1"))

# Storage locations (allow override via env, default under ./storage)
DATA_DIR = os.getenv("DATA_DIR", "./storage")
DENSE_INDEX_DIR = os.getenv("DENSE_INDEX_DIR", os.path.join(DATA_DIR, "index", "dense"))
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", os.path.join(DATA_DIR, "index", "bm25"))
MEMORY_STORE_DIR = os.getenv("MEMORY_STORE_DIR", os.path.join(DATA_DIR, "memory"))
PAGE_STORE_DIR = os.getenv("PAGE_STORE_DIR", os.path.join(DATA_DIR, "pages"))

print("Config loaded successfully!")
print(f"Model: {MODEL_NAME} | Using GROQ Key: {GROQ_API_KEY[:10]}...{GROQ_API_KEY[-4:]}")
print(f"Data dir: {os.path.abspath(DATA_DIR)}")