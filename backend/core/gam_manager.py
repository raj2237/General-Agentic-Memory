import os
import sys
import shutil
import warnings
import logging

# Suppress FlagEmbedding warnings
warnings.filterwarnings('ignore', category=UserWarning, module='FlagEmbedding')
warnings.filterwarnings('ignore', message='.*query_instruction_format.*')

# Suppress transformers tokenizer warnings
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)

from gam import (
    MemoryAgent, ResearchAgent, OpenAIGenerator, OpenAIGeneratorConfig,
    InMemoryMemoryStore, InMemoryPageStore, MemoryState,
    DenseRetrieverConfig, DenseRetriever, BM25Retriever
)
from .config import (
    GROQ_API_KEY, MODEL_NAME, EMBEDDING_MODEL, MAX_RESEARCH_ITERS,
    DENSE_INDEX_DIR, BM25_INDEX_DIR, MEMORY_STORE_DIR, PAGE_STORE_DIR
)

class GAMManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    async def init(self):
        if self.initialized:
            return self

        # Generator config - optimized for speed
        gen_config = OpenAIGeneratorConfig(
            model_name=MODEL_NAME,
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.1,
            max_tokens=512
        )
        self.generator = OpenAIGenerator.from_config(gen_config)

        # Ensure storage directories exist
        os.makedirs(DENSE_INDEX_DIR, exist_ok=True)
        os.makedirs(BM25_INDEX_DIR, exist_ok=True)
        os.makedirs(MEMORY_STORE_DIR, exist_ok=True)

        # Database-backed page store
        from storage.chunk_db import ChunkDB
        from storage.db_page_store import DatabasePageStore
        
        self.chunk_db = ChunkDB()
        self.page_store = DatabasePageStore(self.chunk_db)
        
        # Memory store
        self.memory_store = InMemoryMemoryStore(dir_path=MEMORY_STORE_DIR)

        self.memory_agent = MemoryAgent(
            generator=self.generator,
            memory_store=self.memory_store,
            page_store=self.page_store
        )

        # Dense retriever - lazy loading
        dense_config = DenseRetrieverConfig(
            model_name=EMBEDDING_MODEL,
            index_dir=DENSE_INDEX_DIR
        )
        self.dense_retriever = DenseRetriever(dense_config.__dict__)
        
        try:
            self.dense_retriever.load()
        except Exception:
            pass  # Will build on first document

        # BM25 retriever - keyword search
        bm25_config = {
            "index_dir": BM25_INDEX_DIR,
            "threads": 4
        }
        
        try:
            self.bm25_retriever = BM25Retriever(bm25_config)
            self.bm25_retriever.load()
        except Exception:
            # Create dummy retriever if BM25 unavailable
            class DummyBM25:
                def update(self, *args, **kwargs): pass
                def build(self, *args, **kwargs): pass
                def search(self, queries, *args, **kwargs): return [[] for _ in queries]
                def clear(self): pass
            self.bm25_retriever = DummyBM25()

        # Hybrid retrieval: DenseRetriever + BM25Retriever
        self.research_agent = ResearchAgent(
            page_store=self.page_store,
            memory_store=self.memory_store,
            retrievers={
                "dense": self.dense_retriever,
                "bm25": self.bm25_retriever
            },  
            generator=self.generator,
            max_iters=MAX_RESEARCH_ITERS
        )

        self.initialized = True
        print("✅ GAM Manager initialized successfully!")
        return self

    def get_agents(self):
        return self.memory_agent, self.research_agent

    def clear_all(self):
        """
        Clear memory store, page store (database), and all retriever indices.
        Keeps the initialized generator/agents intact.
        """
        # Reset memory
        self.memory_store.save(MemoryState())
        
        # Clear database (this is the source of truth now!)
        self.chunk_db.clear_all()
        print("[GAMManager] Database cleared")
        
        # Clear page_store cache
        self.page_store.clear()
        print("[GAMManager] Page store cleared")

        # Clear dense index
        if hasattr(self, "dense_retriever") and self.dense_retriever:
            try:
                self.dense_retriever.clear()
                print("[GAMManager] Dense retriever cleared")
            except Exception as e:
                print(f"[GAMManager] Failed to clear dense retriever: {e}")

        # Clear BM25 index
        if hasattr(self, "bm25_retriever") and self.bm25_retriever:
            try:
                bm25_index_dir = self.bm25_retriever.index_dir
                shutil.rmtree(bm25_index_dir, ignore_errors=True)
                os.makedirs(bm25_index_dir, exist_ok=True)
                print("[GAMManager] BM25 retriever cleared")
            except Exception as e:
                print(f"[GAMManager] Failed to clear BM25 retriever: {e}")

        # Clean index directories
        try:
            if hasattr(self.memory_store, '_dir_path'):
                shutil.rmtree(self.memory_store._dir_path, ignore_errors=True)
                os.makedirs(self.memory_store._dir_path, exist_ok=True)
            
            if hasattr(self, "dense_retriever"):
                index_dir = self.dense_retriever._index_dir()
                shutil.rmtree(index_dir, ignore_errors=True)
                os.makedirs(index_dir, exist_ok=True)
                
        except Exception as e:
            print(f"[GAMManager] Cleanup warning: {e}")

        # Force rebuild of research agent retrievers on next use
        if hasattr(self, 'research_agent'):
            self.research_agent._last_page_count = 0

        print("[GAMManager] ✅ Cleared memory, database, and all retriever indices.")
        return {"status": "cleared"}

gam_manager = GAMManager()