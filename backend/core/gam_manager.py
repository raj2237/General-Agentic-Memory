import os
import shutil
from gam import (
    MemoryAgent, ResearchAgent, OpenAIGenerator, OpenAIGeneratorConfig,
    InMemoryMemoryStore, InMemoryPageStore, MemoryState,
    DenseRetrieverConfig, DenseRetriever
)
from .config import (
    GROQ_API_KEY, MODEL_NAME, EMBEDDING_MODEL, MAX_RESEARCH_ITERS,
    DENSE_INDEX_DIR, MEMORY_STORE_DIR, PAGE_STORE_DIR
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

        gen_config = OpenAIGeneratorConfig(
            model_name=MODEL_NAME,
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=1024
        )
        self.generator = OpenAIGenerator.from_config(gen_config)

        # Ensure storage directories exist
        os.makedirs(DENSE_INDEX_DIR, exist_ok=True)
        os.makedirs(MEMORY_STORE_DIR, exist_ok=True)
        os.makedirs(PAGE_STORE_DIR, exist_ok=True)

        # Persisted stores
        self.page_store = InMemoryPageStore(dir_path=PAGE_STORE_DIR)
        self.memory_store = InMemoryMemoryStore(dir_path=MEMORY_STORE_DIR)

        self.memory_agent = MemoryAgent(
            generator=self.generator,
            memory_store=self.memory_store,
            page_store=self.page_store
        )

        # DENSE RETRIEVER
        dense_config = DenseRetrieverConfig(
            model_name=EMBEDDING_MODEL,
            index_dir=DENSE_INDEX_DIR
        )
        self.dense_retriever = DenseRetriever(dense_config.__dict__)
        
        
        try:
            self.dense_retriever.load()
            print("DenseRetriever: Loaded existing index")
        except:
            print("DenseRetriever: Starting fresh – will build index on first document")

        self.research_agent = ResearchAgent(
            page_store=self.page_store,
            memory_store=self.memory_store,
            retrievers={"vector": self.dense_retriever},  
            generator=self.generator,
            max_iters=MAX_RESEARCH_ITERS
        )

        self.initialized = True
        print("GAM Manager initialized successfully!")
        return self

    def get_agents(self):
        return self.memory_agent, self.research_agent

    def clear_all(self):
        """
        Clear memory store, page store, and dense retriever index.
        Keeps the initialized generator/agents intact.
        """
        # Reset memory and pages
        self.memory_store.save(MemoryState())
        self.page_store.save([])

        # Clear dense index files and reset in-memory handles
        if hasattr(self, "dense_retriever") and self.dense_retriever:
            try:
                self.dense_retriever.clear()
            except Exception as e:
                print(f"[GAMManager] Failed to clear dense retriever: {e}")

        # Also clean persisted storage directories to avoid stale reuse
        try:
            shutil.rmtree(self.page_store._dir_path, ignore_errors=True)
            shutil.rmtree(self.memory_store._dir_path, ignore_errors=True)
            shutil.rmtree(self.dense_retriever._index_dir(), ignore_errors=True)
        except Exception as e:
            print(f"[GAMManager] Cleanup warning: {e}")

        # Recreate storage directories
        os.makedirs(self.page_store._dir_path, exist_ok=True)
        os.makedirs(self.memory_store._dir_path, exist_ok=True)
        os.makedirs(self.dense_retriever._index_dir(), exist_ok=True)

        print("[GAMManager] Cleared memory, pages, and dense index.")
        return {"status": "cleared"}

gam_manager = GAMManager()