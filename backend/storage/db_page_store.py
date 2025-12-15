"""
Database-backed Page Store
Replaces file-based InMemoryPageStore with direct database access
"""
from typing import List, Optional
from gam.schemas import Page
from storage.chunk_db import ChunkDB


class DatabasePageStore:
    """
    Page store that reads/writes directly to ChunkDB.
    No JSON files involved - pure database storage.
    """
    
    def __init__(self, chunk_db: ChunkDB):
        """Initialize with a ChunkDB instance"""
        self.chunk_db = chunk_db
        self._cache: List[Page] = []
        self._cache_loaded = False
    
    def load(self) -> List[Page]:
        """Load all pages from database"""
        all_docs = self.chunk_db.get_all_documents()
        pages = []
        
        for doc in all_docs:
            doc_chunks = self.chunk_db.get_document_chunks(doc["id"])
            for chunk_data in doc_chunks:
                page = Page(
                    header=f"{doc['filename']} - Chunk {chunk_data['chunk_index'] + 1}",
                    content=chunk_data["content"]
                )
                pages.append(page)
        
        self._cache = pages
        self._cache_loaded = True
        return pages
    
    def save(self, pages: List[Page]) -> None:
        """
        Save is a no-op since database is already updated.
        Pages are saved via chunk_db.add_document() during upload.
        """
        # Update cache
        self._cache = pages
        self._cache_loaded = True
        # Database is already updated, nothing to do
    
    def add(self, page: Page) -> None:
        """
        Add a page to the cache.
        Note: This doesn't persist to DB - use chunk_db.add_document() for that.
        This is only for in-memory operations.
        """
        if not self._cache_loaded:
            self.load()
        self._cache.append(page)
    
    def get(self, index: int) -> Optional[Page]:
        """Get a page by index"""
        if not self._cache_loaded:
            self.load()
        
        if 0 <= index < len(self._cache):
            return self._cache[index]
        return None
    
    def __len__(self) -> int:
        """Return the number of pages"""
        if not self._cache_loaded:
            self.load()
        return len(self._cache)
    
    def get_all_pages(self) -> List[Page]:
        """Get all pages (alias for load)"""
        return self.load()
    
    def clear(self) -> None:
        """Clear all pages from database"""
        self.chunk_db.clear_all()
        self._cache = []
        self._cache_loaded = False
