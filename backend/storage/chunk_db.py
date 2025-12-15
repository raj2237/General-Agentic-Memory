"""
Chunk Database Module

Fast SQLite-based storage for document chunks to improve upload speed.
Stores all chunks immediately without waiting for embedding generation.
"""
import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class ChunkDB:
    """SQLite database for storing document chunks"""
    
    def __init__(self, db_path: str = "./storage/chunks.db"):
        """Initialize chunk database"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                indexed BOOLEAN DEFAULT 0
            )
        """)
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                total_chunks INTEGER NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                indexed BOOLEAN DEFAULT 0
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexed ON chunks(indexed)")
        
        conn.commit()
        conn.close()
    
    def add_document(self, doc_id: str, filename: str, chunks: List[str], metadata: Optional[Dict] = None) -> int:
        """
        Add a document and its chunks to the database
        Returns: number of chunks added
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Add document record
            cursor.execute("""
                INSERT INTO documents (id, filename, total_chunks, indexed)
                VALUES (?, ?, ?, 0)
            """, (doc_id, filename, len(chunks)))
            
            # Add all chunks in batch
            chunk_data = [
                (doc_id, idx, chunk, json.dumps(metadata) if metadata else None, False)
                for idx, chunk in enumerate(chunks)
            ]
            
            cursor.executemany("""
                INSERT INTO chunks (document_id, chunk_index, content, metadata, indexed)
                VALUES (?, ?, ?, ?, ?)
            """, chunk_data)
            
            conn.commit()
            return len(chunks)
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_unindexed_chunks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get chunks that haven't been indexed yet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT id, document_id, chunk_index, content, metadata
            FROM chunks
            WHERE indexed = 0
            ORDER BY id
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "document_id": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "metadata": json.loads(row[4]) if row[4] else None
            }
            for row in rows
        ]
    
    def mark_chunks_indexed(self, chunk_ids: List[int]):
        """Mark chunks as indexed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ",".join("?" * len(chunk_ids))
        cursor.execute(f"""
            UPDATE chunks
            SET indexed = 1
            WHERE id IN ({placeholders})
        """, chunk_ids)
        
        conn.commit()
        conn.close()
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, chunk_index, content, metadata, indexed
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index
        """, (doc_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]) if row[3] else None,
                "indexed": bool(row[4])
            }
            for row in rows
        ]
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, filename, total_chunks, uploaded_at, indexed
            FROM documents
            ORDER BY uploaded_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "filename": row[1],
                "total_chunks": row[2],
                "uploaded_at": row[3],
                "indexed": bool(row[4])
            }
            for row in rows
        ]
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, filename, total_chunks, uploaded_at, indexed
            FROM documents
            WHERE id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "filename": row[1],
                "total_chunks": row[2],
                "uploaded_at": row[3],
                "indexed": bool(row[4])
            }
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete chunks first
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            # Delete document
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Error deleting document {doc_id}: {e}")
            return False
        finally:
            conn.close()
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def clear_all(self):
        """Clear all data from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM documents")
        
        conn.commit()
        conn.close()
