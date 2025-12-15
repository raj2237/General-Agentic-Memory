from fastapi import APIRouter, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from core.gam_manager import gam_manager
from utils.file_extractor import extract_text_from_file
from storage.chunk_db import ChunkDB
import asyncio
from fastapi import File
from datetime import datetime
import uuid

router = APIRouter(prefix="/api")

# Initialize chunk database
chunk_db = ChunkDB()

# In-memory file history (could be persisted to disk/db later)
file_history = []


def _chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    """
    Ultra-fast text chunking using list comprehension.
    """
    if chunk_size <= overlap:
        overlap = 0
    
    step = chunk_size - overlap
    text_len = len(text)
    
    # Single-pass chunking with list comprehension (FAST!)
    chunks = [
        text[i:i + chunk_size].strip()
        for i in range(0, text_len, step)
        if text[i:i + chunk_size].strip()
    ]
    
    return chunks


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    import time
    start_time = time.time()
    
    if not file.filename:
        raise HTTPException(400, "No file selected")

    content = await file.read()
    print(f"[UPLOAD] ⏱️ File read: {time.time() - start_time:.2f}s")

    try:
        # Extract text
        extract_start = time.time()
        text = extract_text_from_file(content, file.filename)
        print(f"[UPLOAD] ⏱️ Text extraction: {time.time() - extract_start:.2f}s")

        if len(text.strip()) < 50:
            raise ValueError("Document is empty or unreadable")

        # Chunk the text
        chunk_start = time.time()
        chunks = _chunk_text(text)
        print(f"[UPLOAD] ⏱️ Chunking ({len(chunks)} chunks): {time.time() - chunk_start:.2f}s")
        
        if not chunks:
            raise ValueError("No readable chunks found after processing the document")

        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Store all chunks in database - INSTANT!
        db_start = time.time()
        chunk_db.add_document(doc_id, file.filename, chunks)
        print(f"[UPLOAD] ⏱️ Database save: {time.time() - db_start:.2f}s")
        print(f"[UPLOAD] ✅ Total upload time: {time.time() - start_time:.2f}s")

        # BACKGROUND: Update retrievers only (FAST - no memorization!)
        if background_tasks:
            background_tasks.add_task(_update_retrievers_background, doc_id, file.filename)
        else:
            asyncio.create_task(_update_retrievers_background(doc_id, file.filename))
        
        # Skip heavy memorization - it's too slow and not needed for basic chat!
        # Memory agent can build abstracts on-demand during chat if needed

        # Add to file history (preliminary status)
        file_entry = {
            "id": len(file_history) + 1,
            "doc_id": doc_id,
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "size": len(text),
            "chunks": len(chunks),
            "processed_chunks": 0, # Will be updated in background
            "status": "processing"
        }
        file_history.append(file_entry)

        preview = text[:400] + "..." if len(text) > 400 else text
        return JSONResponse({
            "status": "accepted",
            "doc_id": doc_id,
            "filename": file.filename,
            "characters": len(text),
            "chunks": len(chunks),
            "message": "File uploaded and queued for processing",
            "preview": preview
        }, status_code=202)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=f"Upload failed: {str(e)}")




async def _update_retrievers_background(doc_id: str, filename: str):
    """
    Fast retriever indexing in background - BOTH Dense + BM25 for better accuracy!
    """
    import time
    start = time.time()
    print(f"[INDEXER] 🚀 Starting hybrid indexing for {filename}...")
    
    # Update file status to "indexing"
    for f in file_history:
        if f["doc_id"] == doc_id:
            f["status"] = "indexing"
            break
    
    try:
        # Reload page_store from database
        pages = gam_manager.page_store.load()
        print(f"[INDEXER] Loaded {len(pages)} pages in {time.time() - start:.2f}s")
        
        # Update dense retriever (vector search)
        dense_start = time.time()
        try:
            await asyncio.to_thread(gam_manager.dense_retriever.update, gam_manager.page_store)
            print(f"[INDEXER] ✅ Dense index updated in {time.time() - dense_start:.2f}s")
        except Exception as e:
            print(f"[INDEXER] Dense update failed, building fresh: {str(e)[:100]}")
            try:
                await asyncio.to_thread(gam_manager.dense_retriever.build, gam_manager.page_store)
                print(f"[INDEXER] ✅ Dense index built in {time.time() - dense_start:.2f}s")
            except Exception as build_e:
                print(f"[INDEXER] ❌ Dense indexing failed: {str(build_e)[:100]}")
                # Mark as failed and return
                for f in file_history:
                    if f["doc_id"] == doc_id:
                        f["status"] = "failed"
                        f["error"] = "Dense indexing failed"
                        break
                return
        
        # Update BM25 retriever (keyword search) - ENABLED for better accuracy!
        bm25_start = time.time()
        try:
            await asyncio.to_thread(gam_manager.bm25_retriever.update, gam_manager.page_store)
            print(f"[INDEXER] ✅ BM25 index updated in {time.time() - bm25_start:.2f}s")
        except Exception as e:
            print(f"[INDEXER] BM25 update failed, building fresh: {str(e)[:100]}")
            try:
                await asyncio.to_thread(gam_manager.bm25_retriever.build, gam_manager.page_store)
                print(f"[INDEXER] ✅ BM25 index built in {time.time() - bm25_start:.2f}s")
            except Exception as build_e:
                print(f"[INDEXER] ⚠️ BM25 indexing failed (continuing with dense only): {str(build_e)[:100]}")
        
        # Mark as completed
        for f in file_history:
            if f["doc_id"] == doc_id:
                f["status"] = "completed"
                f["processed_chunks"] = len(pages)
                break
        
        print(f"[INDEXER] 🎉 Hybrid indexing complete in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"[INDEXER] ❌ Indexing failed: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        # Mark as failed
        for f in file_history:
            if f["doc_id"] == doc_id:
                f["status"] = "failed"
                f["error"] = str(e)[:200]
                break


async def _process_document_background(doc_id: str, chunks: list[str], filename: str):
    """
    Process chunks in background:
    1. Memorize chunks (MemoryAgent) - chunks already in page_store from upload
    2. Build/Update Retriever Indices
    """
    print(f"[BACKGROUND] Starting processing for {filename} ({doc_id})")
    try:
        memory_agent, _ = gam_manager.get_agents()
        
        # Chunks are already in page_store from upload route, so skip that step
        
        # Process chunks with batch summarization
        # Note: memorize_batch runs in parallel threads internally
        print(f"[BACKGROUND] Memorizing {len(chunks)} chunks in parallel...")
        await asyncio.to_thread(memory_agent.memorize_batch, chunks)
        print(f"[BACKGROUND] ✅ Memorization complete for {filename}")

        # Update retrievers
        print("[BACKGROUND] Updating retriever indices...")
        dense_retriever = gam_manager.dense_retriever
        bm25_retriever = gam_manager.bm25_retriever
        
        try:
            await asyncio.to_thread(dense_retriever.update, gam_manager.page_store)
            print("[BACKGROUND] ✅ Dense retriever updated")
        except Exception as e:
            print(f"[BACKGROUND] ❌ Dense retriever update failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            await asyncio.to_thread(bm25_retriever.update, gam_manager.page_store)
            print("[BACKGROUND] ✅ BM25 retriever updated")
        except Exception as e:
            print(f"[BACKGROUND] ❌ BM25 retriever update failed: {e}")

        # Update status in file history (simple in-memory update)
        for f in file_history:
            if f["doc_id"] == doc_id:
                f["processed_chunks"] = len(chunks)
                f["status"] = "completed"
                break
                
        print(f"[BACKGROUND] 🎉 Processing finished for {filename}")

    except Exception as e:
        print(f"[BACKGROUND] ❌ Processing failed for {filename}: {e}")
        for f in file_history:
            if f["doc_id"] == doc_id:
                f["status"] = "failed"
                break


@router.get("/files")
async def get_file_history():
    """Get the list of uploaded files"""
    return JSONResponse({
        "status": "success",
        "files": file_history
    })


@router.get("/indexing-status/{doc_id}")
async def get_indexing_status(doc_id: str):
    """Get the indexing status of a specific document"""
    for f in file_history:
        if f["doc_id"] == doc_id:
            return JSONResponse({
                "status": f.get("status", "unknown"),
                "processed_chunks": f.get("processed_chunks", 0),
                "total_chunks": f.get("chunks", 0),
                "error": f.get("error", None)
            })
    
    return JSONResponse({"error": "Document not found"}, status_code=404)
