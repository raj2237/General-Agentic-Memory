from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from core.gam_manager import gam_manager
from utils.file_extractor import extract_text_from_file
import asyncio
from fastapi import File

router = APIRouter(prefix="/api")


def _chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    """
    Split text into overlapping chunks so large docs don't blow LLM/context limits.
    """
    if chunk_size <= overlap:
        overlap = 0
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file selected")

    content = await file.read()

    try:
        text = extract_text_from_file(content, file.filename)

        if len(text.strip()) < 50:
            raise ValueError("Document is empty or unreadable")

        memory_agent, _ = gam_manager.get_agents()

        # Chunk large documents to keep within prompt/embedding limits
        chunks = _chunk_text(text)
        if not chunks:
            raise ValueError("No readable chunks found after processing the document")

        for idx, chunk in enumerate(chunks):
            # Process sequentially to avoid overwhelming the generator
            await asyncio.to_thread(memory_agent.memorize, chunk)

        dense_retriever = gam_manager.dense_retriever
        # Use incremental update when possible (falls back to build on first run)
        await asyncio.to_thread(dense_retriever.update, gam_manager.page_store)

        preview = text[:400] + "..." if len(text) > 400 else text
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "characters": len(text),
            "chunks": len(chunks),
            "preview": preview,
            "index_built": True
        })
    except Exception as e:
        raise HTTPException(500, detail=f"Upload failed: {str(e)}")