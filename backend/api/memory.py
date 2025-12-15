from fastapi import APIRouter
from core.gam_manager import gam_manager
from api.documents import file_history, chunk_db

router = APIRouter(prefix="/api")

@router.get("/memories")
def get_memories():
    _, research_agent = gam_manager.get_agents()
    memory_state = research_agent.memory_store.load()
    pages = research_agent.page_store.load()
    return {
        "abstracts": memory_state.abstracts if hasattr(memory_state, 'abstracts') else [],
        "pages": [p.model_dump() for p in pages],
        "stats": {"total_memories": len(memory_state.abstracts) if hasattr(memory_state, 'abstracts') else 0}
    }

@router.get("/graph")
def get_graph():
    """
    Get enhanced knowledge graph data for visualization.
    Shows documents, chunks, and extracted entities with semantic relationships.
    """
    try:
        from storage.chunk_db import ChunkDB
        from utils.graph_builder import build_enhanced_graph
        
        chunk_db = ChunkDB()
        graph_data = build_enhanced_graph(chunk_db)
        
        return graph_data
    except Exception as e:
        print(f"[ERROR] Failed to get graph data: {e}")
        import traceback
        traceback.print_exc()
        return {"nodes": [], "edges": [], "stats": {"total_documents": 0, "total_chunks": 0, "total_entities": 0}}


@router.post("/memory/clear")
def clear_memory():
    result = gam_manager.clear_all()
    # Also clear file history and chunk database
    file_history.clear()
    chunk_db.clear_all()
    return {"status": "success", **result}