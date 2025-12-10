from fastapi import APIRouter
from core.gam_manager import gam_manager

router = APIRouter()

@router.get("/memories")
def get_memories():
    _, research_agent = gam_manager.get_agents()
    return {
        "entities": research_agent.memory_store.get_all_entities(),
        "relations": research_agent.memory_store.get_all_relations(),
        "pages": [p.dict() for p in research_agent.page_store.get_all_pages()],
        "stats": {"total_memories": len(research_agent.memory_store.get_all_memories())}
    }

@router.get("/graph")
def get_graph():
    # Simple format for vis.js or react-flow
    nodes = []
    edges = []
    # ... populate from memory_store
    return {"nodes": nodes, "edges": edges}


@router.post("/memory/clear")
def clear_memory():
    result = gam_manager.clear_all()
    return {"status": "success", **result}