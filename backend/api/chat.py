# api/chat.py (FINAL – 100% WORKING WITH LATEST GAM)

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import asyncio
from core.gam_manager import gam_manager

router = APIRouter(prefix="/api")

@router.post("/chat")
async def chat_endpoint(request: dict):
    user_query = request.get("message", "").strip()
    if not user_query:
        return JSONResponse({"error": "Empty message", "detail": "Message cannot be empty"}, status_code=400)

    if not gam_manager.initialized:
        return JSONResponse({
            "error": "System not initialized",
            "detail": "Please upload a document first"
        }, status_code=503)

    _, research_agent = gam_manager.get_agents()

    # Debug: Check page store and retrievers
    page_count = len(gam_manager.page_store.load())
    retriever_keys = list(research_agent.retrievers.keys()) if research_agent.retrievers else []
    print(f"[DEBUG] Chat request: '{user_query}'")
    print(f"[DEBUG] Page store has {page_count} pages")
    print(f"[DEBUG] Available retrievers: {retriever_keys}")

    if page_count == 0:
        return JSONResponse({
            "error": "No documents",
            "detail": "Upload at least one document before chatting."
        }, status_code=400)

    try:
        
        result = await asyncio.to_thread(research_agent.research, request=user_query)

        # ResearchOutput has: integrated_memory (str) and raw_memory (dict)
        if hasattr(result, "model_dump"):
            data = result.model_dump()
        elif hasattr(result, "dict"):
            data = result.dict()
        elif isinstance(result, dict):
            data = result
        else:
            # Fallback
            data = {"integrated_memory": str(result), "raw_memory": {"iterations": [], "temp_memory": {}}}

        # Extract the final answer from integrated_memory
        final_answer = data.get("integrated_memory", "")
        raw_memory = data.get("raw_memory", {})
        iterations = raw_memory.get("iterations", [])
        temp_memory = raw_memory.get("temp_memory", {})

        # Extract thinking steps from iterations
        thinking_steps = []
        retrieved_chunks_count = 0
        
        for i, iteration in enumerate(iterations):
            step_info = {
                "iteration": i + 1,
                "thought": "Processing query...",
                "retrieved": 0
            }
            
            # Extract plan and decision info
            plan = iteration.get("plan", {})
            decision = iteration.get("decision", {})
            temp_mem = iteration.get("temp_memory", {})
            
            # Get info needs from plan
            info_needs = plan.get("info_needs", [])
            if info_needs:
                step_info["thought"] = f"Searching for: {', '.join(info_needs[:2])}"
            
            # Count retrieved chunks from temp_memory sources
            sources = temp_mem.get("sources", [])
            if sources:
                step_info["retrieved"] = len(sources) if isinstance(sources, list) else 0
                retrieved_chunks_count = max(retrieved_chunks_count, step_info["retrieved"])
            
            # Check if enough information was found
            if decision.get("enough", False):
                step_info["thought"] += " (Information sufficient)"
            
            thinking_steps.append(step_info)

        # If no thinking steps but we have an answer, create a default step
        if not thinking_steps and final_answer:
            thinking_steps = [{
                "iteration": 1,
                "thought": "Retrieved and processed information",
                "retrieved": retrieved_chunks_count
            }]

        # Get final retrieved chunks count from last temp_memory
        if temp_memory and "sources" in temp_memory:
            sources_list = temp_memory["sources"]
            if isinstance(sources_list, list):
                retrieved_chunks_count = len(sources_list)

        return JSONResponse({
            "answer": final_answer.strip() if final_answer else "No relevant information found in the uploaded documents.",
            "thinking_steps": thinking_steps,
            "retrieved_chunks_count": retrieved_chunks_count,
            "status": "success"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "error": "Research failed",
            "detail": str(e)
        }, status_code=500)