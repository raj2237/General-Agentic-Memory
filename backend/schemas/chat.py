from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ResearchStep(BaseModel):
    iteration: int
    thought: str
    retrieved_chunks: List[Dict[str, Any]]
    is_final: bool = False

class FinalAnswer(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]