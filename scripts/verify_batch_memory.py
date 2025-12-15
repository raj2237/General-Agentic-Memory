
import asyncio
import time
from typing import List, Dict, Any

# Mock generator to simulate latency without real API calls
class MockGenerator:
    def __init__(self, latency=0.1):
        self.latency = latency
        
    def generate_single(self, prompt: str, **kwargs) -> Dict[str, Any]:
        time.sleep(self.latency)
        return {"text": f"Abstract for: {prompt[-20:]}...", "json": None, "response": {}}
        
    def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        # Simulate parallel processing where total time is roughly equal to single latency
        # (plus a tiny overhead)
        time.sleep(self.latency + 0.05)
        return [{"text": f"Abstract for: {p[-20:]}...", "json": None, "response": {}} for p in prompts]

# Import MemoryAgent (assuming gam package is in path)
import sys
import os
sys.path.append(os.getcwd())

from gam.agents.memory_agent import MemoryAgent
from gam.schemas import InMemoryMemoryStore, InMemoryPageStore

def test_batch_memorization():
    print("\n=== Testing Batch Memorization ===")
    
    # Setup
    generator = MockGenerator(latency=0.5) # 500ms latency per call
    memory_store = InMemoryMemoryStore()
    page_store = InMemoryPageStore()
    
    agent = MemoryAgent(
        memory_store=memory_store,
        page_store=page_store,
        generator=generator
    )
    
    messages = [f"Message chunk {i}" for i in range(10)]
    
    print(f"Processing {len(messages)} messages with simulated latency of {generator.latency}s per call...")
    
    # Test Sequential (Conceptually, what it was before)
    # We can't easily run the old code, but we know it would be len(messages) * latency
    predicted_sequential_time = len(messages) * generator.latency
    print(f"Predicted sequential time: ~{predicted_sequential_time:.2f}s")
    
    # Test Batch
    start_time = time.time()
    updates = agent.memorize_batch(messages)
    end_time = time.time()
    actual_time = end_time - start_time
    
    print(f"Actual batch time: {actual_time:.2f}s")
    
    # Verification
    if actual_time < predicted_sequential_time * 0.5:
        print("✅ Speedup verified! (Batch time is significantly less than sequential)")
    else:
        print("❌ Speedup NOT observed. (Is generate_batch actually parallel in the mock/real impl?)")
        
    # Check results
    assert len(updates) == len(messages), f"Expected {len(messages)} updates, got {len(updates)}"
    
    saved_pages = page_store.load()
    assert len(saved_pages) == len(messages), f"Expected {len(messages)} stored pages, got {len(saved_pages)}"
    
    print(f"✅ Created {len(updates)} memory updates successfully.")
    print("=== Test Passed ===\n")

if __name__ == "__main__":
    test_batch_memorization()
