import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.gam_manager import gam_manager
from gam.schemas import Page

async def verify_retrieval():
    print("Initializing GAM Manager...")
    await gam_manager.init()
    
    # Create some dummy pages
    pages = [
        Page(id="1", content="The capital of France is Paris.", header="Geography"),
        Page(id="2", content="Machine learning is a subset of artificial intelligence.", header="AI"),
        Page(id="3", content="Python is a popular programming language.", header="Programming"),
        Page(id="4", content=" ", header="Empty Page"), # Test empty content
    ]
    
    print(f"Adding {len(pages)} pages to store...")
    gam_manager.page_store.save(pages)
    
    # Explicitly build indices
    print("Building Dense Index...")
    try:
        gam_manager.dense_retriever.build(gam_manager.page_store)
        print("✅ Dense Index Built")
    except Exception as e:
        print(f"❌ Dense Index Build Failed: {e}")
        import traceback
        traceback.print_exc()

    print("Building BM25 Index...")
    try:
        gam_manager.bm25_retriever.build(gam_manager.page_store)
        print("✅ BM25 Index Built")
    except Exception as e:
        print(f"❌ BM25 Index Build Failed: {e}")

    # Test Search
    queries = ["Paris calls", "AI subset", "Python code"]
    
    print("\nTesting Dense Retrieval:")
    results = gam_manager.dense_retriever.search(queries, top_k=1)
    for q, res in zip(queries, results):
        print(f"Query: '{q}' -> Top Hit: {res[0].snippet if res else 'None'}")

    print("\nTesting BM25 Retrieval:")
    results = gam_manager.bm25_retriever.search(queries, top_k=1)
    for q, res in zip(queries, results):
        print(f"Query: '{q}' -> Top Hit: {res[0].snippet if res else 'None'}")

if __name__ == "__main__":
    asyncio.run(verify_retrieval())
