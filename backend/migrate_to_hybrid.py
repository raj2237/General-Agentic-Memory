"""
Migration script for upgrading to hybrid retrieval system.

This script helps existing installations migrate to the new system with:
1. Chunk database for fast uploads
2. Hybrid retrieval (Dense + BM25)

Usage:
    python migrate_to_hybrid.py [--clear-indices] [--rebuild-all]
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.gam_manager import gam_manager
from storage.chunk_db import ChunkDB


async def check_system():
    """Check current system status"""
    print("\n" + "="*60)
    print("Checking Current System Status")
    print("="*60)
    
    # Check if GAM is initialized
    try:
        await gam_manager.init()
        print("✅ GAM Manager initialized")
    except Exception as e:
        print(f"❌ GAM Manager initialization failed: {e}")
        return False
    
    # Check retrievers
    has_dense = hasattr(gam_manager, 'dense_retriever') and gam_manager.dense_retriever is not None
    has_bm25 = hasattr(gam_manager, 'bm25_retriever') and gam_manager.bm25_retriever is not None
    
    print(f"{'✅' if has_dense else '❌'} Dense retriever: {'Available' if has_dense else 'Not found'}")
    print(f"{'✅' if has_bm25 else '❌'} BM25 retriever: {'Available' if has_bm25 else 'Not found'}")
    
    # Check chunk database
    try:
        chunk_db = ChunkDB()
        docs = chunk_db.get_all_documents()
        print(f"✅ Chunk database: {len(docs)} documents stored")
    except Exception as e:
        print(f"⚠️  Chunk database: New installation (no existing data)")
    
    # Check page store
    try:
        pages = gam_manager.page_store.load()
        print(f"✅ Page store: {len(pages)} pages")
    except Exception as e:
        print(f"⚠️  Page store: {e}")
    
    return has_dense and has_bm25


async def clear_indices():
    """Clear all indices for fresh rebuild"""
    print("\n" + "="*60)
    print("Clearing Indices")
    print("="*60)
    
    try:
        result = gam_manager.clear_all()
        print("✅ All indices cleared")
        print(f"   {result}")
        
        # Also clear chunk database
        chunk_db = ChunkDB()
        chunk_db.clear_all()
        print("✅ Chunk database cleared")
        
        return True
    except Exception as e:
        print(f"❌ Failed to clear indices: {e}")
        return False


async def rebuild_indices():
    """Rebuild all indices from page store"""
    print("\n" + "="*60)
    print("Rebuilding Indices")
    print("="*60)
    
    try:
        pages = gam_manager.page_store.load()
        if not pages:
            print("⚠️  No pages to index")
            return True
        
        print(f"Found {len(pages)} pages to index")
        
        # Rebuild dense index
        print("Building dense index...")
        gam_manager.dense_retriever.build(gam_manager.page_store)
        print("✅ Dense index built")
        
        # Rebuild BM25 index
        print("Building BM25 index...")
        gam_manager.bm25_retriever.build(gam_manager.page_store)
        print("✅ BM25 index built")
        
        return True
    except Exception as e:
        print(f"❌ Failed to rebuild indices: {e}")
        import traceback
        traceback.print_exc()
        return False


async def verify_system():
    """Verify system is working correctly"""
    print("\n" + "="*60)
    print("Verifying System")
    print("="*60)
    
    try:
        # Check research agent
        research_agent = gam_manager.research_agent
        if not research_agent:
            print("❌ Research agent not found")
            return False
        
        # Check retrievers
        retrievers = research_agent.retrievers
        print(f"✅ Research agent has {len(retrievers)} retrievers:")
        for name, retriever in retrievers.items():
            print(f"   - {name}: {type(retriever).__name__}")
        
        # Test search (if pages exist)
        pages = gam_manager.page_store.load()
        if pages:
            print(f"\nTesting search with {len(pages)} pages...")
            test_query = "test query"
            
            # Test vector search
            try:
                vector_results = research_agent._search_by_vector([test_query], top_k=3)
                print(f"✅ Vector search: {len(vector_results[0]) if vector_results else 0} results")
            except Exception as e:
                print(f"⚠️  Vector search: {e}")
            
            # Test keyword search
            try:
                keyword_results = research_agent._search_by_keyword([test_query], top_k=3)
                print(f"✅ Keyword search: {len(keyword_results[0]) if keyword_results else 0} results")
            except Exception as e:
                print(f"⚠️  Keyword search: {e}")
        
        print("\n✅ System verification complete!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Migrate to hybrid retrieval system"
    )
    parser.add_argument(
        "--clear-indices",
        action="store_true",
        help="Clear all existing indices before migration"
    )
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Rebuild all indices from scratch"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify system status without making changes"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("GAM Hybrid Retrieval Migration Tool")
    print("="*60)
    
    # Check system
    system_ok = await check_system()
    
    if args.verify_only:
        if system_ok:
            await verify_system()
        return
    
    # Clear indices if requested
    if args.clear_indices:
        if not await clear_indices():
            print("\n❌ Migration failed at clearing step")
            return
    
    # Rebuild indices if requested
    if args.rebuild_all:
        if not await rebuild_indices():
            print("\n❌ Migration failed at rebuild step")
            return
    
    # Verify system
    if not await verify_system():
        print("\n⚠️  System verification found issues")
        return
    
    print("\n" + "="*60)
    print("✅ Migration Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start the server: uvicorn main:app --reload")
    print("2. Upload documents to test the new system")
    print("3. Query and verify improved accuracy")
    print("\nFor more information, see IMPROVEMENTS.md")


if __name__ == "__main__":
    asyncio.run(main())
