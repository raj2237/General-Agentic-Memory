#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM Basic Usage Example

This example demonstrates how to use the GAM framework for basic memory
construction and question answering. It shows the full workflow of
memory creation, retrieval, and research.
"""

import os
from dotenv import load_dotenv
load_dotenv()
from gam import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    IndexRetrieverConfig,
    BM25Retriever,
    BM25RetrieverConfig,
    DenseRetriever,
    DenseRetrieverConfig,
)


def basic_memory_example():
    """Basic memory construction example"""
    print("=== Basic Memory Construction Example ===\n")
    
    # 1. Configure and create Generator
    gen_config = OpenAIGeneratorConfig(
        model_name="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",  # 👈 This is the key!
        temperature=0.3,
        max_tokens=2048
    )
    generator = OpenAIGenerator.from_config(gen_config)
    
    # 2. Create storage
    memory_store = InMemoryMemoryStore()
    page_store = InMemoryPageStore()
    
    # 3. Create MemoryAgent
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store
    )
    
    # 4. Prepare text to memorize (simulating long documents)
    documents = [
        """Artificial Intelligence (AI) is a branch of computer science 
        dedicated to creating systems capable of performing tasks that 
        typically require human intelligence. Machine learning is a subset 
        of AI that enables computers to learn without being explicitly programmed.""",
        
        """Deep learning is a subset of machine learning that uses multilayer 
        neural networks to simulate how the human brain works. Natural Language 
        Processing (NLP) is another important branch of AI, focusing on enabling 
        computers to understand, interpret, and generate human language.""",
        
        """Computer vision is another key field of AI, dedicated to enabling 
        computers to 'see' and understand visual information. Reinforcement learning 
        is a machine learning method that learns optimal behavior strategies through 
        interaction with the environment.""",
        
        """Neural networks are the foundation of deep learning, composed of 
        interconnected nodes (neurons). Convolutional Neural Networks (CNNs) 
        are particularly suited for image processing tasks, while Recurrent Neural 
        Networks (RNNs) excel at handling sequential data.""",
        
        """The introduction of the Transformer architecture revolutionized the NLP 
        field and laid the foundation for large language models such as GPT and BERT."""
    ]
    
    # 5. Memorize documents one by one
    print(f"Memorizing {len(documents)} documents...")
    for i, doc in enumerate(documents, 1):
        print(f"  Memorizing document {i}/{len(documents)}...")
        memory_agent.memorize(doc)
    
    # 6. View memory state
    memory_state = memory_store.load()
    print(f"\n✅ Memory successfully constructed:")
    print(f"  - Number of memory abstracts: {len(memory_state.abstracts)}")
        
    return memory_agent, memory_store, page_store




def research_example(memory_store, page_store):
    """Research example based on memory"""
    print("\n=== Research Example Based on Memory ===\n")
    
    # 1. Configure and create Generator
    gen_config = OpenAIGeneratorConfig(
        model_name="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",  
        temperature=0.3,
        max_tokens=2048
    )
    generator = OpenAIGenerator.from_config(gen_config)
    
    # 2. Create multiple retrievers
    retrievers = {}
    index_dir = './tmp'
    
    # Index retriever
    try:
        page_index_dir = os.path.join(index_dir, "page_index")
        if os.path.exists(page_index_dir):
            import shutil
            shutil.rmtree(page_index_dir)
        
        index_config = IndexRetrieverConfig(
            index_dir=page_index_dir
        )
        index_retriever = IndexRetriever(index_config.__dict__)
        index_retriever.build(page_store)
        retrievers["page_index"] = index_retriever
        print("✅ Index retriever created successfully")
    except Exception as e:
        print(f"[WARN] Failed to create index retriever: {e}")
    
    # BM25 retriever
    try:
        bm25_index_dir = os.path.join(index_dir, "bm25_index")
        if os.path.exists(bm25_index_dir):
            import shutil
            shutil.rmtree(bm25_index_dir)
        
        bm25_config = BM25RetrieverConfig(
            index_dir=bm25_index_dir,
            threads=1
        )
        bm25_retriever = BM25Retriever(bm25_config.__dict__)
        bm25_retriever.build(page_store)
        retrievers["keyword"] = bm25_retriever
        print("✅ BM25 retriever created successfully")
    except Exception as e:
        print(f"[WARN] Failed to create BM25 retriever: {e}")
    
    # Dense retriever
    try:
        dense_index_dir = os.path.join(index_dir, "dense_index")
        if os.path.exists(dense_index_dir):
            import shutil
            shutil.rmtree(dense_index_dir)
        
        dense_config = DenseRetrieverConfig(
            index_dir=dense_index_dir,
            model_name="BAAI/bge-m3"
        )
        dense_retriever = DenseRetriever(dense_config.__dict__)
        dense_retriever.build(page_store)
        retrievers["vector"] = dense_retriever
        print("✅ Dense retriever created successfully")
    except Exception as e:
        print(f"[WARN] Failed to create dense retriever: {e}")
    
    # 3. Create ResearchAgent
    research_agent_kwargs = {
        "page_store": page_store,
        "memory_store": memory_store,
        "retrievers": retrievers,
        "generator": generator,
        "max_iters": 5
    }
    research_agent = ResearchAgent(**research_agent_kwargs)
    
    # 4. Research question
    question = "What are the key differences between machine learning and deep learning?"
    print(f"\nResearch Question: {question}\n")
    
    research_result = research_agent.research(question)
    research_summary = research_result.integrated_memory
    
    # 5. Show results
    print(f"✅ Research Completed:")
    print(f"  - Iterations: {len(research_result.raw_memory.get('iterations', []))}")
    print(f"\nResearch Summary:")
    print(f"  {research_summary}")
    
    return research_result


def main():
    """Main function"""
    print("=" * 60)
    print("GAM Framework Quick Start Example")
    print("=" * 60)
    print()
    
    # Check API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set the environment variable OPENAI_API_KEY")
        print("   export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # 1. Run basic memory construction example
        memory_agent, memory_store, page_store = basic_memory_example()
        
        # 2. Run research example based on memory
        research_result = research_example(memory_store, page_store)
        
        print("\n" + "=" * 60)
        print("✅ Example Execution Completed!")
        print("=" * 60)
        print("\nYou can develop your own application based on these examples!")
        print("\nTips:")
        print("  - Modify the document content to test different scenarios")
        print("  - Try different questions to test research capability")
        print("  - Check the eval/ directory for more evaluation examples")
        
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
        print("\nPlease check:")
        print("  1. Whether your network connection is working")
        print("  2. Whether your API Key is correct")
        print("  3. Whether required dependencies are installed: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
