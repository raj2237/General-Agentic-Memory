# Implementation Plan: GAM System Improvements

## Issue 1: Retriever Naming Confusion
**Problem**: Using "vector" and "keyword" names instead of actual retriever names (DenseRetriever, BM25)
**Solution**: 
- Update GAMManager to use "dense" and "bm25" as retriever keys
- Update ResearchAgent to use "dense" and "bm25" in search methods
- Update all references throughout the codebase

## Issue 2: Document Index Separation
**Problem**: All documents share the same index, causing mixing of chunks
**Solution**:
- Modify ChunkDB to track document-specific metadata
- Create per-document index structure in retrievers
- Update retriever build/update methods to maintain document boundaries
- Add document filtering in search methods

## Issue 3: Poor Knowledge Graph Quality
**Problem**: Graph nodes show poor content (truncated text, no semantic meaning)
**Solution**:
- Extract key entities and concepts from chunks using LLM
- Create semantic summaries for each chunk
- Build relationships based on semantic similarity
- Improve node labels with meaningful titles
- Add entity extraction for better graph structure

## Implementation Steps:

### Step 1: Fix Retriever Naming (15 min)
- [x] Update gam_manager.py retriever keys
- [ ] Update research_agent.py search method calls
- [ ] Test retrieval still works

### Step 2: Per-Document Indexing (30 min)
- [ ] Extend ChunkDB schema with document metadata
- [ ] Create DocumentRetriever wrapper class
- [ ] Modify DenseRetriever to support document filtering
- [ ] Modify BM25Retriever to support document filtering
- [ ] Update upload flow to create per-document indices

### Step 3: Enhanced Knowledge Graph (45 min)
- [ ] Create entity extraction module
- [ ] Add semantic summarization for chunks
- [ ] Build graph with entity nodes + chunk nodes
- [ ] Add relationship detection
- [ ] Update /api/graph endpoint
- [ ] Improve frontend visualization

## Priority: HIGH
## Estimated Time: 90 minutes
## Dependencies: None
