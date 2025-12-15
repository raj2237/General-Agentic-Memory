"""ResearchAgent for GAM framework - handles research tasks and information retrieval."""


from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json

from gam.prompts import Planning_PROMPT, Integrate_PROMPT, InfoCheck_PROMPT, GenerateRequests_PROMPT
from gam.schemas import (
    MemoryState, SearchPlan, Hit, Result, 
    ReflectionDecision, ResearchOutput, MemoryStore, PageStore, Retriever, 
    ToolRegistry, InMemoryMemoryStore,
    PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA
)
from gam.generator import AbsGenerator

class ResearchAgent:
    """
    Public API:
      - research(request) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search(plan) -> SearchResults  (calls keyword/vector/page_id + tools)
      - _integrate(search_results, temp_memory) -> TempMemory
      - _reflection(request, memory_state, temp_memory) -> ReflectionDecision

    Note: Uses MemoryStore to dynamically load current memory state.
    This allows ResearchAgent to access the latest memory updates from MemoryAgent.
    """

    def __init__(
        self,
        page_store: PageStore,
        memory_store: MemoryStore | None = None,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        generator: AbsGenerator | None = None,
        max_iters: int = 3,
        dir_path: Optional[str] = None,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters
        
        default_system_prompts = {"planning": "", "integration": "", "reflection": ""}
        self.system_prompts = {**default_system_prompts, **(system_prompts or {})}

        for name, r in self.retrievers.items():
            try:
                r.build(self.page_store)
                print(f"Successfully built {name} retriever")
            except Exception as e:
                print(f"Failed to build {name} retriever: {e}")

    def research(self, request: str) -> ResearchOutput:
        self._update_retrievers()
        
        temp = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request

        for step in range(self.max_iters):
            # Load current memory state dynamically
            memory_state = self.memory_store.load()
            plan = self._planning(next_request, memory_state)

            temp = self._search(plan, temp, request)

            decision = self._reflection(request, temp)

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
            })

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request


        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
        }
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    def _update_retrievers(self):
        """Update retriever indices if page count changed."""
        current_page_count = len(self.page_store.load())
        
        if hasattr(self, '_last_page_count') and current_page_count != self._last_page_count:
            print(f"Page count changed ({self._last_page_count} -> {current_page_count}), updating indices...")
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    print(f"✅ Updated {name} retriever")
                except Exception as e:
                    print(f"❌ Failed to update {name}: {e}")
        
        self._last_page_count = current_page_count

    def _planning(
        self, 
        request: str, 
        memory_state: MemoryState,
        planning_prompt: Optional[str] = None
    ) -> SearchPlan:
        """Generate search plan with info needs, tools, and queries."""

        if not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        
        system_prompt = self.system_prompts.get("planning")
        template_prompt = Planning_PROMPT.format(request=request, memory=memory_context)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt


        try:
            response = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),

                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", [])
            )
        except Exception as e:
            print(f"Error in planning: {e}")
            return SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_index=[]
            )
    

    def _search(
        self, 
        plan: SearchPlan, 
        result: Result, 
        question: str,
        searching_prompt: Optional[str] = None
    ) -> Result:
        """Hybrid search using Dense + BM25 retrievers with RRF fusion."""

        # Fallbacks so we always search even if planning returned an empty plan
        if not plan.tools:
            plan.tools = ["dense", "bm25"]  # Use both by default
        if "dense" in plan.tools and not plan.vector_queries:
            plan.vector_queries = [question]
        if "bm25" in plan.tools and not plan.keyword_collection:
            plan.keyword_collection = [question]
        
        # Collect hits from each retriever separately for score fusion
        keyword_hits: List[Hit] = []
        vector_hits: List[Hit] = []
        page_index_hits: List[Hit] = []

        # Execute each planned tool and collect hits
        for tool in plan.tools:
            if tool == "bm25" or tool == "keyword":  # Support both names for compatibility
                if plan.keyword_collection:
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_bm25([combined_keywords], top_k=10)
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            keyword_hits.extend(result_list)
                    else:
                        keyword_hits.extend(keyword_results)
                    
            elif tool == "dense" or tool == "vector":  # Support both names for compatibility
                if plan.vector_queries:
                    vector_results = self._search_by_dense(plan.vector_queries, top_k=10)
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            vector_hits.extend(result_list)
                    else:
                        vector_hits.extend(vector_results)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            page_index_hits.extend(result_list)
                    else:
                        page_index_hits.extend(page_results)

        # Apply Reciprocal Rank Fusion (RRF) for score combination
        fused_hits = self._reciprocal_rank_fusion(
            keyword_hits=keyword_hits,
            vector_hits=vector_hits,
            page_index_hits=page_index_hits,
            k=60  # RRF constant
        )

        # Fallback if no hits found
        if not fused_hits:
            fallback_hits = self._search_by_vector([question], top_k=3)
            if fallback_hits and isinstance(fallback_hits[0], list):
                fused_hits = fallback_hits[0]
            if not fused_hits:
                return result
        
        # Take top 8 hits for integration (increased from 5 for better context)
        top_hits = fused_hits[:8]
        
        # Integrate with LLM
        return self._integrate(top_hits, result, question)

    def _reciprocal_rank_fusion(
        self,
        keyword_hits: List[Hit],
        vector_hits: List[Hit],
        page_index_hits: List[Hit],
        k: int = 60
    ) -> List[Hit]:
        """Combine hits from multiple retrievers using Reciprocal Rank Fusion (RRF)."""
        # Build rank maps for each retriever
        page_scores: Dict[str, float] = {}  # page_id -> fused score
        page_hit_map: Dict[str, Hit] = {}   # page_id -> Hit object
        
        # Process keyword hits
        for rank, hit in enumerate(keyword_hits):
            if hit.page_id:
                rrf_score = 1.0 / (k + rank + 1)
                page_scores[hit.page_id] = page_scores.get(hit.page_id, 0) + rrf_score
                if hit.page_id not in page_hit_map:
                    page_hit_map[hit.page_id] = hit
        
        # Process vector hits
        for rank, hit in enumerate(vector_hits):
            if hit.page_id:
                rrf_score = 1.0 / (k + rank + 1)
                page_scores[hit.page_id] = page_scores.get(hit.page_id, 0) + rrf_score
                if hit.page_id not in page_hit_map:
                    page_hit_map[hit.page_id] = hit
        
        # Process page index hits (usually exact matches, give higher weight)
        for rank, hit in enumerate(page_index_hits):
            if hit.page_id:
                rrf_score = 2.0 / (k + rank + 1)  # 2x weight for exact matches
                page_scores[hit.page_id] = page_scores.get(hit.page_id, 0) + rrf_score
                if hit.page_id not in page_hit_map:
                    page_hit_map[hit.page_id] = hit
        
        # Sort by fused score
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final hit list with updated scores
        fused_hits: List[Hit] = []
        for rank, (page_id, fused_score) in enumerate(sorted_pages):
            hit = page_hit_map[page_id]
            updated_meta = hit.meta.copy() if hit.meta else {}
            updated_meta["rank"] = rank
            updated_meta["score"] = fused_score
            updated_meta["fusion"] = "rrf"
            
            fused_hits.append(
                Hit(
                    page_id=hit.page_id,
                    snippet=hit.snippet,
                    source="hybrid",  # Mark as hybrid retrieval
                    meta=updated_meta
                )
            )
        
        return fused_hits

    def _search_no_integrate(self, plan: SearchPlan, result: Result, question: str) -> Result:
        """Search without LLM integration - returns raw formatted hits."""
        all_hits: List[Hit] = []

        # Execute each planned tool and collect hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=5)
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=5)
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

        # Format all hits as text content without integration
        if not all_hits:
            return result
        
        unique_hits: Dict[str, Hit] = {}
        hits_without_id: List[Hit] = []
        for hit in all_hits:
            if hit.page_id:
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    existing_score = unique_hits[hit.page_id].meta.get("score", 0) if unique_hits[hit.page_id].meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                hits_without_id.append(hit)
        
        evidence_text = []
        sources = []
        seen_sources = set()
        
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, key=lambda h: h.meta.get("score", 0) if h.meta else 0, reverse=True)
        
        for i, hit in enumerate(sorted_hits, 1):
            source_info = f"[{hit.source}]({hit.page_id})" if hit.page_id else f"[{hit.source}]"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            
            if hit.page_id and hit.page_id not in seen_sources:
                sources.append(hit.page_id)
                seen_sources.add(hit.page_id)
        
        formatted_content = "\n".join(evidence_text)
        
        return Result(
            content=formatted_content if formatted_content else result.content,
            sources=sources if sources else result.sources
        )

    def _integrate(
        self, 
        hits: List[Hit], 
        result: Result, 
        question: str,
        integration_prompt: Optional[str] = None
    ) -> Result:
        """Integrate search hits with LLM to generate answer."""
        
        evidence_text = []
        sources = []
        sources_with_scores = []  # Store sources with their relevance scores
        
        top_hits = hits[:2]
        
        for i, hit in enumerate(top_hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            snippet = hit.snippet[:150] + "..." if len(hit.snippet) > 150 else hit.snippet
            evidence_text.append(f"{i}. {source_info} {snippet}")
            
            if hit.page_id:
                sources.append(hit.page_id)
                # Extract relevance score from hit metadata
                relevance_score = hit.meta.get("score", 0.0) if hit.meta else 0.0
                sources_with_scores.append({
                    "page_id": hit.page_id,
                    "relevance_score": float(relevance_score),
                    "snippet": snippet,
                    "source_type": hit.source
                })
        
        evidence_context = "\n".join(evidence_text) if evidence_text else "No search results"
        
        system_prompt = self.system_prompts.get("integration")
        template_prompt = Integrate_PROMPT.format(
            question=question, 
            evidence_context=evidence_context,
            result=result.content if result.content else "No previous information."
        )
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            response = self.generator.generate_single(prompt=prompt, schema=INTEGRATE_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            
            llm_sources = data.get("sources", [])
            if llm_sources:
                sources = [str(s) for s in llm_sources if s is not None] or sources
            
            return Result(
                content=data.get("content", ""),
                sources=sources,
                retrieval_metadata=sources_with_scores[:2]
            )
        except Exception as e:
            print(f"Error in integration: {e}")
            return result


    def _search_by_bm25(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        """Search using BM25 retriever"""
        r = self.retrievers.get("bm25")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in BM25 search: {e}")
                return []
        # naive fallback: scan pages for substring
        out: List[List[Hit]] = []
        for query in query_list:
            query_hits: List[Hit] = []
            q = query.lower()
            for i, p in enumerate(self.page_store.load()):
                if q in p.content.lower() or q in p.header.lower():
                    snippet = p.content
                    query_hits.append(Hit(page_id=str(i), snippet=snippet, source="bm25", meta={}))
                    if len(query_hits) >= top_k:
                        break
            out.append(query_hits)
        return out
    
    def _search_by_keyword(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        """Alias for _search_by_bm25 for backward compatibility"""
        return self._search_by_bm25(query_list, top_k)

    def _search_by_dense(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        """Search using Dense retriever"""
        r = self.retrievers.get("dense")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in dense search: {e}")
                return []
        # fallback: none
        return []
    
    def _search_by_vector(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        """Alias for _search_by_dense for backward compatibility"""
        return self._search_by_dense(query_list, top_k)

    def _search_by_page_index(self, page_index: List[int]) -> List[List[Hit]]:
        r = self.retrievers.get("page_index")
        if r is not None:
            try:
                # IndexRetriever 现在期望 List[str]，将 page_index 转换为逗号分隔的字符串
                query_string = ",".join([str(idx) for idx in page_index])
                hits = r.search([query_string], top_k=len(page_index))
                return hits if hits else []
            except Exception as e:
                print(f"Error in page index search: {e}")
                return []
        
        # fallback: 直接通过 page_store 获取页面
        out: List[Hit] = []
        for idx in page_index:
            p = self.page_store.get(idx)
            if p:
                out.append(Hit(page_id=str(idx), snippet=p.content, source="page_index", meta={}))
        return [out]  # 包装成 List[List[Hit]] 格式
        
        

    # ---- reflection & summarization ----
    def _reflection(
        self, 
        request: str, 
        result: Result,
        reflection_prompt: Optional[str] = None
    ) -> ReflectionDecision:
        """Fast reflection - returns enough=True after first iteration."""
        if result.content and len(result.content.strip()) > 20:
            return ReflectionDecision(enough=True, new_request=None)
        
        return ReflectionDecision(enough=True, new_request=None)