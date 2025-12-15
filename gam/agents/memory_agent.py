# memory_agent.py
# -*- coding: utf-8 -*-
"""
MemoryAgent Module

This module defines the MemoryAgent for the GAM (General-Agentic-Memory) framework.

- Memory is represented as a list[str] of abstracts (no events/tags included).
- MemoryAgent exposes only: memorize(message) -> MemoryUpdate, allowing the agent to store new information.
- Prompts within the module are used as placeholders for future prompt templates or instructions.
"""


from __future__ import annotations

from typing import Dict, Optional, Tuple

from gam.prompts import MemoryAgent_PROMPT
from gam.schemas import (
    MemoryState, Page, MemoryUpdate, MemoryStore, PageStore,
    InMemoryMemoryStore, InMemoryPageStore, Retriever
)
from gam.generator import AbsGenerator

class MemoryAgent:
    """
    Public API:
      - memorize(message) -> MemoryUpdate
    Internal only:
      - _decorate(message, memory_state) -> (abstract, header, decorated_new_page)
    Note: memory_state contains ONLY abstracts (list[str]).
    """

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        page_store: PageStore | None = None,
        generator: AbsGenerator | None = None,  # 必须传入Generator实例
        dir_path: Optional[str] = None,  # 新增：文件系统存储路径
        system_prompts: Optional[Dict[str, str]] = None,  # 新增：system prompts字典
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for MemoryAgent")
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.page_store = page_store or InMemoryPageStore(dir_path=dir_path)
        self.generator = generator
        
        # 初始化 system_prompts，默认值为空字符串
        default_system_prompts = {
            "memory": ""
        }
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:
            # 合并用户提供的 prompts 和默认值
            self.system_prompts = {**default_system_prompts, **system_prompts}


    # ---- Public ----
    def memorize(self, message: str) -> MemoryUpdate:
        """
        Update long-term memory with a new message and persist a decorated page.
        Steps:
          1) _decorate(...) => abstract, header, decorated_new_page
          2) Merge into MemoryState (append unique abstract)
          3) Write Page into page_store  (page_id left None by default)
        """
        message = message.strip()
        state = self.memory_store.load()

        # (1) Decorate - this generates the abstract and decorated page
        abstract, header, decorated_new_page = self._decorate(message, state)

        # (2) Add abstract to memory (with built-in uniqueness check)
        self.memory_store.add(abstract)

        # (3) Persist page
        page = Page(header=header, content=message, meta={"decorated": decorated_new_page})
        self.page_store.add(page)
        
        # (4) Get updated state after adding abstract
        updated_state = self.memory_store.load()

        return MemoryUpdate(new_state=updated_state, new_page=page, debug={"decorated_page": decorated_new_page})

    def memorize_batch(self, messages: list[str]) -> list[MemoryUpdate]:
        """
        Batch update long-term memory with multiples messages and persist decorated pages.
        Crucially, this uses generator.generate_batch for parallel processing.
        """
        messages = [m.strip() for m in messages if m.strip()]
        if not messages:
            return []
            
        state = self.memory_store.load()
        
        # Prepare prompts for all messages
        # Note: We use the same current memory state for all chunks in this batch
        # This is the trade-off for parallelization
        prompts = []
        
        # Build memory context from all abstracts (concatenate all memories)
        if state.abstracts:
            memory_context_lines = []
            for i, abstract in enumerate(state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        else:
            memory_context = "No memory currently."
            
        system_prompt = self.system_prompts.get("memory")
        
        for message in messages:
            template_prompt = MemoryAgent_PROMPT.format(
                input_message=message,
                memory_context=memory_context
            )
            if system_prompt:
                prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
            else:
                prompt = template_prompt
            prompts.append(prompt)
            
        # Generate all abstracts in parallel
        try:
            responses = self.generator.generate_batch(prompts=prompts)
        except Exception as e:
            print(f"Error generating batch abstracts: {e}")
            # Fallback to empty abstracts on failure
            responses = [{"text": m[:200]} for m in messages]
            
        results = []
        for i, message in enumerate(messages):
            abstract = responses[i].get("text", "").strip()
            if not abstract:
                abstract = message[:200]
                
            header = f"[ABSTRACT] {abstract}".strip()
            decorated_new_page = f"{header}; {message}"
            
            # Update stores
            self.memory_store.add(abstract)
            page = Page(header=header, content=message, meta={"decorated": decorated_new_page})
            self.page_store.add(page)
            
            # For result, we return the state as it is *after* this specific addition
            # (Though in reality for the caller, only the final state might matter)
            current_state = self.memory_store.load()
            results.append(MemoryUpdate(
                new_state=current_state,
                new_page=page,
                debug={"decorated_page": decorated_new_page}
            ))
            
        return results


    # ---- Internal----

    def _decorate(self, message: str, memory_state: MemoryState) -> Tuple[str, str, str]:
        """
        Private. Generate abstract for the message and compose: "abstract; header; new_page".
        Returns: (abstract, header, decorated_new_page)
        """
        # Build memory context from all abstracts (concatenate all memories)
        if memory_state.abstracts:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        else:
            memory_context = "No memory currently."
        
        # Generate abstract for the current message using LLM with memory context
        system_prompt = self.system_prompts.get("memory")
        template_prompt = MemoryAgent_PROMPT.format(
            input_message=message,
            memory_context=memory_context
        )
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt
        
        try:
            response = self.generator.generate_single(prompt=prompt)
            abstract = response.get("text", "").strip()
        except Exception as e:
            print(f"Error generating abstract: {e}")
            abstract = message[:200]
        
        # Create header with the new abstract
        header = f"[ABSTRACT] {abstract}".strip()
        decorated_new_page = f"{header}; {message}"
        return abstract, header, decorated_new_page