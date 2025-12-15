"""Enhanced Knowledge Graph Generator - extracts entities and creates semantic visualizations."""
from typing import List, Dict, Any, Optional
import re
from core.gam_manager import gam_manager


def extract_key_phrases(text: str, max_phrases: int = 3) -> List[str]:
    """
    Extract key phrases from text using simple heuristics.
    Returns the most important phrases/concepts.
    """
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    
    if not sentences:
        return []
    
    # Take first sentence as primary concept
    phrases = []
    if sentences:
        first_sentence = sentences[0]
        # Extract noun phrases (simple heuristic: capitalized words or phrases with 2-4 words)
        words = first_sentence.split()
        if len(words) <= 5:
            phrases.append(first_sentence)
        else:
            phrases.append(' '.join(words[:5]) + '...')
    
    return phrases[:max_phrases]


def generate_chunk_summary(content: str, max_length: int = 80) -> str:
    """
    Generate a concise summary of a chunk for display in the graph.
    """
    # Clean the content
    content = content.strip()
    
    # Extract first meaningful sentence or phrase
    sentences = [s.strip() for s in re.split(r'[.!?]', content) if s.strip() and len(s.strip()) > 10]
    
    if sentences:
        summary = sentences[0]
        if len(summary) > max_length:
            summary = summary[:max_length] + '...'
        return summary
    
    # Fallback: just truncate
    if len(content) > max_length:
        return content[:max_length] + '...'
    return content


def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Extract named entities from text using simple pattern matching.
    Returns list of entities with their types.
    """
    entities = []
    
    # Extract capitalized phrases (potential proper nouns)
    capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    matches = re.findall(capitalized_pattern, text)
    
    for match in matches[:5]:  # Limit to 5 entities
        if len(match) > 2 and match not in ['The', 'This', 'That', 'These', 'Those']:
            entities.append({
                "text": match,
                "type": "entity"
            })
    
    # Extract numbers/dates
    number_pattern = r'\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b|\b\d{4}\b'
    numbers = re.findall(number_pattern, text)
    for num in numbers[:3]:
        entities.append({
            "text": num,
            "type": "number"
        })
    
    return entities


def build_enhanced_graph(chunk_db) -> Dict[str, Any]:
    """
    Build an enhanced knowledge graph with semantic content.
    
    Returns:
        Dict with nodes, edges, and stats
    """
    all_docs = chunk_db.get_all_documents()
    
    if not all_docs:
        return {
            "nodes": [],
            "edges": [],
            "stats": {"total_documents": 0, "total_chunks": 0, "total_entities": 0}
        }
    
    nodes = []
    edges = []
    entity_map = {}  # Track entities to avoid duplicates
    entity_counter = 0
    
    # Process each document
    for doc_idx, doc in enumerate(all_docs[:10]):  # Limit to 10 documents for performance
        doc_id = doc["id"]
        filename = doc["filename"]
        
        # Add document node
        doc_node_id = f"doc_{doc_idx}"
        nodes.append({
            "id": doc_node_id,
            "label": filename[:30] + ('...' if len(filename) > 30 else ''),
            "type": "document",
            "description": f"Document: {filename}",
            "size": 20,
            "color": "#4A90E2"
        })
        
        # Get chunks for this document
        doc_chunks = chunk_db.get_document_chunks(doc_id)
        
        # Process chunks (limit to 5 per document)
        for chunk_data in doc_chunks[:5]:
            chunk_id = f"doc{doc_idx}_chunk{chunk_data['chunk_index']}"
            content = chunk_data["content"]
            
            # Generate meaningful summary
            summary = generate_chunk_summary(content)
            key_phrases = extract_key_phrases(content)
            
            # Create chunk node with better label
            chunk_label = key_phrases[0] if key_phrases else f"Chunk #{chunk_data['chunk_index']+1}"
            
            nodes.append({
                "id": chunk_id,
                "label": chunk_label[:40] + ('...' if len(chunk_label) > 40 else ''),
                "type": "chunk",
                "description": summary,
                "size": 12,
                "color": "#50C878"
            })
            
            # Connect chunk to document
            edges.append({
                "id": f"edge_doc{doc_idx}_chunk{chunk_data['chunk_index']}",
                "source": doc_node_id,
                "target": chunk_id,
                "label": "contains",
                "color": "#999"
            })
            
            # Extract entities from chunk
            entities = extract_entities(content)
            for entity in entities[:3]:  # Limit entities per chunk
                entity_text = entity["text"]
                
                # Check if entity already exists
                if entity_text not in entity_map:
                    entity_node_id = f"entity_{entity_counter}"
                    entity_map[entity_text] = entity_node_id
                    entity_counter += 1
                    
                    # Add entity node
                    nodes.append({
                        "id": entity_node_id,
                        "label": entity_text[:25],
                        "type": "entity",
                        "description": f"Entity: {entity_text}",
                        "size": 8,
                        "color": "#FFB347"
                    })
                else:
                    entity_node_id = entity_map[entity_text]
                
                # Connect chunk to entity
                edges.append({
                    "id": f"edge_{chunk_id}_{entity_node_id}",
                    "source": chunk_id,
                    "target": entity_node_id,
                    "label": "mentions",
                    "color": "#CCC"
                })
            
            # Connect sequential chunks
            if chunk_data['chunk_index'] > 0:
                prev_chunk_id = f"doc{doc_idx}_chunk{chunk_data['chunk_index']-1}"
                edges.append({
                    "id": f"edge_seq_{chunk_id}",
                    "source": prev_chunk_id,
                    "target": chunk_id,
                    "label": "next",
                    "color": "#DDD"
                })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_documents": len(all_docs),
            "total_chunks": sum(len(chunk_db.get_document_chunks(doc["id"])) for doc in all_docs),
            "total_entities": entity_counter
        }
    }
