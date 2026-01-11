import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import openai
import logging
from src.core.utils import auto_config
from .prompts import MEMORY_PROMPTS  # Import the centralized prompts

logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, max_short_term=10):
        self.short_term = []  # Recent interactions (list of message objects)
        self.long_term = []   # Persistent storage (list of dict objects with embeddings)
        self.max_short_term = max_short_term
        self.client = openai.OpenAI(api_key=auto_config.OPENAI_API_KEY)
    
    def get_recent_entries(self):
        """Get recent entries from short-term memory
        
        Returns:
            list: List of recent memory entries
        """
        return self.short_term.copy()
    
    def add_to_short_term(self, message):
        """Add a message to short-term memory"""
        self.short_term.append(message)
        
        # Trim to max size if needed
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term:]
    
    def add_to_long_term(self, entry):
        """Add an entry to long-term memory with embedding"""
        # Create embedding for the entry
        try:
            # Create a more comprehensive text for embedding
            text_to_embed = f"""
            Query: {entry.get('query', '')}
            Response: {entry.get('response', '')}
            Steps: {entry.get('steps', [])}
            Observations: {entry.get('observations', [])}
            Reflection: {entry.get('reflection', '')}
            """
            
            embedding = self._get_embedding(text_to_embed)
            
            # Add timestamp and embedding
            entry['timestamp'] = datetime.now().isoformat()
            entry['embedding'] = embedding
            
            # Store additional metadata for better retrieval
            entry['metadata'] = {
                'step_count': len(entry.get('steps', [])),
                'observation_count': len(entry.get('observations', [])),
                'has_final_response': bool(entry.get('response')),
                'has_reflection': bool(entry.get('reflection'))
            }
            
            self.long_term.append(entry)
        except Exception as e:
            logger.error(f"Error adding to long-term memory: {str(e)}")
    
    def retrieve_relevant(self, query, k=3):
        """Retrieve k most relevant entries from long-term memory"""
        if not self.long_term:
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Calculate similarities with additional metadata weighting
            similarities = []
            for entry in self.long_term:
                if 'embedding' in entry:
                    # Calculate base similarity
                    base_similarity = cosine_similarity(
                        [query_embedding], 
                        [entry['embedding']]
                    )[0][0]
                    
                    # Apply metadata-based weighting
                    metadata = entry.get('metadata', {})
                    weight = 1.0
                    
                    # Increase weight for entries with more observations
                    weight *= (1 + 0.1 * metadata.get('observation_count', 0))
                    
                    # Increase weight for entries with final responses
                    if metadata.get('has_final_response'):
                        weight *= 1.2
                    
                    # Increase weight for entries with reflections
                    if metadata.get('has_reflection'):
                        weight *= 1.2
                    
                    # Calculate final weighted similarity
                    weighted_similarity = base_similarity * weight
                    similarities.append((weighted_similarity, entry))
            
            # Sort by weighted similarity and return top k
            similarities.sort(reverse=True, key=lambda x: x[0])
            return [entry for _, entry in similarities[:k]]
        except Exception as e:
            logger.error(f"Error retrieving from memory: {str(e)}")
            return []
    
    def get_context_for_llm(self, current_query):
        """Format messages for LLM context"""
        # Start with system message
        messages = [{"role": "system", "content": MEMORY_PROMPTS["context"]}]
        
        # Add short term memory
        messages.extend(self.short_term)
        
        # Add relevant long-term memories if they exist
        relevant_memories = self.retrieve_relevant(current_query)
        if relevant_memories:
            memory_text = "Relevant past information:\n"
            for memory in relevant_memories:
                memory_text += f"- Query: {memory['query']}\n"
                memory_text += f"  Response: {memory['response']}\n"
                memory_text += f"  Reflection: {memory['reflection']}\n\n"
            
            messages.append({"role": "system", "content": memory_text})
        
        # Add current query if not in short term already
        if messages[-1]['role'] != 'user':
            messages.append({"role": "user", "content": current_query})
        
        return messages
    
    def _get_embedding(self, text):
        """Get embedding for text using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                model=auto_config.OPENAI_TEXT_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Size of OpenAI embeddings

    def get_relevant_context(self, query, current_step, context_type="general"):
        """Get relevant context from memory for a specific query and step"""
        try:
            # Combine query and step for better context matching
            search_text = f"{query} {current_step} {context_type}"
            
            # Get relevant entries from long-term memory
            relevant_entries = self.retrieve_relevant(search_text)
            
            # Also include recent short-term memory
            recent_messages = self.short_term[-5:] if len(self.short_term) > 5 else self.short_term
            
            # Format the context
            context_text = ""
            
            # Add recent short-term memory first
            if recent_messages:
                context_text += "Recent context:\n"
                for msg in recent_messages:
                    context_text += f"- {msg['role']}: {msg['content']}\n"
                context_text += "\n"
            
            # Add relevant long-term memory
            if relevant_entries:
                context_text += f"Relevant past experiences for {context_type}:\n"
                for entry in relevant_entries:
                    if context_type == "general":
                        context_text += f"- Query: {entry.get('query', 'N/A')}\n"
                        context_text += f"  Response: {entry.get('response', 'N/A')}\n"
                        context_text += f"  Steps: {', '.join(entry.get('steps', []))}\n"
                        context_text += f"  Reflection: {entry.get('reflection', 'N/A')}\n\n"
                    elif context_type == "action":
                        if "action" in entry:
                            context_text += f"- Action: {entry['action']}\n"
                            context_text += f"  Result: {entry.get('result', 'N/A')}\n\n"
                    elif context_type == "observation":
                        if "observations" in entry:
                            context_text += f"- Observations: {', '.join(entry['observations'])}\n"
                            context_text += f"  Outcome: {entry.get('outcome', 'N/A')}\n\n"
                    elif context_type == "completion":
                        if "steps" in entry and "observations" in entry:
                            context_text += f"- Step: {entry.get('current_step', 'N/A')}\n"
                            context_text += f"  Completion Status: {entry.get('completion_status', 'N/A')}\n"
                            context_text += f"  Observations: {', '.join(entry['observations'])}\n\n"
                    elif context_type == "termination":
                        if "response" in entry and "reflection" in entry:
                            context_text += f"- Final Response: {entry['response']}\n"
                            context_text += f"  Reflection: {entry['reflection']}\n"
                            context_text += f"  Success: {entry.get('success', 'N/A')}\n\n"
            
            return context_text if context_text else None
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return None
