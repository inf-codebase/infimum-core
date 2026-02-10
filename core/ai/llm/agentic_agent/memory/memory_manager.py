"""
Memory management system for the AI Agent v2.
Integrates with LangChain memory components for production use.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from .vector_store import VectorMemoryStore
from .conversation_memory import ConversationMemory
from ..config import get_settings


class MemoryManager:
    """
    Comprehensive memory management system using LangChain components.
    
    Combines multiple memory types:
    - Short-term conversation memory
    - Long-term vector-based memory
    - Summarization for context compression
    """
    
    def __init__(
        self,
        memory_type: str = "buffer",
        max_tokens: int = 2000,
        use_vector_memory: bool = True,
        session_id: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.memory_type = memory_type
        self.max_tokens = max_tokens
        self.session_id = session_id or f"session_{datetime.now().isoformat()}"
        
        # Initialize LLM for summarization
        self.llm = ChatOpenAI(
            model=self.settings.openai_model,
            temperature=0.1,
            openai_api_key=self.settings.openai_api_key,
        )
        
        # Initialize conversation memory
        self._init_conversation_memory()
        
        # Initialize vector memory if enabled
        self.vector_memory = None
        if use_vector_memory:
            try:
                self.vector_memory = VectorMemoryStore(
                    embedding_model=self.settings.embedding_model,
                    collection_name=f"memory_{self.session_id}",
                )
            except Exception as e:
                print(f"Warning: Vector memory initialization failed: {e}")
        
        # Custom conversation memory for additional features
        self.custom_memory = ConversationMemory(session_id=self.session_id)
    
    def _init_conversation_memory(self) -> None:
        """Initialize LangChain conversation memory based on type."""
        if self.memory_type == "summary":
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.max_tokens,
                return_messages=True,
                memory_key="chat_history",
            )
        else:  # buffer (default)
            self.memory = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                return_messages=True,
                memory_key="chat_history",
            )
    
    def add_message(
        self,
        message: BaseMessage,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to all memory systems."""
        # Add to LangChain memory
        if isinstance(message, HumanMessage):
            self.memory.chat_memory.add_user_message(message.content)
        elif isinstance(message, AIMessage):
            self.memory.chat_memory.add_ai_message(message.content)
        
        # Add to custom memory with metadata
        self.custom_memory.add_message(message, metadata)
        
        # Add to vector memory for long-term storage
        if self.vector_memory and len(message.content) > 50:
            try:
                self.vector_memory.add_memory(
                    content=message.content,
                    metadata={
                        "type": message.__class__.__name__,
                        "session_id": self.session_id,
                        "timestamp": datetime.now().isoformat(),
                        **(metadata or {}),
                    }
                )
            except Exception as e:
                print(f"Warning: Vector memory storage failed: {e}")
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a user message."""
        message = HumanMessage(content=content)
        self.add_message(message, metadata)
    
    def add_ai_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an AI message."""
        message = AIMessage(content=content)
        self.add_message(message, metadata)
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """Get conversation history from LangChain memory."""
        messages = self.memory.chat_memory.messages
        if limit:
            return messages[-limit:]
        return messages
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for LangChain integration."""
        return self.memory.load_memory_variables({})
    
    def get_context_for_prompt(self, query: str, max_results: int = 3) -> str:
        """Get relevant context for the current query."""
        context_parts = []
        
        # Get conversation history
        history = self.get_conversation_history(limit=5)
        if history:
            recent_context = "Recent conversation:\n"
            for msg in history[-3:]:  # Last 3 messages
                role = "Human" if isinstance(msg, HumanMessage) else "AI"
                recent_context += f"{role}: {msg.content[:200]}\n"
            context_parts.append(recent_context)
        
        # Get relevant memories from vector store
        if self.vector_memory:
            try:
                relevant_memories = self.vector_memory.search_memories(
                    query=query,
                    k=max_results,
                )
                if relevant_memories:
                    vector_context = "Relevant past information:\n"
                    for memory in relevant_memories:
                        vector_context += f"- {memory['content'][:150]}...\n"
                    context_parts.append(vector_context)
            except Exception as e:
                print(f"Warning: Vector memory search failed: {e}")
        
        return "\n\n".join(context_parts)
    
    def clear_session_memory(self) -> None:
        """Clear current session memory."""
        self.memory.clear()
        self.custom_memory.clear()
        
        if self.vector_memory:
            try:
                self.vector_memory.clear_session(self.session_id)
            except Exception as e:
                print(f"Warning: Vector memory clear failed: {e}")
    
    def save_session(self) -> Dict[str, Any]:
        """Save session data for persistence."""
        return {
            "session_id": self.session_id,
            "messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content,
                }
                for msg in self.get_conversation_history()
            ],
            "metadata": self.custom_memory.get_session_metadata(),
            "timestamp": datetime.now().isoformat(),
        }
    
    def load_session(self, session_data: Dict[str, Any]) -> None:
        """Load session data from persistence."""
        self.session_id = session_data["session_id"]
        
        # Recreate conversation memory
        self._init_conversation_memory()
        
        # Reload messages
        for msg_data in session_data.get("messages", []):
            if msg_data["type"] == "HumanMessage":
                self.add_user_message(msg_data["content"])
            elif msg_data["type"] == "AIMessage":
                self.add_ai_message(msg_data["content"])
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            "session_id": self.session_id,
            "memory_type": self.memory_type,
            "conversation_messages": len(self.get_conversation_history()),
            "vector_memory_enabled": self.vector_memory is not None,
        }
        
        if self.vector_memory:
            try:
                stats["vector_memory_count"] = len(self.vector_memory.get_all_memories())
            except:
                stats["vector_memory_count"] = "unknown"
        
        return stats