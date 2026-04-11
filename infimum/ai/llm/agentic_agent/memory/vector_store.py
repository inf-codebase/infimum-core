"""
Vector-based memory storage for long-term memory retrieval.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from ..config import get_settings


class VectorMemoryStore:
    """
    Vector-based memory store using ChromaDB for long-term memory.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        collection_name: str = "agent_memory",
        persist_directory: Optional[str] = None,
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Install with: pip install chromadb")
        
        self.settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./data/vector_store"
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )
        
        # Initialize ChromaDB
        self._init_vector_store()
    
    def _init_vector_store(self) -> None:
        """Initialize the vector store."""
        try:
            # ChromaDB settings for persistence
            chroma_settings = Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            )
            
            # Initialize Chroma client
            self.chroma_client = chromadb.Client(settings=chroma_settings)
            
            # Initialize LangChain Chroma wrapper
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            
        except Exception as e:
            # Fallback to in-memory store
            print(f"Warning: Failed to initialize persistent vector store: {e}")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
    
    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a memory to the vector store."""
        if not content.strip():
            return ""
        
        # Prepare metadata
        memory_metadata = {
            "timestamp": datetime.now().isoformat(),
            "content_length": len(content),
        }
        if metadata:
            memory_metadata.update(metadata)
        
        # Create document
        document = Document(
            page_content=content,
            metadata=memory_metadata,
        )
        
        # Add to vector store
        try:
            doc_ids = self.vector_store.add_documents([document])
            return doc_ids[0] if doc_ids else ""
        except Exception as e:
            print(f"Warning: Failed to add memory to vector store: {e}")
            return ""
    
    def search_memories(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        try:
            # Perform similarity search
            if filter_metadata:
                docs = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_metadata,
                )
            else:
                docs = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                )
            
            # Format results
            results = []
            for doc, score in docs:
                if score >= score_threshold:  # Filter by relevance score
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": score,
                    })
            
            # Sort by relevance score (higher is better)
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return results
            
        except Exception as e:
            print(f"Warning: Memory search failed: {e}")
            return []
    
    def get_memories_by_session(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get memories for a specific session."""
        return self.search_memories(
            query="",  # Empty query to get all
            k=limit,
            filter_metadata={"session_id": session_id},
            score_threshold=0.0,  # Include all results
        )
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        try:
            self.vector_store.delete([memory_id])
            return True
        except Exception as e:
            print(f"Warning: Failed to delete memory: {e}")
            return False
    
    def clear_session(self, session_id: str) -> int:
        """Clear all memories for a session."""
        try:
            # Get all memories for the session
            session_memories = self.get_memories_by_session(session_id)
            
            # Delete each memory
            deleted_count = 0
            for memory in session_memories:
                if "id" in memory.get("metadata", {}):
                    if self.delete_memory(memory["metadata"]["id"]):
                        deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            print(f"Warning: Failed to clear session memories: {e}")
            return 0
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories (for debugging/admin purposes)."""
        try:
            # Get collection
            collection = self.chroma_client.get_collection(self.collection_name)
            
            # Get all documents
            results = collection.get()
            
            memories = []
            if results and "documents" in results:
                for i, doc in enumerate(results["documents"]):
                    metadata = results.get("metadatas", [{}])[i] if i < len(results.get("metadatas", [])) else {}
                    memories.append({
                        "content": doc,
                        "metadata": metadata,
                    })
            
            return memories
            
        except Exception as e:
            print(f"Warning: Failed to retrieve all memories: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            all_memories = self.get_all_memories()
            
            return {
                "total_memories": len(all_memories),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embeddings.model,
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get stats: {e}",
                "collection_name": self.collection_name,
            }
    
    def persist(self) -> None:
        """Persist the vector store to disk."""
        try:
            if hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
        except Exception as e:
            print(f"Warning: Failed to persist vector store: {e}")