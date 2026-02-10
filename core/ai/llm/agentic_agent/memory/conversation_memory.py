"""
Custom conversation memory with enhanced features.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

from langchain_core.messages import BaseMessage


class ConversationMemory:
    """
    Custom conversation memory with additional features like
    metadata tracking, conversation topics, and message categorization.
    """
    
    def __init__(
        self,
        session_id: str,
        max_messages: int = 100,
        max_topics: int = 10,
    ):
        self.session_id = session_id
        self.max_messages = max_messages
        self.max_topics = max_topics
        
        # Store messages with metadata
        self.messages: deque = deque(maxlen=max_messages)
        
        # Track conversation topics
        self.topics: deque = deque(maxlen=max_topics)
        
        # Session metadata
        self.metadata: Dict[str, Any] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_count": 0,
            "topics": [],
        }
        
        # Conversation statistics
        self.stats = {
            "user_messages": 0,
            "ai_messages": 0,
            "tool_calls": 0,
            "errors": 0,
            "average_response_length": 0.0,
        }
    
    def add_message(
        self,
        message: BaseMessage,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message with optional metadata."""
        message_data = {
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "index": len(self.messages),
        }
        
        self.messages.append(message_data)
        self._update_stats(message, metadata)
        self._update_metadata()
        
        # Extract and track topics
        if len(message.content) > 20:  # Only for substantial messages
            self._extract_topic(message.content)
    
    def _update_stats(
        self,
        message: BaseMessage,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update conversation statistics."""
        message_type = message.__class__.__name__.lower()
        
        if "human" in message_type:
            self.stats["user_messages"] += 1
        elif "ai" in message_type:
            self.stats["ai_messages"] += 1
            
            # Update average response length
            total_ai_length = self.stats["average_response_length"] * (self.stats["ai_messages"] - 1)
            new_length = len(message.content)
            self.stats["average_response_length"] = (total_ai_length + new_length) / self.stats["ai_messages"]
        
        if metadata:
            if metadata.get("tool_call"):
                self.stats["tool_calls"] += 1
            if metadata.get("error"):
                self.stats["errors"] += 1
    
    def _update_metadata(self) -> None:
        """Update session metadata."""
        self.metadata.update({
            "last_updated": datetime.now().isoformat(),
            "message_count": len(self.messages),
            "topics": list(self.topics),
        })
    
    def _extract_topic(self, content: str) -> None:
        """Extract and track conversation topics (simple keyword-based)."""
        # Simple topic extraction based on common patterns
        content_lower = content.lower()
        
        # Check for common topic indicators
        topic_keywords = {
            "weather": ["weather", "temperature", "rain", "snow", "sunny", "cloudy"],
            "calculation": ["calculate", "math", "compute", "formula", "equation"],
            "search": ["search", "find", "look up", "google", "information"],
            "time": ["time", "date", "schedule", "when", "today", "tomorrow"],
            "finance": ["money", "investment", "interest", "finance", "stock", "price"],
            "technology": ["code", "programming", "software", "computer", "tech"],
        }
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)
        
        # Add detected topics
        for topic in detected_topics:
            if topic not in self.topics:
                self.topics.append(topic)
    
    def get_messages(
        self,
        limit: Optional[int] = None,
        message_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get messages with optional filtering."""
        messages = list(self.messages)
        
        # Filter by message type if specified
        if message_type:
            messages = [
                msg for msg in messages
                if message_type.lower() in msg["message"].__class__.__name__.lower()
            ]
        
        # Apply limit
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_recent_context(self, num_messages: int = 5) -> str:
        """Get recent conversation context as a formatted string."""
        recent_messages = self.get_messages(limit=num_messages)
        
        context_lines = []
        for msg_data in recent_messages:
            message = msg_data["message"]
            timestamp = msg_data["timestamp"]
            
            # Format message
            if "human" in message.__class__.__name__.lower():
                role = "User"
            elif "ai" in message.__class__.__name__.lower():
                role = "Assistant"
            else:
                role = "System"
            
            # Truncate long messages
            content = message.content[:200]
            if len(message.content) > 200:
                content += "..."
            
            context_lines.append(f"[{timestamp}] {role}: {content}")
        
        return "\n".join(context_lines)
    
    def search_messages(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search messages by content."""
        query_lower = query.lower()
        matching_messages = []
        
        for msg_data in self.messages:
            if query_lower in msg_data["message"].content.lower():
                matching_messages.append(msg_data)
        
        # Sort by recency (most recent first)
        matching_messages.sort(
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return matching_messages[:limit]
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        return {
            "session_id": self.session_id,
            "duration": self._get_session_duration(),
            "message_count": len(self.messages),
            "topics": list(self.topics),
            "statistics": self.stats.copy(),
            "metadata": self.metadata.copy(),
        }
    
    def _get_session_duration(self) -> str:
        """Calculate session duration."""
        if not self.messages:
            return "0 minutes"
        
        created_at = datetime.fromisoformat(self.metadata["created_at"])
        last_updated = datetime.fromisoformat(self.metadata["last_updated"])
        
        duration = last_updated - created_at
        minutes = int(duration.total_seconds() / 60)
        
        if minutes < 1:
            return "< 1 minute"
        elif minutes < 60:
            return f"{minutes} minutes"
        else:
            hours = minutes // 60
            remaining_minutes = minutes % 60
            return f"{hours}h {remaining_minutes}m"
    
    def clear(self) -> None:
        """Clear all conversation data."""
        self.messages.clear()
        self.topics.clear()
        
        # Reset stats
        self.stats = {
            "user_messages": 0,
            "ai_messages": 0,
            "tool_calls": 0,
            "errors": 0,
            "average_response_length": 0.0,
        }
        
        # Update metadata
        self.metadata.update({
            "last_updated": datetime.now().isoformat(),
            "message_count": 0,
            "topics": [],
        })
    
    def get_session_metadata(self) -> Dict[str, Any]:
        """Get session metadata for persistence."""
        return {
            "session_id": self.session_id,
            "metadata": self.metadata.copy(),
            "statistics": self.stats.copy(),
            "topics": list(self.topics),
        }