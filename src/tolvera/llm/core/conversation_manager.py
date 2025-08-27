"""
Conversation Manager - Manages conversational memory for sketch refinement and repair.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic_ai.messages import ModelMessage


@dataclass
class ConversationEntry:
    """Represents a single conversation exchange between user and agent."""
    timestamp: datetime = field(default_factory=datetime.now)
    user_request: str = ""
    agent_response_summary: str = ""
    interaction_type: str = "refinement"  # "refinement" or "repair"
    success: bool = True
    error_info: Optional[str] = None


class ConversationManager:
    """
    Manages conversation history for sketch refinement and repair operations.
    
    This class maintains context across multiple interactions to enable the AI
    to make more informed, context-aware decisions during iterative refinement.
    """
    
    def __init__(self, max_history_length: int = 10, max_context_tokens: int = 2000):
        """
        Initialize the conversation manager.
        
        Args:
            max_history_length: Maximum number of conversation entries to keep
            max_context_tokens: Approximate maximum tokens for conversation context
        """
        self.max_history_length = max_history_length
        self.max_context_tokens = max_context_tokens
        self.conversation_history: List[ConversationEntry] = []
        self.pydantic_message_history: List[ModelMessage] = []
        
    def add_conversation_entry(
        self,
        user_request: str,
        agent_response_summary: str,
        interaction_type: str = "refinement",
        success: bool = True,
        error_info: Optional[str] = None,
        pydantic_messages: Optional[List[ModelMessage]] = None
    ) -> None:
        """
        Add a new conversation entry to the history.
        
        Args:
            user_request: The user's refinement/repair request
            agent_response_summary: Summary of the agent's response (not full code)
            interaction_type: Type of interaction ("refinement" or "repair")
            success: Whether the interaction was successful
            error_info: Error information if the interaction failed
            pydantic_messages: Full pydantic-ai message history from this interaction
        """
        entry = ConversationEntry(
            user_request=user_request,
            agent_response_summary=agent_response_summary,
            interaction_type=interaction_type,
            success=success,
            error_info=error_info
        )
        
        self.conversation_history.append(entry)
        
        # Store pydantic message history if provided
        if pydantic_messages:
            self.pydantic_message_history.extend(pydantic_messages)
        
        # Trim history if it gets too long
        self._trim_history()
    
    def get_conversation_context(self, include_timestamps: bool = False) -> str:
        """
        Get formatted conversation context for inclusion in prompts.
        
        Args:
            include_timestamps: Whether to include timestamps in the context
            
        Returns:
            Formatted conversation history string
        """
        if not self.conversation_history:
            return "No previous conversation history."
        
        context_lines = ["## CONVERSATION HISTORY"]
        context_lines.append("Review the following conversation history to understand the user's evolving requirements:")
        context_lines.append("")
        
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp_str = ""
            if include_timestamps:
                timestamp_str = f" [{entry.timestamp.strftime('%H:%M:%S')}]"
            
            # Format the interaction
            interaction_prefix = "🔧" if entry.interaction_type == "repair" else "🧬"
            success_indicator = "✅" if entry.success else "❌"
            
            context_lines.append(f"{i}. {interaction_prefix} {entry.interaction_type.title()}{timestamp_str} {success_indicator}")
            context_lines.append(f"   User: {entry.user_request}")
            context_lines.append(f"   Agent: {entry.agent_response_summary}")
            
            if entry.error_info:
                context_lines.append(f"   Error: {entry.error_info}")
            
            context_lines.append("")
        
        context_lines.append("---")
        context_lines.append("")
        
        return "\n".join(context_lines)
    
    def get_pydantic_message_history(self) -> List[ModelMessage]:
        """
        Get the pydantic-ai message history for conversation continuity.
        
        Returns:
            List of pydantic-ai ModelMessage objects
        """
        return self.pydantic_message_history.copy()
    
    def get_recent_interactions(self, count: int = 3) -> List[ConversationEntry]:
        """
        Get the most recent conversation interactions.
        
        Args:
            count: Number of recent interactions to return
            
        Returns:
            List of recent conversation entries
        """
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def has_previous_attempts(self, user_request: str, similarity_threshold: float = 0.7) -> bool:
        """
        Check if a similar request was made before (to avoid repetitive mistakes).
        
        Args:
            user_request: The current user request
            similarity_threshold: Similarity threshold for matching (not implemented yet)
            
        Returns:
            True if similar requests were made before
        """
        # Simple keyword-based matching for now
        # Could be enhanced with semantic similarity in the future
        request_lower = user_request.lower()
        
        for entry in self.conversation_history:
            if any(word in entry.user_request.lower() for word in request_lower.split() if len(word) > 3):
                return True
        
        return False
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the conversation history.
        
        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_history:
            return {"total_interactions": 0}
        
        total = len(self.conversation_history)
        successful = sum(1 for entry in self.conversation_history if entry.success)
        refinements = sum(1 for entry in self.conversation_history if entry.interaction_type == "refinement")
        repairs = sum(1 for entry in self.conversation_history if entry.interaction_type == "repair")
        
        return {
            "total_interactions": total,
            "successful_interactions": successful,
            "failed_interactions": total - successful,
            "refinements": refinements,
            "repairs": repairs,
            "success_rate": successful / total if total > 0 else 0.0
        }
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.conversation_history.clear()
        self.pydantic_message_history.clear()
    
    def _trim_history(self) -> None:
        """Trim conversation history to stay within limits."""
        # Trim by count
        if len(self.conversation_history) > self.max_history_length:
            # Keep the most recent entries
            excess = len(self.conversation_history) - self.max_history_length
            self.conversation_history = self.conversation_history[excess:]
        
        # Trim pydantic message history - keep last N messages
        # This is approximate since we don't calculate exact token counts
        max_messages = self.max_history_length * 4  # Rough estimate: 4 messages per interaction
        if len(self.pydantic_message_history) > max_messages:
            excess = len(self.pydantic_message_history) - max_messages
            self.pydantic_message_history = self.pydantic_message_history[excess:]
    
    def export_history(self, format: str = "dict") -> Any:
        """
        Export conversation history for persistence or analysis.
        
        Args:
            format: Export format ("dict", "json")
            
        Returns:
            Exported conversation history
        """
        if format == "dict":
            return {
                "conversation_history": [
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "user_request": entry.user_request,
                        "agent_response_summary": entry.agent_response_summary,
                        "interaction_type": entry.interaction_type,
                        "success": entry.success,
                        "error_info": entry.error_info
                    }
                    for entry in self.conversation_history
                ],
                "stats": self.get_summary_stats()
            }
        elif format == "json":
            import json
            return json.dumps(self.export_history("dict"), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_history(self, data: Dict[str, Any]) -> None:
        """
        Import conversation history from exported data.
        
        Args:
            data: Previously exported conversation data
        """
        self.clear_history()
        
        if "conversation_history" in data:
            for entry_data in data["conversation_history"]:
                entry = ConversationEntry(
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    user_request=entry_data["user_request"],
                    agent_response_summary=entry_data["agent_response_summary"],
                    interaction_type=entry_data["interaction_type"],
                    success=entry_data["success"],
                    error_info=entry_data.get("error_info")
                )
                self.conversation_history.append(entry)