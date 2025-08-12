"""
Enhanced panel components with additional features
"""

from textual.app import ComposeResult
from textual.widgets import TextArea, ListView, ListItem, Label, Static
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual.widgets.text_area import TextAreaTheme
from datetime import datetime
import difflib
from typing import Set, Optional


class EnhancedCodeEditor(TextArea):
    """Enhanced code editor with diff highlighting features."""
    
    DEFAULT_CSS = """
    EnhancedCodeEditor {
        height: 100%;
    }
    
    EnhancedCodeEditor.diff-mode {
        border: solid #39FF14 60%;
    }
    """
    
    def __init__(self, *args, **kwargs):
        # Extract theme and language if provided, with defaults
        theme = kwargs.pop('theme', 'dracula')
        language = kwargs.pop('language', 'python')
        show_line_numbers = kwargs.pop('show_line_numbers', True)
        
        super().__init__(
            *args,
            language=language,
            theme=theme,
            show_line_numbers=show_line_numbers,
            **kwargs
        )
        self.last_saved = None
        self.modified = False
        
        # Diff highlighting state
        self.pre_refinement_code: Optional[str] = None
        self.refined_code_clean: Optional[str] = None  # Store clean refined code
        self.diff_lines: Set[int] = set()
        self.diff_enabled = False
        self.original_theme = "dracula"
        
        # Create and register diff theme
        self._create_diff_theme()
    
    def _create_diff_theme(self):
        """Create a custom theme for diff highlighting."""
        # We can't highlight individual line numbers in the gutter,
        # so we'll just keep the normal theme and use markers instead
        pass
    
    def store_pre_refinement_code(self):
        """Store the current code before refinement."""
        self.pre_refinement_code = self.text
        self.diff_lines.clear()
        self.diff_enabled = False
    
    def compute_diff_lines(self, refined_code: str) -> Set[int]:
        """
        Compute which lines have changed between original and refined code.
        
        Returns:
            Set of 0-based line numbers that have been added or modified
        """
        if not self.pre_refinement_code:
            return set()
        
        original_lines = self.pre_refinement_code.splitlines(keepends=False)
        refined_lines = refined_code.splitlines(keepends=False)
        
        # Use SequenceMatcher to find differences
        matcher = difflib.SequenceMatcher(None, original_lines, refined_lines)
        changed_lines = set()
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('replace', 'insert'):
                # Lines j1 to j2 in the refined code are new/changed
                for line_num in range(j1, j2):
                    changed_lines.add(line_num)
            elif tag == 'delete':
                # Lines were deleted, mark the line after deletion if exists
                if j1 < len(refined_lines):
                    changed_lines.add(j1)
        
        return changed_lines
    
    def apply_diff_highlighting(self, refined_code: str):
        """Apply diff highlighting to show changes from refinement."""
        # Store the clean refined code for later restoration
        self.refined_code_clean = refined_code
        
        # Compute which lines changed
        self.diff_lines = self.compute_diff_lines(refined_code)
        
        if self.diff_lines:
            self.diff_enabled = True
            self.add_class("diff-mode")
            
            # Apply visual indicators to changed lines
            self._apply_line_highlights(refined_code)
    
    def _apply_line_highlights(self, code: str):
        """Apply visual highlights to changed lines."""
        if not self.diff_enabled or not self.diff_lines:
            return
        
        # Add subtle markers only to changed lines
        lines = code.splitlines()
        marked_lines = []
        
        for i, line in enumerate(lines):
            if i in self.diff_lines:
                # Add a subtle green dot at the end of changed lines
                # This keeps the code readable while indicating changes
                if line.strip() and not line.strip().startswith('#'):
                    marked_lines.append(f"{line}  # 🟢")
                elif line.strip().startswith('#'):
                    # For existing comments, add marker
                    marked_lines.append(f"{line} 🟢")
                else:
                    # For empty lines that changed
                    marked_lines.append(f"{line}  # 🟢")
            else:
                marked_lines.append(line)
        
        # Load the marked code
        marked_code = '\n'.join(marked_lines)
        self.load_text(marked_code)
    
    def clear_diff_highlighting(self):
        """Clear diff highlighting and return to normal view."""
        self.diff_enabled = False
        self.diff_lines.clear()
        self.remove_class("diff-mode")
        self.pre_refinement_code = None
        self.refined_code_clean = None
    
    def toggle_diff_highlighting(self) -> bool:
        """Toggle diff highlighting on/off. Returns new state."""
        if self.diff_enabled:
            # Simply restore the clean refined code we stored earlier
            if self.refined_code_clean:
                self.load_text(self.refined_code_clean)
            
            self.diff_enabled = False
            self.remove_class("diff-mode")
            return False
        elif self.pre_refinement_code and self.diff_lines and self.refined_code_clean:
            self.diff_enabled = True
            self.add_class("diff-mode")
            # Re-apply the highlights using the clean code
            self._apply_line_highlights(self.refined_code_clean)
            return True
        return False
    
    def get_diff_summary(self) -> str:
        """Get a summary of the changes."""
        if not self.pre_refinement_code:
            return "No refinement applied yet"
        
        if not self.diff_lines:
            return "No changes detected"
        
        return f"{len(self.diff_lines)} lines modified"
    
    def on_text_area_changed(self):
        """Track when the code has been modified."""
        self.modified = True
    
    def save_state(self):
        """Mark the current state as saved."""
        self.last_saved = datetime.now()
        self.modified = False
    
    def get_status(self) -> str:
        """Get the editor status."""
        status_parts = []
        
        if self.modified:
            status_parts.append("Modified")
        elif self.last_saved:
            status_parts.append(f"Saved at {self.last_saved.strftime('%H:%M:%S')}")
        else:
            status_parts.append("Ready")
        
        if self.diff_enabled:
            status_parts.append(f"Diff: {len(self.diff_lines)} changes")
        
        return " | ".join(status_parts)


class ChatMessage(Static):
    """A single chat message with role and content."""
    
    DEFAULT_CSS = """
    ChatMessage {
        margin: 0 0 1 0;
        padding: 1;
    }
    
    ChatMessage.user {
        background: $primary 20%;
        border-left: thick $primary;
    }
    
    ChatMessage.assistant {
        background: $success 20%;
        border-left: thick $success;
    }
    
    ChatMessage.error {
        background: $error 20%;
        border-left: thick $error;
    }
    
    .role {
        text-style: bold;
        margin-bottom: 0;
    }
    
    .timestamp {
        color: $text-muted;
        text-style: italic;
        float: right;
    }
    """
    
    def __init__(self, role: str, content: str, timestamp: datetime = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        
        # Format the message
        time_str = self.timestamp.strftime("%H:%M:%S")
        message_html = f"""<span class="role">{role}:</span> <span class="timestamp">{time_str}</span>
{content}"""
        
        super().__init__(message_html, classes=role.lower())


class EnhancedChatPanel(Vertical):
    """Enhanced chat panel with message history and formatting."""
    
    DEFAULT_CSS = """
    EnhancedChatPanel {
        height: 100%;
    }
    
    #chat-messages {
        height: 1fr;
        overflow-y: scroll;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }
    
    #chat-stats {
        height: 3;
        border-top: solid $primary;
        padding: 1;
        color: $text-muted;
    }
    """
    
    message_count = reactive(0)
    
    def __init__(self):
        super().__init__()
        self.messages = []
    
    def compose(self) -> ComposeResult:
        with Vertical(id="chat-messages"):
            yield Static("No messages yet", id="empty-placeholder")
        yield Static(
            "Messages: 0 | Refinements: 0",
            id="chat-stats"
        )
    
    def add_message(self, role: str, content: str):
        """Add a new message to the chat."""
        message = ChatMessage(role, content)
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': message.timestamp
        })
        
        # Remove placeholder if present
        placeholder = self.query("#empty-placeholder")
        if placeholder:
            placeholder.first().remove()
        
        # Add message to display
        messages_container = self.query_one("#chat-messages", Vertical)
        messages_container.mount(message)
        
        # Update stats
        self.message_count += 1
        self.update_stats()
        
        # Scroll to bottom
        messages_container.scroll_end()
    
    def update_stats(self):
        """Update the chat statistics display."""
        stats = self.query_one("#chat-stats", Static)
        refinement_count = len([m for m in self.messages if m['role'] == 'user'])
        stats.update(f"Messages: {self.message_count} | Refinements: {refinement_count}")
    
    def clear_messages(self):
        """Clear all messages from the chat."""
        self.messages = []
        self.message_count = 0
        messages_container = self.query_one("#chat-messages", Vertical)
        messages_container.remove_children()
        messages_container.mount(Static("No messages yet", id="empty-placeholder"))
        self.update_stats()
    
    def export_history(self) -> str:
        """Export chat history as formatted text."""
        output = []
        for msg in self.messages:
            time_str = msg['timestamp'].strftime("%H:%M:%S")
            output.append(f"[{time_str}] {msg['role']}: {msg['content']}")
        return "\n".join(output)