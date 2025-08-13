"""
Enhanced panel components with additional features
"""

from textual.app import ComposeResult
from textual.widgets import TextArea, ListView, ListItem, Label, Static
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual.widgets.text_area import TextAreaTheme
from rich.style import Style
from datetime import datetime
import difflib
from typing import Set, Optional, Dict


class EnhancedCodeEditor(TextArea):
    """Enhanced code editor with diff highlighting features."""
    
    DEFAULT_CSS = """
    EnhancedCodeEditor {
        height: 100%;
    }
    
    EnhancedCodeEditor.diff-mode {
        border: solid #39FF14 60%;
    }
    
    /* CSS approach for diff line highlighting */
    EnhancedCodeEditor.diff-mode .diff-added {
        background: #1a4a1a;
        color: #90ee90;
    }
    
    EnhancedCodeEditor.diff-mode .diff-modified {
        background: #4a4a1a;
        color: #ffd700;
    }
    
    EnhancedCodeEditor.diff-mode .diff-deleted {
        background: #4a1a1a;
        color: #ff6b6b;
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
        
        # Enhanced diff state
        self.pre_refinement_code: Optional[str] = None
        self.refined_code_clean: Optional[str] = None  # Store clean refined code
        self.diff_lines: Set[int] = set()
        self.diff_enabled = False
        self.original_theme = "dracula"
        self.diff_data: Dict[int, Dict[str, str]] = {}  # Line number -> diff info
        
    
    def store_pre_refinement_code(self):
        """Store the current code before refinement."""
        self.pre_refinement_code = self.text
        self.diff_lines.clear()
        self.diff_enabled = False
    
    def compute_enhanced_diff(self, refined_code: str) -> Dict[int, Dict[str, str]]:
        """
        Compute detailed word-level diff information for changed lines.
        
        Returns:
            Dict mapping line number -> {old_line, new_line, changes_desc}
        """
        if not self.pre_refinement_code:
            return {}
        
        original_lines = self.pre_refinement_code.splitlines()
        refined_lines = refined_code.splitlines()
        
        # Use SequenceMatcher to find differences at line level
        matcher = difflib.SequenceMatcher(None, original_lines, refined_lines)
        diff_data = {}
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Lines were modified - do word-level diff for each pair
                for idx in range(min(i2 - i1, j2 - j1)):
                    old_line = original_lines[i1 + idx] if i1 + idx < len(original_lines) else ""
                    new_line = refined_lines[j1 + idx] if j1 + idx < len(refined_lines) else ""
                    line_num = j1 + idx  # Use refined code line number
                    
                    if old_line != new_line:
                        changes_desc = self._compute_word_level_changes(old_line, new_line)
                        diff_data[line_num] = {
                            'old_line': old_line,
                            'new_line': new_line,
                            'changes_desc': changes_desc,
                            'type': 'modified'
                        }
            
            elif tag == 'insert':
                # Lines were added
                for idx in range(j2 - j1):
                    line_num = j1 + idx
                    new_line = refined_lines[line_num]
                    diff_data[line_num] = {
                        'old_line': '',
                        'new_line': new_line,
                        'changes_desc': f"Added entire line: {new_line.strip()}",
                        'type': 'added'
                    }
            
            elif tag == 'delete':
                # Lines were deleted - mark next line if it exists
                if j1 < len(refined_lines):
                    diff_data[j1] = {
                        'old_line': original_lines[i1] if i1 < len(original_lines) else '',
                        'new_line': refined_lines[j1],
                        'changes_desc': f"Deleted line: {original_lines[i1].strip() if i1 < len(original_lines) else ''}",
                        'type': 'deleted_before'
                    }
        
        return diff_data
    
    def _compute_word_level_changes(self, old_line: str, new_line: str) -> str:
        """
        Compute word-level changes between two lines.
        
        Returns a human-readable description of what changed.
        """
        old_words = old_line.split()
        new_words = new_line.split()
        
        matcher = difflib.SequenceMatcher(None, old_words, new_words)
        changes = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                old_text = ' '.join(old_words[i1:i2])
                new_text = ' '.join(new_words[j1:j2])
                changes.append(f'"{old_text}" → "{new_text}"')
            elif tag == 'delete':
                deleted_text = ' '.join(old_words[i1:i2])
                changes.append(f'removed "{deleted_text}"')
            elif tag == 'insert':
                added_text = ' '.join(new_words[j1:j2])
                changes.append(f'added "{added_text}"')
        
        if changes:
            return f"Changes: {', '.join(changes)}"
        else:
            return "Line modified (whitespace/formatting changes)"
    
    def apply_diff_highlighting(self, refined_code: str):
        """Apply enhanced diff highlighting to show detailed changes from refinement."""
        # Store the clean refined code for later restoration
        self.refined_code_clean = refined_code
        
        # Compute detailed diff information
        self.diff_data = self.compute_enhanced_diff(refined_code)
        self.diff_lines = set(self.diff_data.keys())
        
        if self.diff_lines:
            self.diff_enabled = True
            self.add_class("diff-mode")
            
            # Apply enhanced diff indicators to changed lines
            self._apply_enhanced_diff_indicators(refined_code)
    
    def _apply_enhanced_diff_indicators(self, code: str):
        """Apply enhanced diff indicators showing detailed word-level changes."""
        if not self.diff_enabled or not self.diff_data:
            return
        
        lines = code.splitlines()
        enhanced_lines = []
        
        for i, line in enumerate(lines):
            if i in self.diff_data:
                diff_info = self.diff_data[i]
                change_type = diff_info['type']
                changes_desc = diff_info['changes_desc']
                
                # Choose indicator based on change type - make them more prominent
                if change_type == 'added':
                    indicator = f"  # 🟢 ADDED → {changes_desc}"
                elif change_type == 'deleted_before':
                    indicator = f"  # 🔴 DELETED → {changes_desc}"
                elif change_type == 'modified':
                    indicator = f"  # 🟡 CHANGED → {changes_desc}"
                else:
                    indicator = f"  # 🔵 MODIFIED → {changes_desc}"
                
                # Add the detailed change information
                if line.strip() and not line.rstrip().endswith('#'):
                    enhanced_lines.append(f"{line}{indicator}")
                elif line.strip().startswith('#'):
                    # For existing comments, add indicator on new line
                    enhanced_lines.append(line)
                    enhanced_lines.append(f"#{indicator}")
                else:
                    # For empty lines that changed
                    enhanced_lines.append(f"{line}{indicator}")
            else:
                enhanced_lines.append(line)
        
        # Load the enhanced code
        enhanced_code = '\n'.join(enhanced_lines)
        self.load_text(enhanced_code)
    
    def clear_diff_highlighting(self):
        """Clear diff highlighting and return to normal view."""
        self.diff_enabled = False
        self.diff_lines.clear()
        self.diff_data.clear()
        self.remove_class("diff-mode")
        
        self.pre_refinement_code = None
        self.refined_code_clean = None
    
    def toggle_diff_highlighting(self) -> bool:
        """Toggle enhanced diff highlighting on/off. Returns new state."""
        if self.diff_enabled:
            # Simply restore the clean refined code we stored earlier
            if self.refined_code_clean:
                self.load_text(self.refined_code_clean)
            
            self.diff_enabled = False
            self.remove_class("diff-mode")
            return False
        elif self.pre_refinement_code and self.diff_data and self.refined_code_clean:
            self.diff_enabled = True
            self.add_class("diff-mode")
            # Re-apply the enhanced diff indicators using the clean code
            self._apply_enhanced_diff_indicators(self.refined_code_clean)
            return True
        return False
    
    def get_diff_summary(self) -> str:
        """Get an enhanced summary of the changes."""
        if not self.pre_refinement_code:
            return "No refinement applied yet"
        
        if not self.diff_data:
            return "No changes detected"
        
        added_count = sum(1 for info in self.diff_data.values() if info['type'] == 'added')
        modified_count = sum(1 for info in self.diff_data.values() if info['type'] == 'modified')
        deleted_count = sum(1 for info in self.diff_data.values() if info['type'] == 'deleted_before')
        
        summary_parts = []
        if added_count > 0:
            summary_parts.append(f"{added_count} added")
        if modified_count > 0:
            summary_parts.append(f"{modified_count} modified")
        if deleted_count > 0:
            summary_parts.append(f"{deleted_count} deleted")
        
        if summary_parts:
            return f"{len(self.diff_data)} lines changed: {', '.join(summary_parts)}"
        else:
            return f"{len(self.diff_data)} lines changed"
    
    def on_text_area_changed(self):
        """Track when the code has been modified."""
        self.modified = True
    
    def save_state(self):
        """Mark the current state as saved."""
        self.last_saved = datetime.now()
        self.modified = False
    
    def get_clean_code(self) -> str:
        """
        Get clean code without diff markers for LLM refinement.
        
        If diff highlighting is enabled, returns the clean refined code.
        Otherwise, strips any remaining diff markers from the current text.
        """
        if self.diff_enabled and self.refined_code_clean:
            # Return the stored clean version
            return self.refined_code_clean
        else:
            # Strip any enhanced diff markers from current text
            current_text = self.text
            lines = current_text.splitlines()
            clean_lines = []
            
            for line in lines:
                # Remove enhanced diff markers with new format
                if '  # 🟢 ADDED →' in line:
                    clean_line = line.split('  # 🟢 ADDED →')[0].rstrip()
                    clean_lines.append(clean_line)
                elif '  # 🔴 DELETED →' in line:
                    clean_line = line.split('  # 🔴 DELETED →')[0].rstrip()
                    clean_lines.append(clean_line)
                elif '  # 🟡 CHANGED →' in line:
                    clean_line = line.split('  # 🟡 CHANGED →')[0].rstrip()
                    clean_lines.append(clean_line)
                elif '  # 🔵 MODIFIED →' in line:
                    clean_line = line.split('  # 🔵 MODIFIED →')[0].rstrip()
                    clean_lines.append(clean_line)
                # Handle old-style markers for backward compatibility
                elif '  # ✅' in line:
                    clean_line = line.split('  # ✅')[0].rstrip()
                    clean_lines.append(clean_line)
                elif '  # ❌' in line:
                    clean_line = line.split('  # ❌')[0].rstrip()
                    clean_lines.append(clean_line)
                elif '  # 🔄' in line:
                    clean_line = line.split('  # 🔄')[0].rstrip()
                    clean_lines.append(clean_line)
                elif '  # 🟢' in line:
                    clean_line = line.split('  # 🟢')[0].rstrip()
                    clean_lines.append(clean_line)
                elif line.startswith('#  # '):
                    # Skip lines that are purely diff comment additions
                    continue
                else:
                    clean_lines.append(line)
            
            return '\n'.join(clean_lines)
    
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