"""
Dialog components for the Textual UI
"""

from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, TextArea, DirectoryTree, Static
from textual import on, events
from textual.binding import Binding


class SaveDialog(ModalScreen):
    """Dialog for saving sketches with custom naming."""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
    ]
    
    DEFAULT_CSS = """
    SaveDialog {
        align: center middle;
    }
    
    #save-dialog {
        width: 60;
        height: 20;
        border: thick $background 80%;
        background: $surface;
        padding: 1;
    }
    
    #filename-input {
        margin: 1 0;
    }
    
    #path-display {
        margin: 1 0;
        color: $text-muted;
    }
    """
    
    def __init__(self, default_name: str = "sketch"):
        super().__init__()
        self.default_name = default_name
        self.result_path = None
    
    def compose(self) -> ComposeResult:
        with Container(id="save-dialog"):
            yield Label("Save Sketch", id="title")
            yield Input(
                value=self.default_name,
                placeholder="Enter filename (without .py)",
                id="filename-input"
            )
            yield Static(
                f"Will save to: examples/generated_sketches/{self.default_name}.py",
                id="path-display"
            )
            with Horizontal():
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", variant="default", id="cancel")
    
    @on(Input.Changed, "#filename-input")
    def update_path_display(self, event: Input.Changed):
        """Update the path display when filename changes."""
        filename = event.value or "sketch"
        if not filename.endswith(".py"):
            filename += ".py"
        path_display = self.query_one("#path-display", Static)
        path_display.update(f"Will save to: examples/generated_sketches/{filename}")
    
    @on(Button.Pressed, "#save")
    def save_file(self):
        """Handle save button press."""
        filename_input = self.query_one("#filename-input", Input)
        filename = filename_input.value or self.default_name
        if not filename.endswith(".py"):
            filename += ".py"
        # Get project root (6 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        self.result_path = project_root / "examples/generated_sketches" / filename
        self.dismiss(str(self.result_path))
    
    @on(Button.Pressed, "#cancel")
    def cancel_save(self):
        """Handle cancel button press."""
        self.dismiss(None)
    
    def action_dismiss(self) -> None:
        """Close the modal when ESC is pressed."""
        self.dismiss(None)
    
    def on_click(self, event: events.Click) -> None:
        """Close modal when clicking outside the dialog."""
        clicked, _ = self.get_widget_at(event.screen_x, event.screen_y)
        if clicked is self:
            self.dismiss(None)


class LoadDialog(ModalScreen):
    """Dialog for loading existing sketches."""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
    ]
    
    DEFAULT_CSS = """
    LoadDialog {
        align: center middle;
    }
    
    #load-dialog {
        width: 80;
        height: 30;
        border: thick $background 80%;
        background: $surface;
        padding: 1;
    }
    
    #file-tree {
        height: 20;
        border: solid $primary;
        margin: 1 0;
    }
    
    #selected-file {
        margin: 1 0;
        color: $text-muted;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.selected_file = None
    
    def compose(self) -> ComposeResult:
        with Container(id="load-dialog"):
            yield Label("Load Sketch", id="title")
            yield DirectoryTree(
                "examples/generated_sketches",
                id="file-tree"
            )
            yield Static("No file selected", id="selected-file")
            with Horizontal():
                yield Button("Load", variant="primary", id="load", disabled=True)
                yield Button("Cancel", variant="default", id="cancel")
    
    @on(DirectoryTree.FileSelected)
    def file_selected(self, event: DirectoryTree.FileSelected):
        """Handle file selection in the tree."""
        if event.path.suffix == ".py":
            self.selected_file = event.path
            self.query_one("#selected-file", Static).update(f"Selected: {event.path.name}")
            self.query_one("#load", Button).disabled = False
            # Auto-load on double-click/enter
            self.dismiss(str(self.selected_file))
    
    @on(Button.Pressed, "#load")
    def load_file(self):
        """Handle load button press."""
        if self.selected_file:
            self.dismiss(str(self.selected_file))
    
    @on(Button.Pressed, "#cancel")
    def cancel_load(self):
        """Handle cancel button press."""
        self.dismiss(None)
    
    def action_dismiss(self) -> None:
        """Close the modal when ESC is pressed."""
        self.dismiss(None)
    
    def on_click(self, event: events.Click) -> None:
        """Close modal when clicking outside the dialog."""
        clicked, _ = self.get_widget_at(event.screen_x, event.screen_y)
        if clicked is self:
            self.dismiss(None)


class ErrorDialog(ModalScreen):
    """Dialog for displaying errors with suggestions."""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
    ]
    
    DEFAULT_CSS = """
    ErrorDialog {
        align: center middle;
    }
    
    #error-dialog {
        width: 70;
        height: 25;
        border: thick $error 80%;
        background: $surface;
        padding: 1;
    }
    
    #error-title {
        color: $error;
        text-style: bold;
    }
    
    #error-content {
        height: 15;
        margin: 1 0;
        border: solid $error;
    }
    
    #fix-suggestion {
        margin: 1 0;
        color: $warning;
    }
    """
    
    def __init__(self, error_text: str, suggestion: str = None):
        super().__init__()
        self.error_text = error_text
        self.suggestion = suggestion or "Try checking the code for syntax errors or missing imports."
    
    def compose(self) -> ComposeResult:
        with Container(id="error-dialog"):
            yield Label("❌ Error Detected", id="error-title")
            yield TextArea(
                self.error_text,
                id="error-content",
                read_only=True
            )
            yield Static(f"💡 Suggestion: {self.suggestion}", id="fix-suggestion")
            with Horizontal():
                yield Button("Auto-Fix", variant="warning", id="autofix")
                yield Button("Close", variant="default", id="close")
    
    @on(Button.Pressed, "#autofix")
    def request_autofix(self):
        """Request automatic fix for the error."""
        self.dismiss("autofix")
    
    @on(Button.Pressed, "#close")
    def close_dialog(self):
        """Close the error dialog."""
        self.dismiss(None)
    
    def action_dismiss(self) -> None:
        """Close the modal when ESC is pressed."""
        self.dismiss(None)
    
    def on_click(self, event: events.Click) -> None:
        """Close modal when clicking outside the dialog."""
        clicked, _ = self.get_widget_at(event.screen_x, event.screen_y)
        if clicked is self:
            self.dismiss(None)


class HelpDialog(ModalScreen):
    """Dialog for displaying help information."""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
    ]
    
    DEFAULT_CSS = """
    HelpDialog {
        align: center middle;
    }
    
    #help-dialog {
        width: 70;
        height: 30;
        border: thick $primary 80%;
        background: $surface;
        padding: 1;
    }
    
    #help-content {
        height: 25;
        overflow-y: scroll;
        margin: 1 0;
    }
    """
    
    HELP_TEXT = """
# Tölvera Textual UI Help

## Quick Start
1. Enter a behavior description in the left panel
2. Click "Generate Sketch" to create the code
3. Click "Run Sketch" to see it in action
4. Use the chat panel for refinements

## Keyboard Shortcuts
- Ctrl+N: New sketch
- Ctrl+R: Run current sketch
- Ctrl+S: Save sketch
- Ctrl+Q: Quit application
- F1: Show this help

## Behavior Examples
- "particles fall with gravity"
- "red predators chase blue prey"
- "particles form a cellular automaton"
- "draw glowing trails behind particles"

## Refinement Examples
- "make gravity stronger"
- "particles move too fast"
- "change colors to blue"
- "add random drift"

## Tips
- The code editor supports Python syntax highlighting
- You can manually edit the generated code
- All sketches are saved with timestamps
- Check the trace panel for generation details
"""
    
    def compose(self) -> ComposeResult:
        with Container(id="help-dialog"):
            yield Label("📚 Help & Documentation", id="title")
            yield TextArea(
                self.HELP_TEXT,
                id="help-content",
                read_only=True,
                language="markdown"
            )
            yield Button("Close", variant="primary", id="close")
    
    @on(Button.Pressed, "#close")
    def close_help(self):
        """Close the help dialog."""
        self.dismiss()
    
    def action_dismiss(self) -> None:
        """Close the modal when ESC is pressed."""
        self.dismiss()
    
    def on_click(self, event: events.Click) -> None:
        """Close modal when clicking outside the dialog."""
        clicked, _ = self.get_widget_at(event.screen_x, event.screen_y)
        if clicked is self:
            self.dismiss()