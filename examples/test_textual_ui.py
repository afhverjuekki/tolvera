#!/usr/bin/env python3
"""
Test script to verify Textual UI can run with current dependencies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Button, Label
    from textual.containers import Container
    
    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Header()
            with Container():
                yield Label("Textual is working!")
                yield Button("Test Button")
            yield Footer()
    
    app = TestApp()
    print("✅ Textual imported successfully!")
    print("You can run the full UI with: python examples/tolvera_textual_ui.py")
    
except ImportError as e:
    print(f"❌ Textual not installed: {e}")
    print("\nTo install Textual manually:")
    print("pip install textual==0.89.1")
    print("\nNote: This may conflict with iipyper's requirements.")
    print("Consider creating a separate virtual environment for the UI.")