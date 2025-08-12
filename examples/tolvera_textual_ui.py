#!/usr/bin/env python3
"""
Tölvera Textual UI - Interactive sketch generator with Terminal User Interface
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Grid
from textual.widgets import (
    Button, Footer, Header, Input, Label, LoadingIndicator,
    Log, Static, TextArea, RadioSet, RadioButton
)
from textual.screen import Screen, ModalScreen
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tolvera import Tolvera
from tolvera.llm import BehaviorAgent
from tolvera.llm.core.sketch_refiner import SketchRefiner
from tolvera.llm.debug.tracing import get_collector
from tolvera.llm.debug.console_tracer import enable_console_tracing
from tolvera.llm.debug.trace_html_report import generate_html_report
from tolvera.llm.core.model_factory import ModelFactory
from dotenv import load_dotenv

# Import enhanced components
from textual_components import (
    SaveDialog, LoadDialog, ErrorDialog, HelpDialog,
    EnhancedCodeEditor, EnhancedChatPanel
)


from textual.containers import Grid


class ModelSelectorScreen(ModalScreen[str]):
    """Modal screen for selecting the LLM model."""
    
    DEFAULT_CSS = """
    /* Bioluminescent modal screen styling */
    ModelSelectorScreen {
        align: center middle;
        background: #000814 95%;  /* Deep ocean backdrop */
    }
    
    #model-selection-dialog {
        grid-size: 1;
        grid-rows: auto auto 1fr auto;
        width: 80;
        height: 25;
        border: thick #00D9FF 60%;  /* Electric cyan border */
        background: #000B1A 90%;
        padding: 2;
        /* Glow effect - box-shadow not supported in Textual CSS */
    }
    
    #title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: #00F5FF;  /* Bright aqua title */
        text-style: bold italic;
    }
    
    #api-status {
        margin-bottom: 1;
        text-align: center;
        color: #39FF14 80%;  /* Neon green for status */
    }
    
    #model-radio-set {
        height: 12;
        margin-bottom: 1;
        border: solid #00D9FF 50%;
        padding: 0 1;
        background: #000814 90%;
        overflow-y: auto;
    }
    
    #model-radio-set RadioButton {
        background: #000814 50%;
        color: #00F5FF 90%;
        padding: 0;
        margin: 0;
        border: none;
        height: 1;
        min-height: 1;
    }
    
    #model-radio-set RadioButton:hover {
        background: #00D9FF 20%;
        color: #00F5FF;
    }
    
    #model-radio-set RadioButton:focus {
        background: #7209B7 30%;
        border-left: solid #7209B7;
    }
    
    /* Selected radio button - much more prominent */
    #model-radio-set RadioButton.-selected {
        background: #39FF14 25%;
        border-left: thick #39FF14;
        color: #39FF14;
        text-style: bold;
    }
    
    #model-radio-set RadioButton.-selected:hover {
        background: #39FF14 30%;
        color: #39FF14;
        text-style: bold;
    }
    
    #model-radio-set RadioButton:disabled {
        color: #666666;
        opacity: 0.5;
    }
    
    #custom-model-input {
        margin-bottom: 1;
        width: 100%;
        height: 3;
        background: #001629;
        border: solid #7209B7 40%;
        color: #D4ADFC;
    }
    
    #custom-model-input:focus {
        border: solid #7209B7 80%;
        background: #0A0014;
    }
    
    #button-container {
        layout: horizontal;
        height: 4;
        align: center middle;
        margin-top: 1;
        padding: 0;
    }
    
    #button-container Button {
        margin: 0 2;
        min-width: 15;
        height: 4;
        padding: 0 2;
        text-align: center;
        background: #001629;
        border: solid #00D9FF 40%;
        color: #00F5FF 90%;
    }
    
    #button-container Button:hover {
        background: #00D9FF 25%;
        border: solid #00F5FF 80%;
        text-style: bold;
    }
    
    ModelSelectorScreen #button-container Button.primary {
        border: solid #39FF14 60%;
        color: #39FF14;
    }
    
    ModelSelectorScreen #button-container Button.primary:hover {
        background: #39FF14 25%;
        border: solid #39FF14;
        color: #39FF14;
    }
    """
    
    def compose(self) -> ComposeResult:
        # Get available providers and their status
        providers_info = ModelFactory.list_available_providers()
        
        yield Grid(
            Label("Select LLM Provider & Model", id="title"),
            Label("Checking provider status...", id="api-status"),
            self._get_radio_set_widget(providers_info),
            Container(
                Button("Confirm", variant="primary", id="confirm"),
                Button("Cancel", variant="default", id="cancel"),
                id="button-container"
            ),
            id="model-selection-dialog",
        )
    
    def _get_radio_set_widget(self, providers_info):
        """Get a RadioSet widget with available providers."""
        radio_buttons = []
        default_selected = False
        
        # Sort providers to show ready ones first
        sorted_providers = sorted(providers_info.items(), key=lambda x: (x[1]['status'] != 'ready', x[0]))
        
        for provider, info in sorted_providers:
            default_model = info.get('default_model', 'unknown')
            is_ready = info['status'] == 'ready'
            
            # Create label with status indicator
            if is_ready:
                label = f"{provider.capitalize()}: {default_model} ✅"
            else:
                label = f"{provider.capitalize()}: {default_model} ❌ (not configured)"
            
            # Store the full provider:model value
            value = f"{provider}:{default_model}"
            
            # Create valid ID (replace all invalid characters)
            valid_id = f"{provider}_{default_model}"
            for char in [':', '.', '-', '/', ' ']:
                valid_id = valid_id.replace(char, '_')
            
            # Select first ready provider (prefer Gemini)
            is_selected = False
            if is_ready and not default_selected:
                if provider == 'gemini' or not any(p == 'gemini' and providers_info[p]['status'] == 'ready' for p in providers_info):
                    is_selected = True
                    default_selected = True
            
            # Create radio button
            radio_button = RadioButton(label, value=is_selected, disabled=not is_ready, id=valid_id)
            radio_button._model_value = value
            radio_buttons.append(radio_button)
        
        # If no providers at all, add placeholder
        if not radio_buttons:
            radio_buttons.append(RadioButton("No providers available", value=True, disabled=True, id="none"))
        
        return RadioSet(*radio_buttons, id="model-radio-set")
    
    def on_mount(self):
        """Check API keys on mount."""
        self.check_providers()
        # Set focus to the radio set
        radio_set = self.query_one("#model-radio-set", RadioSet)
        radio_set.focus()
    
    def check_providers(self):
        """Check which providers are available."""
        status = self.query_one("#api-status", Label)
        providers_info = ModelFactory.list_available_providers()
        
        ready_providers = []
        not_configured = []
        
        for provider, info in providers_info.items():
            if info['status'] == 'ready':
                ready_providers.append(provider.capitalize())
            else:
                not_configured.append(provider.capitalize())
        
        status_text = ""
        if ready_providers:
            status_text = f"✅ Ready: {', '.join(ready_providers)}"
        if not_configured:
            if status_text:
                status_text += " | "
            status_text += f"❌ Not configured: {', '.join(not_configured)}"
        
        if not status_text:
            status_text = "⚠️ No providers configured - check your .env file"
        
        status.update(status_text)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "confirm":
            # Get the selected radio button
            radio_set = self.query_one("#model-radio-set", RadioSet)
            
            if radio_set.pressed_button and not radio_set.pressed_button.disabled:
                # Use stored model value if available, otherwise fall back to ID
                if hasattr(radio_set.pressed_button, '_model_value'):
                    selected_value = radio_set.pressed_button._model_value
                else:
                    selected_value = radio_set.pressed_button.id
                
                # Check if it's a valid selection (not the "none" placeholder)
                if selected_value != "none":
                    self.dismiss(selected_value)
                else:
                    self.dismiss(None)
            else:
                # No valid selection, use default
                self.dismiss("gemini-2.0-flash")
        else:
            # Cancel was pressed
            self.dismiss(None)



class TutorialScreen(ModalScreen):
    """Interactive tutorial walkthrough for new users."""
    
    DEFAULT_CSS = """
    TutorialScreen {
        align: center middle;
        background: #000814 95%;
    }
    
    #tutorial-container {
        width: 95;
        height: 45;
        border: double #00D9FF 80%;
        background: #000814 95%;
        padding: 2;
        align: center middle;
    }
    
    #tutorial-title {
        text-align: center;
        color: #00F5FF;
        text-style: bold;
        margin-bottom: 1;
        height: 2;
    }
    
    #step-indicator {
        text-align: center;
        color: #7209B7 80%;
        margin-bottom: 1;
        height: 1;
    }
    
    #tutorial-content {
        height: 24;
        padding: 1;
        background: #000814 90%;
        border: solid #00D9FF 40%;
        margin-bottom: 1;
        overflow-y: auto;
    }
    
    .tutorial-text {
        color: #00F5FF 90%;
        margin-bottom: 1;
        text-align: left;
        padding: 0 1;
    }
    
    #tutorial-buttons {
        layout: grid;
        grid-size: 4 1;
        grid-columns: 1fr 1fr 1fr 1fr;
        height: 10;
        width: 100%;
        padding: 1 0;
        margin-top: 1;
    }
    
    #tutorial-buttons Button {
        width: 100%;
        height: 3;
        margin: 0;
        background: #001629;
        border: solid #00D9FF 40%;
        color: #00F5FF 90%;
    }
    
    #tutorial-buttons Button:hover {
        background: #00D9FF 25%;
        border: solid #00F5FF 80%;
        text-style: bold;
    }
    
    #tutorial-buttons Button.primary {
        border: solid #39FF14 60%;
        color: #39FF14;
    }
    
    #tutorial-buttons Button.primary:hover {
        background: #39FF14 25%;
        border: solid #39FF14;
    }
    
    #tutorial-buttons Button:disabled {
        color: #666666;
        opacity: 0.5;
        border: solid #666666 30%;
    }
    """
    
    TUTORIAL_STEPS = [
        {
            "title": "🌟 Welcome to Tölvera",
            "content": """Welcome to the Tölvera Artificial Life Synthesis System!

This tutorial will guide you through creating, running, and refining digital life forms using natural language.

What you'll learn:
• Generate particle behaviors from descriptions
• Run and visualize artificial life simulations  
• Refine behaviors with natural language
• Use diff highlighting to see changes
• Navigate the interface efficiently

🦠 Tölvera converts your words into living, breathing particle systems that exhibit complex emergent behaviors like flocking, predator-prey dynamics, and cellular automata.

Let's start your journey into alife creation!"""
        },
        {
            "title": "Step 1: Generate Your First Sketch",
            "content": """Now we'll generate your first alife form!

The description is already set to:
"Two species, red and green, repel each other strongly."

This describes a simple but interesting behavior where particles of different colors push away from each other, creating dynamic boundaries and patterns.

Your task:
• Click the "Gen" button below for automatic generation
• OR close this tutorial and click "Generate Sketch" manually
• Watch the status log for synthesis progress
• See the generated Python code appear in the code editor

⚡ The synthesis process:
1. LLM analyzes your description
2. Generates expert behavior functions  
3. Creates complete runnable sketch
4. Applies syntax highlighting

This may take 10-20 seconds depending on your model."""
        },
        {
            "title": "Step 2: Run the Sketch",
            "content": """Great! Your sketch has been generated. Now let's bring it to life!

The code editor now contains a complete Python program that implements your artificial life system using Taichi for GPU acceleration.

Your task:
• Close this tutorial and click "Run Sketch"
• Watch the status log for execution messages
• A new window will open showing your particles in motion

What you'll see:
• Red and green particles repelling each other
• Dynamic boundaries forming between species
• Real-time physics simulation
• Emergent patterns and behaviors

💡 Tip: The simulation runs in a separate process, so you can continue using the UI while it's running.

Leave the sketch running for a moment to observe the behavior, then return here for the next step."""
        },
        {
            "title": "⏹Step 3: Stop the Simulation",
            "content": """Now let's learn how to stop the running simulation.

When a sketch is running, you'll see:
• "Run Sketch" button becomes disabled
• "Stop Sketch" button becomes enabled  
• Status messages showing the simulation is active

Your task:
• Close this tutorial and click "Stop Sketch"
• Watch the status log confirm termination
• Notice the buttons return to normal state

Process control:
• Simulations run in separate processes for stability
• Stopping is immediate and clean
• No data is lost from the code editor
• You can restart anytime

This clean process management ensures your UI stays responsive even with complex simulations running."""
        },
        {
            "title": "Step 4: Refine Your Creation",
            "content": """Refine your alife with natural language!

The refinement system lets you modify behaviors by simply describing what you want to change.

Your task:
We'll refine the sketch by changing species colors:
• Click the "🎨 Color" button below for automatic refinement
• OR close tutorial and type in Refinement Chat: "Let's change the species colors to be a rust orange and teal please."
• Press Enter to submit the refinement request

What happens:
1. Your request is sent to the LLM
2. The current code is analyzed
3. Targeted changes are made
4. New code replaces the old
5. Changes are highlighted automatically

This process takes 5-10 seconds as the AI carefully modifies your code while preserving the core behavior structure."""
        },
        {
            "title": "Step 5: View Code Changes",
            "content": """Excellent! Your sketch has been refined with new colors.

Notice several important changes:
• Code editor now shows the updated Python code
• A diff indicator appears showing changed lines
• The "Toggle Diff" button is now enabled
• Chat panel shows the AI's response about changes made

Explore the diff system:
• Click "Toggle Diff" to see highlighted changes
• Green highlighting shows modified lines  
• The diff indicator shows which lines changed
• Toggle back to normal view when done

💡 Pro tip: Use Ctrl+T to hide/show the chat panel for better code reading!

The diff system helps you understand exactly what the LLM changed."""
        },
        {
            "title": "🌈 Step 6: Test Your Refined Creation",
            "content": """Time to see your refined alife in action!

Your sketch now features rust orange and teal particles instead of red and green. The behavior remains the same (mutual repulsion) but with your custom aesthetic.

Your task:
• Close this tutorial and click "Run Sketch" again
• Observe the new color scheme in action
• Compare the behavior to what you saw before
• Stop the sketch when ready

Refinement power:
• Colors, behaviors, physics parameters
• Add new species or interactions  
• Modify visual effects and trails
• Combine multiple behavior types
• All through natural language!

This demonstrates the core workflow: Generate → Run → Refine → Run → Repeat until perfect."""
        },
        {
            "title": "Step 7: Additional Features",
            "content": """Let's explore the other powerful features available:

🔧 File Operations:
• "Save Sketch" - Export your creation with timestamp
• "Load Sketch" - Import previous creations
• "Reset" - Clear everything and start fresh

⚙️ System Controls:
• "Change Model" - Switch between AI providers (Gemini, Claude, GPT-4)
• "Toggle Diff" - Show/hide change highlighting
• "Hide Chat" - Maximize code editor space
• Ctrl+T - Quick chat panel toggle

📊 Debugging Tools:
• Status & Logs panel tracks all operations
• Trace Information shows generation details
• "View Report" - Detailed HTML analysis
• Real-time progress monitoring

Keyboard shortcuts are shown in the footer for quick access!"""
        },
        {
            "title": "🎓 Tutorial Complete!",
            "content": """Congratulations! You've mastered the Tölvera interface!

What you've learned:
✅ Generate artificial life from natural language
✅ Run and visualize particle simulations
✅ Refine behaviors with conversational AI  
✅ Use diff highlighting to track changes
✅ Navigate the interface efficiently
✅ Understand the core workflow

Next steps:
• Try more complex descriptions like "fish school and avoid sharks"
• Experiment with cellular automata: "Conway's Game of Life"
• Create predator-prey ecosystems
• Add visual effects like trails and glows
• Combine multiple behavior types

💡 Remember:
• F2 opens this tutorial anytime
• Ctrl+T toggles the chat panel
• The status log shows helpful progress info
• Each generation creates unique, living systems

Welcome to the world of alife creation! 🦠✨"""
        }
    ]
    
    def __init__(self, initial_step=0):
        super().__init__()
        self.current_step = initial_step
        self.total_steps = len(self.TUTORIAL_STEPS)
        self.app_reference = None  # Will be set when opened
        
    def compose(self) -> ComposeResult:
        with Container(id="tutorial-container"):
            yield Label("Tutorial Walkthrough", id="tutorial-title")
            yield Label(f"Step {self.current_step + 1} of {self.total_steps}", id="step-indicator")
            
            with ScrollableContainer(id="tutorial-content"):
                yield Static("", id="tutorial-text", classes="tutorial-text")
            
            with Container(id="tutorial-buttons"):
                yield Button("◀◀", id="prev-btn", disabled=True)
                yield Button("▶▶", id="next-btn", variant="primary")
                yield Button("Auto", id="auto-btn", disabled=True)
                yield Button("Close", id="close-btn")
                
    def on_mount(self):
        """Initialize tutorial content."""
        self.update_step_content()
        
    def update_step_content(self):
        """Update the content for the current step."""
        step_data = self.TUTORIAL_STEPS[self.current_step]
        
        # Update title and indicator
        title = self.query_one("#tutorial-title", Label)
        title.update(step_data["title"])
        
        indicator = self.query_one("#step-indicator", Label)
        indicator.update(f"Step {self.current_step + 1} of {self.total_steps}")
        
        # Update content
        content = self.query_one("#tutorial-text", Static)
        content.update(step_data["content"])
        
        # Update button states
        prev_btn = self.query_one("#prev-btn", Button)
        next_btn = self.query_one("#next-btn", Button)
        auto_btn = self.query_one("#auto-btn", Button)
        
        prev_btn.disabled = (self.current_step == 0)
        
        if self.current_step == self.total_steps - 1:
            next_btn.label = "Finish"
        else:
            next_btn.label = "▶▶"
        
        # Enable auto button for specific steps
        if self.current_step == 1:  # Generate sketch step
            auto_btn.label = "Gen"
            auto_btn.disabled = False
        elif self.current_step == 4:  # Refinement step
            auto_btn.label = "🎨 Color"
            auto_btn.disabled = False
        else:
            auto_btn.label = "Auto"
            auto_btn.disabled = True
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle tutorial navigation."""
        if event.button.id == "prev-btn":
            if self.current_step > 0:
                self.current_step -= 1
                self.update_step_content()
                
        elif event.button.id == "next-btn":
            if self.current_step < self.total_steps - 1:
                self.current_step += 1
                self.update_step_content()
            else:
                # Finish tutorial - return completed status and final step
                self.dismiss(("completed", self.current_step))
                
        elif event.button.id == "auto-btn":
            self.handle_auto_action()
            
        elif event.button.id == "close-btn":
            # Close tutorial - return current step for persistence
            self.dismiss(("closed", self.current_step))
    
    def handle_auto_action(self):
        """Handle auto action button press for current step."""
        if not self.app_reference:
            return
        
        if self.current_step == 1:  # Generate sketch step
            # Close the modal and trigger generation - save current step
            self.dismiss(("generate", self.current_step))
        elif self.current_step == 4:  # Refinement step
            # Close the modal and trigger refinement - save current step
            self.dismiss(("refine", self.current_step))


class WelcomeScreen(ModalScreen[bool]):
    """Welcome screen with artificial life animations."""
    
    DEFAULT_CSS = """
    WelcomeScreen {
        align: center middle;
        background: #000814;
    }
    
    #welcome-container {
        width: 100;
        height: 50;
        border: double #00D9FF 60%;
        background: #000814 90%;
        padding: 2;
        align: center middle;
        /* Animated glow effect - animations not supported in Textual CSS */
    }
    
    /* Keyframes animations not supported in Textual CSS */
    
    #title-section {
        height: 8;
        align: center middle;
        margin-bottom: 1;
        width: 100%;
    }
    
    .welcome-title {
        text-align: center;
        color: #00F5FF;
        text-style: bold;
        margin-bottom: 1;
        width: 100%;
        content-align: center middle;
    }
    
    .welcome-subtitle {
        text-align: center;
        color: #7209B7 80%;
        text-style: italic;
        width: 100%;
        content-align: center middle;
    }
    
    #animation-container {
        height: 25;
        border: solid #00D9FF 30%;
        background: #000814 95%;
        padding: 1;
        margin: 1;
    }
    
    #animation-display {
        width: 100%;
        height: 100%;
        color: #00F5FF;
        text-align: center;
        content-align: center middle;
        align: center middle;
    }
    
    #info-section {
        height: 8;
        margin: 1;
        padding: 1;
        align: center middle;
        width: 100%;
    }
    
    .info-text {
        text-align: center;
        color: #39FF14 70%;
        margin-bottom: 1;
        width: 100%;
        content-align: center middle;
    }
    
    .setup-text {
        text-align: center;
        color: #FFB700 90%;
        margin-bottom: 1;
        width: 100%;
        content-align: center middle;
        text-style: bold;
    }
    
    .continue-text {
        text-align: center;
        color: #FF6B35 80%;
        margin-top: 1;
        width: 100%;
        content-align: center middle;
        text-style: italic;
    }
    
    WelcomeScreen #continue-btn {
        dock: bottom;
        width: 30;
        height: 3;
        margin: 1;
        align: center middle;
        background: #001629;
        border: solid #39FF14 60%;
        color: #39FF14;
    }
    
    WelcomeScreen #continue-btn:hover {
        background: #39FF14 30%;
        border: solid #39FF14;
        text-style: bold;
        color: #39FF14;
    }
    
    /* Specific colors for different animation states */
    .life-cell {
        color: #00F5FF;
    }
    
    .dna-strand {
        color: #7209B7;
    }
    
    .organism {
        color: #FFB700;
    }
    
    .colony {
        color: #F72585;
    }
    """
    
    # Animation patterns
    ANIMATIONS = {
        "game_of_life": [
            # Glider pattern
            [
                "                                                ",
                "                    ◉                          ",
                "                      ◉                        ",
                "                  ◉ ◉ ◉                        ",
                "                                                ",
                "                                                ",
            ],
            [
                "                                                ",
                "                                                ",
                "                    ◉ ◉                        ",
                "                  ◉   ◉                        ",
                "                      ◉                        ",
                "                                                ",
            ],
            [
                "                                                ",
                "                      ◉                        ",
                "                    ◉                          ",
                "                  ◉   ◉                        ",
                "                      ◉                        ",
                "                                                ",
            ],
        ],
        "dna_helix": [
            [
                "       ╱◈━━━━━━━━◈╲                             ",
                "      ╱            ╲                            ",
                "     ◈              ◈                           ",
                "    ╱                ╲                          ",
                "   ◈━━━━━━━━━━━━━━━━━━◈                         ",
                "    ╲                 ╱                          ",
                "     ◈               ◈                           ",
                "      ╲             ╱                            ",
                "       ╲◈━━━━━━━━━◈╱                             ",
            ],
            [
                "        ◈━━━━━━━━◈                              ",
                "       ╱          ╲                             ",
                "      ◈            ◈                            ",
                "     ╱              ╲                           ",
                "    ◈                ◈                          ",
                "   ╱━━━━━━━━━━━━━━━━━╲                         ",
                "    ◈                ◈                          ",
                "     ╲              ╱                           ",
                "      ◈━━━━━━━━━━━━◈                            ",
            ],
            [
                "       ╲◈━━━━━━━━━◈╱                             ",
                "      ╲            ╱                            ",
                "     ◈              ◈                           ",
                "    ╲                ╱                          ",
                "   ◈━━━━━━━━━━━━━━━━━◈                         ",
                "    ╱                ╲                          ",
                "     ◈               ◈                           ",
                "      ╱             ╲                            ",
                "       ╱◈━━━━━━━━━◈╲                             ",
            ],
        ],
        "cellular_growth": [
            [
                "                    ·                          ",
                "                                                ",
                "                                                ",
                "                                                ",
                "                                                ",
            ],
            [
                "                    ·                          ",
                "                   ·◦·                         ",
                "                    ·                          ",
                "                                                ",
                "                                                ",
            ],
            [
                "                   ·◦·                         ",
                "                  ·◉◉·                         ",
                "                   ·◦·                         ",
                "                                                ",
                "                                                ",
            ],
            [
                "                  ·◦◉◦·                        ",
                "                 ·◉●●◉·                        ",
                "                  ·◦◉◦·                        ",
                "                    ·                          ",
                "                                                ",
            ],
            [
                "                 ·◦◉●◉◦·                       ",
                "                ·◉●⬤⬤●◉·                       ",
                "                 ·◦◉●◉◦·                       ",
                "                  ·◦◉◦·                        ",
                "                    ·                          ",
            ],
        ],
        "boids_flock": [
            [
                "     ▹          ▹     ▹                        ",
                "         ▹                  ▹                  ",
                "   ▹         ▹       ▹                         ",
                "       ▹         ▹       ▹                     ",
                "            ▹         ▹                        ",
            ],
            [
                "      ▸         ▸      ▸                       ",
                "         ▸                 ▸                   ",
                "    ▸        ▸        ▸                        ",
                "        ▸        ▸      ▸                      ",
                "           ▸          ▸                        ",
            ],
            [
                "       ▹        ▹       ▹                      ",
                "          ▹               ▹                    ",
                "     ▹       ▹         ▹                       ",
                "         ▹       ▹     ▹                       ",
                "          ▹           ▹                        ",
            ],
        ],
        "reaction_diffusion": [
            [
                "        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░         ",
                "        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░         ",
                "        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░         ",
                "        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░         ",
            ],
            [
                "        ░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░         ",
                "        ░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░         ",
                "        ░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░         ",
                "        ░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░         ",
            ],
            [
                "        ░░░░▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░░░░░░░░░         ",
                "        ░░░▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒░░░░░░░░░░         ",
                "        ░░░▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒░░░░░░░░░░         ",
                "        ░░░░▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░░░░░░░░░         ",
            ],
            [
                "        ░▒▒▓▓▓▓████████████▓▓▓▓▒▒░░░░░░░░░         ",
                "        ▒▒▓▓▓████████████████▓▓▓▒▒░░░░░░░░         ",
                "        ▒▒▓▓▓████████████████▓▓▓▒▒░░░░░░░░         ",
                "        ░▒▒▓▓▓▓████████████▓▓▓▓▒▒░░░░░░░░░         ",
            ],
        ],
    }
    
    def __init__(self):
        super().__init__()
        self.animation_index = 0
        self.frame_index = 0
        self.current_animation = "game_of_life"
        self.animation_timer = None
        
    def compose(self) -> ComposeResult:
        """Create the welcome screen layout."""
        with Vertical(id="welcome-container"):
            with Vertical(id="title-section"):
                yield Label("T Ö L V E R A", classes="welcome-title")
                yield Label("Artificial Life Synthesis System", classes="welcome-subtitle")
                yield Label("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", classes="welcome-subtitle")
            
            with Container(id="animation-container"):
                yield Static("", id="animation-display")
            
            with Vertical(id="info-section"):
                yield Label("🌿 Build ALife programs through natural language 💬", classes="info-text")
                yield Label("Before you begin, make sure you have the appropriate models setup in your .env file!", classes="setup-text")
                yield Label("Hit enter to continue...", classes="continue-text")
            
            yield Button("Continue to Genesis →", variant="primary", id="continue-btn")
    
    def on_mount(self):
        """Start animations when mounted."""
        self.start_animations()
        self.update_animation()
    
    def start_animations(self):
        """Start the animation timer."""
        self.animation_timer = self.set_interval(0.8, self.next_frame)  # Slower animation for better viewing
    
    def next_frame(self):
        """Advance to the next animation frame."""
        animations = list(self.ANIMATIONS.keys())
        current_frames = self.ANIMATIONS[self.current_animation]
        
        # Advance frame
        self.frame_index = (self.frame_index + 1) % len(current_frames)
        
        # Every full cycle, switch to next animation
        if self.frame_index == 0:
            self.animation_index = (self.animation_index + 1) % len(animations)
            self.current_animation = animations[self.animation_index]
        
        self.update_animation()
    
    def update_animation(self):
        """Update the animation display."""
        display = self.query_one("#animation-display", Static)
        
        # Get current frame
        current_frames = self.ANIMATIONS[self.current_animation]
        frame = current_frames[self.frame_index]
        
        # Add title for each animation
        titles = {
            "game_of_life": "◇ Conway's Game of Life ◇",
            "dna_helix": "◇ DNA Double Helix ◇",
            "cellular_growth": "◇ Cellular Mitosis ◇",
            "boids_flock": "◇ Emergent Flocking ◇",
            "reaction_diffusion": "◇ Turing Patterns ◇",
        }
        
        # Combine title and animation with better centering
        title = titles.get(self.current_animation, "")
        # Process each line: strip whitespace, then center properly
        centered_frame = []
        for line in frame:
            # First strip all leading/trailing whitespace
            stripped_line = line.strip()
            # Then center the actual content within a reasonable width
            if stripped_line:  # Only center if there's content
                centered_line = stripped_line.center(5)
            else:
                centered_line = "".center(5)  # Empty centered line
            centered_frame.append(centered_line)
        
        # Add more vertical padding for better centering
        animation_text = f"\n\n{title}\n\n" + "\n".join(centered_frame) + "\n\n"
        
        # Add some dynamic elements based on animation type
        if self.current_animation == "game_of_life":
            animation_text = animation_text.replace("◉", f"[#00F5FF]◉[/]")
        elif self.current_animation == "dna_helix":
            animation_text = animation_text.replace("◈", f"[#7209B7]◈[/]")
            animation_text = animation_text.replace("━", f"[#F72585]━[/]")
        elif self.current_animation == "cellular_growth":
            animation_text = animation_text.replace("●", f"[#FFB700]●[/]")
            animation_text = animation_text.replace("⬤", f"[#F72585]⬤[/]")
            animation_text = animation_text.replace("◉", f"[#00F5FF]◉[/]")
        elif self.current_animation == "boids_flock":
            animation_text = animation_text.replace("▹", f"[#00D9FF]▹[/]")
            animation_text = animation_text.replace("▸", f"[#39FF14]▸[/]")
        elif self.current_animation == "reaction_diffusion":
            animation_text = animation_text.replace("█", f"[#F72585]█[/]")
            animation_text = animation_text.replace("▓", f"[#7209B7]▓[/]")
            animation_text = animation_text.replace("▒", f"[#00D9FF]▒[/]")
            animation_text = animation_text.replace("░", f"[#001629]░[/]")
        
        display.update(animation_text)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle continue button press."""
        if event.button.id == "continue-btn":
            # Stop animation timer
            if self.animation_timer:
                self.animation_timer.stop()
            self.dismiss(True)


class TolveraTextualUI(App):
    """Main Textual UI application for Tölvera sketch generation."""
    
    CSS = """
    /* Bioluminescent color palette inspired by deep sea creatures */
    TolveraTextualUI {
        background: #000814;  /* Deep ocean black */
    }
    
    #sketch-input-container {
        height: 11;
        border: solid #00D9FF 80%;  /* Electric cyan - bioluminescent jellyfish */
        padding: 1;
        margin-bottom: 1;
        background: #000814 95%;
    }
    
    #description-input {
        height: 5;
        width: 85%;
        background: #001629;
        border: solid #00D9FF 30%;
    }
    
    #generate-btn {
        width: 15%;
        height: 5;
        margin-left: 1;
        background: #00D9FF 20%;
        border: solid #00D9FF;
    }
    
    #generate-btn:hover {
        background: #00D9FF 40%;
        text-style: bold;
    }
    
    #main-grid {
        layout: grid;
        grid-size: 3 2;
        grid-rows: 3fr 1fr;
        grid-columns: 1fr 2fr 1fr;
        height: 1fr;
    }
    
    /* When chat is collapsed, expand code editor to span chat's column */
    #main-grid.chat-collapsed #code-editor-container {
        column-span: 2;
    }
    
    #control-panel {
        border: solid #FFB700 70%;  /* Bioluminescent gold - firefly squid */
        padding: 1;
        layout: vertical;
        background: #000814 90%;
    }
    
    #code-editor-container {
        border: solid #00F5FF 60%;  /* Neon aqua - deep sea fish */
        padding: 1;
        height: 100%;
        background: #000B1A;
    }
    
    #code-editor {
        height: 100%;
        /* Dracula theme colors will be applied automatically */
    }
    
    /* Enhanced styles for diff mode */
    EnhancedCodeEditor.diff-mode {
        border: thick #39FF14 !important;  /* Thick bright green border in diff mode */
        box-sizing: border-box;
    }
    
    .diff-indicator {
        dock: right;
        padding: 0 1;
        color: #39FF14;
        text-style: bold italic;
        width: auto;
        height: 1;
    }
    
    #diff-btn {
        border: solid #39FF14 40%;
        color: #39FF14 70%;
    }
    
    #diff-btn:hover {
        background: #39FF14 20%;
        border: solid #39FF14 60%;
        color: #39FF14;
    }
    
    #diff-btn:disabled {
        border: solid #39FF14 20%;
        color: #39FF14 40%;
        opacity: 0.5;
    }
    
    #chat-panel {
        border: solid #7209B7 60%;  /* Deep purple - bioluminescent coral */
        padding: 1;
        background: #0A0014 95%;
    }
    
    /* Hide chat panel when parent grid is collapsed */
    #main-grid.chat-collapsed #chat-panel {
        display: none;
    }
    
    #trace-info {
        border: solid #F72585 60%;  /* Neon pink - deep sea jellyfish */
        padding: 1;
        background: #140008 95%;
    }
    
    #status-log {
        border: solid #39FF14 50%;  /* Neon green - plankton bloom */
        padding: 1;
        column-span: 2;
        height: 100%;
        background: #001405 95%;
    }
    
    #log-output {
        height: 100%;
        max-height: 8;
        color: #00F5FF 80%;  /* Aqua text for logs */
    }
    
    .panel-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: #00D9FF;  /* Bright cyan titles */
        text-style: bold italic;
    }
    
    /* Global loading overlay - full screen modal */
    #loading-overlay {
        display: none;
        dock: top;
        layer: overlay;
        width: 100%;
        height: 100%;
        background: #000814 90%;
        align: center middle;
    }
    
    #loading-overlay.visible {
        display: block;
    }
    
    #loading-content {
        width: 50;
        height: 12;
        background: #000B1A 95%;
        border: double #00D9FF 80%;
        padding: 2;
        align: center middle;
        layout: vertical;
    }
    
    #loading-content LoadingIndicator {
        background: transparent;
        color: #00F5FF;
        margin-bottom: 2;
    }
    
    #loading-text {
        text-align: center;
        color: #00F5FF;
        text-style: bold;
    }
    
    Button {
        margin: 1 0;
        width: 100%;
        height: 3;
        background: #001629;
        border: solid #00D9FF 40%;
        color: #00F5FF 90%;
    }
    
    Button:hover {
        background: #00D9FF 20%;
        border: solid #00D9FF 80%;
        text-style: bold;
    }
    
    Button:focus {
        background: #00D9FF 30%;
        border: solid #00F5FF;
    }
    
    ModelSelectorScreen #button-container Button.primary:focus {
        background: #39FF14 25%;
        border: solid #39FF14;
        color: #39FF14;
    }
    
    #control-panel Button {
        height: 3;
        margin: 1 0;
    }
    
    #control-panel Button.warning {
        border: solid #FFB700 60%;
        color: #FFB700;
    }
    
    #control-panel Button.error {
        border: solid #F72585 60%;
        color: #F72585;
    }
    
    #control-panel Button.success {
        border: solid #39FF14 60%;
        color: #39FF14;
    }
    
    #chat-history {
        height: 1fr;
        margin-bottom: 1;
        background: #0A0014 50%;
        padding: 1;
    }
    
    .chat-message {
        padding: 1;
        margin-bottom: 1;
        width: 100%;
        min-height: 3;
    }
    
    .user-message {
        background: #7209B7 10%;
        border-left: thick #7209B7 60%;
        color: #D4ADFC;
    }
    
    .agent-message {
        background: #FF6B35 10%;
        border-left: thick #FF6B35 60%;
        color: #FFB570;
    }
    
    #refinement-input {
        dock: bottom;
        height: 3;
        background: #0A0014;
        border: solid #7209B7 40%;
        color: #D4ADFC;
    }
    
    #refinement-input:focus {
        border: solid #7209B7 80%;
    }
    
    /* Animate borders for organic feel - animations not supported in Textual CSS */
    
    Header {
        background: #00D9FF 20%;
        color: #00F5FF;
    }
    
    Footer {
        background: #000814;
        color: #00D9FF 70%;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+n", "new_sketch", "New Sketch", priority=True),
        Binding("ctrl+r", "run_sketch", "Run Sketch", priority=True),
        Binding("ctrl+s", "save_sketch", "Save Sketch", priority=True),
        Binding("ctrl+t", "toggle_chat", "Toggle Chat", priority=True),
        Binding("f2", "show_tutorial", "Tutorial", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("f1", "show_help", "Help"),
    ]
    
    # Reactive properties
    current_sketch_path = reactive(None)
    is_generating = reactive(False)
    is_running = reactive(False)
    is_initializing = reactive(False)
    agents_ready = reactive(False)
    model_name = reactive("gemini-2.0-flash")
    chat_visible = reactive(True)
    
    def __init__(self):
        super().__init__()
        self.behavior_agent = None
        self.sketch_refiner = None
        self.collector = None
        self.main_trace = None
        self.sketch_process = None
        self.chat_history = []
        self.current_sketch_code = ""
        self.tv = None
        self.tutorial_current_step = 0  # Persist tutorial state
        self.tutorial_completed = False  # Track if tutorial was completed
        
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(show_clock=True)
        
        # Sketch input at the top, spanning full width
        with Vertical(id="sketch-input-container"):
            yield Label("Sketch Description", classes="panel-title")
            with Horizontal():
                yield TextArea(
                    "Two species, red and green, repel each other strongly.",
                    id="description-input",
                    language=None
                )
                yield Button("Generate Sketch", variant="primary", id="generate-btn", disabled=True)
        
        # Main grid layout
        with Container(id="main-grid"):
            # Top row - Controls (left), Code Editor (center), Chat (right)
            with Vertical(id="control-panel"):
                yield Label("Controls", classes="panel-title")
                yield Button("Run Sketch", variant="success", id="run-btn")
                yield Button("Stop Sketch", variant="error", id="stop-btn", disabled=True)
                yield Button("Save Sketch", variant="primary", id="save-btn")
                yield Button("Load Sketch", variant="default", id="load-btn")
                yield Button("Reset", variant="warning", id="reset-btn")
                yield Button("Toggle Diff", variant="default", id="diff-btn", disabled=True)
                yield Button("Hide Chat", variant="default", id="chat-toggle-btn")
                yield Button("Change Model", variant="default", id="model-btn")
            
            with Vertical(id="code-editor-container"):
                with Horizontal():
                    yield Label("Generated Code", classes="panel-title")
                    yield Static("", id="diff-indicator", classes="diff-indicator")
                # Create enhanced code editor with diff highlighting support
                yield EnhancedCodeEditor(
                    "# Generated code will appear here\n# Python syntax highlighting is enabled",
                    id="code-editor",
                    language="python",
                    theme="dracula",
                    show_line_numbers=True,
                    tab_behavior="indent",
                    read_only=False
                )
            
            with Vertical(id="chat-panel"):
                yield Label("Refinement Chat", classes="panel-title")
                yield ScrollableContainer(id="chat-history")
                yield Input(
                    placeholder="Enter refinement request...",
                    id="refinement-input"
                )
            
            # Bottom row - Trace (left), Status spanning the rest
            with Vertical(id="trace-info"):
                yield Label("Trace Information", classes="panel-title")
                yield Static("No trace data yet", id="trace-display")
                yield Button("View Report", variant="primary", id="report-btn", disabled=True)
            
            with Vertical(id="status-log"):
                yield Label("Status & Logs", classes="panel-title")
                yield Log(id="log-output", highlight=True)
        
        yield Footer()
        
        # Global loading overlay - appears on top of everything when visible
        with Container(id="loading-overlay"):
            with Container(id="loading-content"):
                yield LoadingIndicator()
                yield Label("Processing...", id="loading-text")
    
    def on_mount(self):
        """Initialize the application when mounted."""
        # Load environment variables
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            self.log_message(f"🧬 Loaded environment from: {env_path}")
        
        # Check for available providers
        providers_info = ModelFactory.list_available_providers()
        ready_providers = [p for p, info in providers_info.items() if info['status'] == 'ready']
        
        if ready_providers:
            self.log_message(f"🔑 Available providers: {', '.join(ready_providers)}")
        else:
            self.log_message("⚠️ No providers configured - please check your .env file")
        
        # Setup tracing
        enable_console_tracing(colored=False)  # Disable console colors in TUI
        self.collector = get_collector()
        self.collector.enabled = True
        self.collector.capture_llm_content = True
        
        # Initialize directories
        Path("examples/generated_sketches").mkdir(parents=True, exist_ok=True)
        Path("examples/generated_sketches/traces").mkdir(parents=True, exist_ok=True)
        
        # Show welcome screen first
        self.set_timer(0.1, self.show_welcome_screen)
    
    def show_welcome_screen(self):
        """Show the welcome screen with animations."""
        self.push_screen(WelcomeScreen(), self.handle_welcome_screen)
    
    def handle_welcome_screen(self, continued: bool | None) -> None:
        """Handle the welcome screen dismissal."""
        if continued:
            self.log_message("🌟 Welcome sequence complete")
            # Now show the model selector
            self.set_timer(0.001, self.show_model_selector_delayed)
    
    def show_model_selector_delayed(self):
        """Show model selector after a delay to ensure UI is ready."""
        self.log_message("Opening model selection dialog...")
        self.push_screen(ModelSelectorScreen(), self.handle_model_selection)
    
    def handle_model_selection(self, model: str | None) -> None:
        """Handle the model selection from the modal."""
        if model:
            self.model_name = model
            # Parse the model to show provider info
            provider, actual_model = ModelFactory.parse_model_string(model)
            self.log_message(f"Selected provider: {provider}, model: {actual_model}")
            self.log_message("Starting agent initialization...")
            
            # Show loading indicator
            try:
                loading = self.query_one("#loading-overlay", Container)
                loading.add_class("visible")
                loading_text = self.query_one("#loading-text", Label)
                loading_text.update("Initializing agents... This may take 10-30 seconds on first run.")
            except:
                pass
            
            # Disable generate button during initialization
            try:
                generate_btn = self.query_one("#generate-btn", Button)
                generate_btn.disabled = True
            except:
                pass
            
            # Start the initialization worker
            self.initialize_agents()
        else:
            self.log_message("Model selection cancelled - using default")
            self.model_name = "gemini-2.0-flash"
            
            # Show loading indicator
            try:
                loading = self.query_one("#loading-overlay", Container)
                loading.add_class("visible")
                loading_text = self.query_one("#loading-text", Label)
                loading_text.update("Initializing agents with default model...")
            except:
                pass
            
            self.initialize_agents()
    
    def log_message(self, message: str):
        """Add a message to the log."""
        try:
            log = self.query_one("#log-output", Log)
            timestamp = datetime.now().strftime('%H:%M:%S')
            log.write_line(f"[{timestamp}] {message}")
        except Exception:
            # Log not available yet (during initialization)
            pass
    
    @work(exclusive=True)
    async def initialize_agents(self):
        """Initialize the behavior agent and refiner asynchronously."""
        import asyncio
        
        try:
            self.is_initializing = True
            self.agents_ready = False
            
            self.log_message("🦠 Spawning artificial life agents...")
            self.log_message("⏳ Incubating neural pathways... (10-30 seconds)")
            
            # Small delay to let UI update
            await asyncio.sleep(0.1)
            
            # Initialize Tölvera - this is fast
            self.log_message("  🌊 Creating digital ocean environment...")
            try:
                self.tv = Tolvera(width=1920, height=1080, pn=500, sn=4)
                self.log_message("  ✨ Tölvera ecosystem initialized")
            except Exception as e:
                self.log_message(f"  ❌ Tölvera failed: {e}")
                raise
            
            # Small delay to let UI update
            await asyncio.sleep(0.1)
            
            # Initialize behavior agent - THIS IS THE SLOW PART
            provider, actual_model = ModelFactory.parse_model_string(self.model_name)
            self.log_message(f"  🐙 Cultivating BehaviorAgent with {provider} provider...")
            self.log_message(f"  Model: {actual_model}")
            self.log_message("  🧪 Synthesizing neural networks...")
            
            try:
                self.behavior_agent = BehaviorAgent(self.tv, model_name=self.model_name)
                self.log_message("  🦋 BehaviorAgent emerged successfully")
                
            except Exception as e:
                self.log_message(f"  ❌ BehaviorAgent failed: {e}")
                raise
            
            # Small delay to let UI update
            await asyncio.sleep(0.1)
            
            # Initialize sketch refiner - also potentially slow
            self.log_message(f"  🪸 Growing SketchRefiner with {provider}...")
            try:
                self.sketch_refiner = SketchRefiner(model_name=self.model_name)
                self.log_message(f"  🌺 SketchRefiner bloomed with {provider}")
            except Exception as e:
                self.log_message(f"  ❌ SketchRefiner failed: {e}")
                raise
            
            # Mark as ready
            self.agents_ready = True
            self.log_message("🌟 Digital ecosystem ready! Begin creating artificial life...")
            
            # Enable the generate button
            try:
                generate_btn = self.query_one("#generate-btn", Button)
                generate_btn.disabled = False
                # Also update loading indicator if visible
                try:
                    loading = self.query_one("#loading-overlay", Container)
                    loading.remove_class("visible")
                except:
                    pass
            except Exception as e:
                self.log_message(f"⚠️ Could not enable generate button: {e}")
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Failed to initialize agents: {str(e)}"
            self.log_message(error_msg)
            traceback_str = traceback.format_exc()
            for line in traceback_str.split('\n'):
                if line.strip():
                    self.log_message(f"  {line}")
            self.agents_ready = False
            
            self.log_message("💡 Try selecting a different model or check your API keys")
            
        finally:
            self.is_initializing = False
            self.log_message("🧬 Genesis complete")
    
    @on(Button.Pressed, "#generate-btn")
    def generate_sketch(self):
        """Generate a new sketch from the description."""
        if self.is_initializing:
            self.log_message("⏳ Agents are still initializing. Please wait...")
            return
            
        if not self.agents_ready or not self.behavior_agent:
            self.log_message("⚠️ Agents not initialized. Please wait or try selecting a model again.")
            # Show model selector again
            self.push_screen(ModelSelectorScreen(), self.handle_model_selection)
            return
        
        self.log_message("🦠 Initiating artificial life synthesis...")
        self.generate_sketch_worker()
    
    @work(exclusive=True)
    async def generate_sketch_worker(self):
        """Async worker for sketch generation."""
        import asyncio
        
        try:
            # Get description
            description_input = self.query_one("#description-input", TextArea)
            description = description_input.text.strip()
            
            if not description:
                self.log_message("⚠️ Please enter a behavior description")
                return
            
            # Show loading indicator
            loading = self.query_one("#loading-overlay", Container)
            loading.add_class("visible")
            loading_text = self.query_one("#loading-text", Label)
            loading_text.update("Generating artificial life sketch...")
            self.is_generating = True
            
            self.log_message(f"🧫 Cultivating behaviors: {description}")
            
            # Start trace
            self.main_trace = self.collector.start_trace("Textual UI Generation", "ui")
            
            # Add behavior (this is the async operation)
            result = await self.behavior_agent.add_behavior(description, weight=1.0)
            self.log_message(f"🐠 Behaviors evolved: {result['experts_added']} expert organisms")
            
            # Generate sketch
            _, sketch_path = self.behavior_agent.generate_sketch(
                description="Generated via Textual UI",
                filename="textual_sketch",
                use_timestamp=True,
                validate=False
            )
            
            self.current_sketch_path = sketch_path
            self.log_message(f"🌊 Life form preserved at: {sketch_path}")
            
            # Load code into editor
            with open(sketch_path, 'r') as f:
                code = f.read()
            
            self.current_sketch_code = code
            
            code_editor = self.query_one("#code-editor", EnhancedCodeEditor)
            # Ensure language is set before loading to guarantee highlighting.
            code_editor.language = "python"
            # Use load_text to ensure highlighting is applied
            code_editor.load_text(code)
            # Clear any previous diff state since this is a new generation
            code_editor.clear_diff_highlighting()
            
            # Complete trace
            self.main_trace.complete("success")
            self.update_trace_info()
            
        except Exception as e:
            self.log_message(f"❌ Generation failed: {e}")
            if self.main_trace:
                self.main_trace.complete("failed")
        finally:
            # Hide loading indicator
            loading = self.query_one("#loading-overlay", Container)
            loading.remove_class("visible")
            self.is_generating = False
    
    @on(Button.Pressed, "#run-btn")
    def run_sketch(self):
        """Run the current sketch."""
        self.run_sketch_async()
    
    @work(exclusive=True)
    async def run_sketch_async(self):
        """Async worker for running sketches."""
        if not self.current_sketch_path:
            self.log_message("⚠️ No sketch to run. Generate one first.")
            return
        
        try:
            # Save current code first
            code_editor = self.query_one("#code-editor", EnhancedCodeEditor)
            with open(self.current_sketch_path, 'w') as f:
                f.write(code_editor.text)
            
            self.log_message(f"🌀 Animating life form: {self.current_sketch_path}")
            
            # Disable run button, enable stop button
            self.query_one("#run-btn", Button).disabled = True
            self.query_one("#stop-btn", Button).disabled = False
            self.is_running = True
            
            # Create subprocess
            self.sketch_process = await asyncio.create_subprocess_exec(
                sys.executable, self.current_sketch_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Read output asynchronously
            async def read_stream(stream, prefix):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line = line.decode('utf-8', errors='replace').strip()
                    if line:
                        self.log_message(f"{prefix}: {line}")
            
            # Create tasks for reading both streams
            stdout_task = asyncio.create_task(read_stream(self.sketch_process.stdout, "OUT"))
            stderr_task = asyncio.create_task(read_stream(self.sketch_process.stderr, "ERR"))
            
            # Wait for process to complete
            await asyncio.gather(stdout_task, stderr_task)
            await self.sketch_process.wait()
            
            self.log_message("🦋 Life cycle completed")
            
        except Exception as e:
            self.log_message(f"❌ Failed to run sketch: {e}")
        finally:
            self.is_running = False
            self.query_one("#run-btn", Button).disabled = False
            self.query_one("#stop-btn", Button).disabled = True
    
    @on(Button.Pressed, "#stop-btn")
    async def stop_sketch(self):
        """Stop the running sketch."""
        if self.sketch_process:
            self.sketch_process.terminate()
            await asyncio.sleep(0.5)
            if self.sketch_process.returncode is None:
                self.sketch_process.kill()
            self.log_message("💤 Life form hibernated")
            self.is_running = False
            self.query_one("#run-btn", Button).disabled = False
            self.query_one("#stop-btn", Button).disabled = True
    
    @on(Input.Submitted, "#refinement-input")
    async def handle_refinement(self, event: Input.Submitted):
        """Handle refinement request."""
        request = event.value.strip()
        if not request:
            return
        
        if not self.sketch_refiner or not self.current_sketch_code:
            self.log_message("⚠️ No sketch to refine. Generate one first.")
            return
        
        # Clear input
        event.input.value = ""
        
        # Add to chat history with proper styling and wrapping
        chat_container = self.query_one("#chat-history", ScrollableContainer)
        user_message = Static(f"You: {request}", classes="chat-message user-message")
        chat_container.mount(user_message)
        chat_container.scroll_end(animate=False)
        
        # Start refinement worker
        self.apply_refinement_worker(request)
    
    @work(exclusive=True)
    async def apply_refinement_worker(self, request: str):
        """Apply refinement to the current sketch asynchronously."""
        try:
            self.log_message(f"🧬 Evolving behaviors: {request}")
            
            # Get current code and store it for diff
            code_editor = self.query_one("#code-editor", EnhancedCodeEditor)
            current_code = code_editor.text
            
            # Store the pre-refinement code for diff highlighting
            code_editor.store_pre_refinement_code()
            
            # Apply refinement (async operation)
            result = await self.sketch_refiner.refine_sketch(
                current_code,
                request
            )
            
            if result['success']:
                self.current_sketch_code = result['refined_code']
                
                # Update code editor with refined code
                code_editor = self.query_one("#code-editor", EnhancedCodeEditor)
                # Ensure language is set before loading to guarantee highlighting.
                code_editor.language = "python"
                # Use load_text to ensure highlighting is applied
                code_editor.load_text(result['refined_code'])
                
                # Apply diff highlighting to show changes
                code_editor.apply_diff_highlighting(result['refined_code'])
                
                # Enable the diff toggle button
                diff_btn = self.query_one("#diff-btn", Button)
                diff_btn.disabled = False
                
                # Update the diff indicator with initial status
                diff_indicator = self.query_one("#diff-indicator", Static)
                num_changes = len(code_editor.diff_lines)
                if num_changes > 0:
                    # Convert 0-based to 1-based line numbers for display
                    line_nums = sorted([n + 1 for n in code_editor.diff_lines])
                    
                    # Format line numbers nicely
                    if len(line_nums) <= 5:
                        lines_str = ", ".join(str(n) for n in line_nums)
                    else:
                        # Show first few and last with ellipsis
                        lines_str = f"{line_nums[0]}-{line_nums[-1]} ({num_changes} lines)"
                    
                    diff_indicator.update(f"🟢 Lines changed: {lines_str}")
                
                # Add response to chat with diff summary
                chat_container = self.query_one("#chat-history", ScrollableContainer)
                diff_summary = code_editor.get_diff_summary()
                agent_message = Static(f"✓ Agent: {result['changes_made']} ({diff_summary})", classes="chat-message agent-message")
                chat_container.mount(agent_message)
                chat_container.scroll_end(animate=False)
                
                self.log_message(f"🦠 Evolution successful: {result['changes_made']} - {diff_summary}")
                
                # Save updated code
                if self.current_sketch_path:
                    with open(self.current_sketch_path, 'w') as f:
                        f.write(result['refined_code'])
            else:
                error_msg = f"❌ Refinement failed: {result.get('error', 'Unknown error')}"
                self.log_message(error_msg)
                
        except Exception as e:
            self.log_message(f"❌ Refinement error: {e}")
    
    @on(Button.Pressed, "#save-btn")
    def save_sketch(self):
        """Save the current sketch to a new file."""
        if not self.current_sketch_code:
            self.log_message("⚠️ No sketch to save")
            return
        
        # Show save dialog
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.push_screen(SaveDialog(f"sketch_{timestamp}"), self.handle_save_dialog)
    
    def handle_save_dialog(self, save_path: str | None) -> None:
        """Handle the save dialog result."""
        if save_path:
            try:
                code_editor = self.query_one("#code-editor", EnhancedCodeEditor)
                with open(save_path, 'w') as f:
                    f.write(code_editor.text)
                self.current_sketch_path = save_path
                self.log_message(f"🧊 Life form crystallized at: {save_path}")
            except Exception as e:
                self.log_message(f"❌ Failed to save sketch: {e}")
    
    @on(Button.Pressed, "#load-btn")
    def load_sketch(self):
        """Load an existing sketch file."""
        self.push_screen(LoadDialog(), self.handle_load_dialog)
    
    def handle_load_dialog(self, load_path: str | None) -> None:
        """Handle the load dialog result."""
        if load_path:
            try:
                with open(load_path, 'r') as f:
                    code = f.read()
                
                code_editor = self.query_one("#code-editor", EnhancedCodeEditor)
                # Ensure language is set before loading to guarantee highlighting.
                code_editor.language = "python"
                # Use load_text to ensure highlighting is applied
                code_editor.load_text(code)
                # Clear any previous diff state
                code_editor.clear_diff_highlighting()
                # Disable diff button since this is a loaded file
                diff_btn = self.query_one("#diff-btn", Button)
                diff_btn.disabled = True
                # Clear diff indicator
                diff_indicator = self.query_one("#diff-indicator", Static)
                diff_indicator.update("")

                self.current_sketch_code = code
                self.current_sketch_path = load_path
                
                self.log_message(f"🌱 Life form revived from: {load_path}")
            except Exception as e:
                self.log_message(f"❌ Failed to load sketch: {e}")
    
    @on(Button.Pressed, "#reset-btn")
    async def reset_ui(self):
        """Reset the UI to initial state."""
        # Clear inputs
        self.query_one("#description-input", TextArea).text = ""
        code_editor = self.query_one("#code-editor", EnhancedCodeEditor)
        code_editor.load_text("# Generated code will appear here\n# Python syntax highlighting is enabled")
        code_editor.clear_diff_highlighting()
        
        # Disable diff button and clear indicator
        diff_btn = self.query_one("#diff-btn", Button)
        diff_btn.disabled = True
        diff_indicator = self.query_one("#diff-indicator", Static)
        diff_indicator.update("")
        
        # Clear chat
        chat_container = self.query_one("#chat-history", ScrollableContainer)
        chat_container.remove_children()
        
        # Clear logs
        self.query_one("#log-output", Log).clear()
        
        # Reset state
        self.current_sketch_path = None
        self.current_sketch_code = ""
        self.chat_history = []
        
        self.log_message("🌊 Digital ocean cleared")
    
    @on(Button.Pressed, "#diff-btn")
    def toggle_diff_view(self):
        """Toggle the diff highlighting view."""
        code_editor = self.query_one("#code-editor", EnhancedCodeEditor)
        new_state = code_editor.toggle_diff_highlighting()
        
        # Update the diff indicator
        diff_indicator = self.query_one("#diff-indicator", Static)
        
        if new_state:
            # Show diff summary with line numbers in the indicator
            num_changes = len(code_editor.diff_lines)
            if num_changes > 0:
                # Convert 0-based to 1-based line numbers for display
                line_nums = sorted([n + 1 for n in code_editor.diff_lines])
                
                # Format line numbers nicely
                if len(line_nums) <= 5:
                    lines_str = ", ".join(str(n) for n in line_nums)
                else:
                    # Show first few and last with ellipsis
                    lines_str = f"{line_nums[0]}-{line_nums[-1]} ({num_changes} lines)"
                
                diff_indicator.update(f"🟢 Lines changed: {lines_str}")
            self.log_message("🔍 Diff highlighting enabled - showing changes from refinement")
        else:
            diff_indicator.update("")
            self.log_message("📝 Diff highlighting disabled - normal view")
    
    @on(Button.Pressed, "#chat-toggle-btn")
    def toggle_chat_panel(self):
        """Toggle the refinement chat panel horizontally."""
        try:
            # Toggle the reactive property
            self.chat_visible = not self.chat_visible
            
            # Update the grid layout
            main_grid = self.query_one("#main-grid", Container)
            btn = self.query_one("#chat-toggle-btn", Button)
            
            if not self.chat_visible:
                # Hide chat, expand code editor
                main_grid.add_class("chat-collapsed")
                btn.label = "Show Chat"
                self.log_message("📱 Chat panel collapsed horizontally - code editor expanded")
            else:
                # Show chat, normal layout
                main_grid.remove_class("chat-collapsed")
                btn.label = "Hide Chat"
                self.log_message("💬 Chat panel expanded - normal layout restored")
                
        except Exception as e:
            self.log_message(f"Error toggling chat: {e}")
    
    @on(Button.Pressed, "#model-btn")
    def change_model(self):
        """Change the LLM model."""
        try:
            self.push_screen(ModelSelectorScreen(), self.handle_model_change)
        except Exception as e:
            self.log_message(f"Error opening model selector: {e}")
    
    def handle_model_change(self, model: str | None) -> None:
        """Handle model change from the modal."""
        if model:
            self.model_name = model
            self.log_message(f"Changed model to: {model}")
            self.log_message("Re-initializing agents...")
            self.initialize_agents()
        else:
            self.log_message("Model change cancelled")
    
    @on(Button.Pressed, "#report-btn")
    async def view_report(self):
        """Open the trace report in browser."""
        if self.main_trace:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = f"examples/generated_sketches/traces/textual_trace_{timestamp}.json"
            
            # Save trace
            json_trace = self.collector.export_trace(self.main_trace.id, format="json")
            with open(json_path, "w") as f:
                f.write(json_trace)
            
            # Generate HTML report
            try:
                html_path = generate_html_report(json_path)
                self.log_message(f"📊 Report generated: {html_path}")
                
                # Open in browser
                import webbrowser
                webbrowser.open(f"file://{Path(html_path).absolute()}")
            except Exception as e:
                self.log_message(f"⚠️ Could not generate report: {e}")
    
    def update_trace_info(self):
        """Update the trace information display."""
        if self.main_trace and self.collector:
            trace_display = self.query_one("#trace-display", Static)
            
            try:
                # Get basic trace info that's always available
                info = f"""Trace ID: {self.main_trace.id[:8]}...
Status: {self.main_trace.status}"""
                
                # Try to get duration if available
                if hasattr(self.main_trace, 'duration_ms'):
                    info += f"\nDuration: {self.main_trace.duration_ms:.0f}ms"
                
                # Try to get events count if possible
                try:
                    # Use the collector to get trace data safely
                    if self.collector.enabled:
                        info += "\nTrace data available"
                except:
                    pass
                    
                trace_display.update(info)
                self.query_one("#report-btn", Button).disabled = False
                
            except Exception as e:
                # Fallback to basic info
                trace_display.update(f"Trace ID: {self.main_trace.id[:8]}...\nStatus: {self.main_trace.status}")
                self.query_one("#report-btn", Button).disabled = False
    
    def action_new_sketch(self):
        """Create a new sketch (reset UI)."""
        self.run_worker(self.reset_ui())
    
    def action_run_sketch(self):
        """Run the current sketch."""
        if not self.is_running:
            self.run_sketch()
        else:
            self.run_worker(self.stop_sketch())
    
    def action_save_sketch(self):
        """Save the current sketch."""
        self.save_sketch()
    
    def action_toggle_chat(self):
        """Toggle the chat panel."""
        self.toggle_chat_panel()
    
    def action_show_help(self):
        """Show the help dialog."""
        self.push_screen(HelpDialog())
    
    def action_show_tutorial(self):
        """Show the tutorial modal."""
        # Resume from saved step or start fresh if completed
        initial_step = 0 if self.tutorial_completed else self.tutorial_current_step
        tutorial = TutorialScreen(initial_step=initial_step)
        tutorial.app_reference = self  # Give tutorial access to main app
        self.push_screen(tutorial, self.handle_tutorial_completion)
    
    def handle_tutorial_completion(self, result) -> None:
        """Handle tutorial completion or action."""
        if isinstance(result, tuple):
            action, step = result
            
            # Always save the current step
            self.tutorial_current_step = step
            
            if action == "completed":
                # Tutorial was completed
                self.tutorial_completed = True
                self.tutorial_current_step = 0  # Reset for next time
                self.log_message("🎓 Tutorial completed! You're ready to create artificial life!")
            elif action == "closed":
                # Tutorial was closed - step already saved above
                self.log_message("📚 Tutorial paused - press F2 to continue where you left off")
            elif action == "generate":
                # Auto-advance to next step after generation
                self.tutorial_current_step = min(step + 1, 8)  # Advance but don't exceed max
                self.log_message("🎓 Tutorial: Triggering sketch generation...")
                # Set description and trigger generation
                self.set_timer(0.1, self.tutorial_generate_from_main)
            elif action == "refine":
                # Auto-advance to next step after refinement
                self.tutorial_current_step = min(step + 1, 8)  # Advance but don't exceed max
                self.log_message("🎓 Tutorial: Triggering color refinement...")
                # Trigger refinement
                self.set_timer(0.1, self.tutorial_refine_from_main)
        else:
            # Fallback for unexpected results
            self.log_message("📚 Tutorial closed")
    
    def tutorial_generate_from_main(self):
        """Generate sketch from main UI after modal closes."""
        try:
            description_input = self.query_one("#description-input", TextArea)
            description_input.text = "Two species, red and green, repel each other strongly."
            self.generate_sketch_worker()
        except Exception as e:
            self.log_message(f"⚠️ Could not trigger generation: {e}")
    
    def tutorial_refine_from_main(self):
        """Apply refinement from main UI after modal closes."""
        try:
            self.tutorial_apply_color_refinement()
        except Exception as e:
            self.log_message(f"⚠️ Could not trigger refinement: {e}")
    
    def tutorial_generate_demo_sketch(self):
        """Tutorial helper: Generate the demo sketch automatically."""
        if not self.agents_ready or not self.behavior_agent:
            self.notify("⚠️ Agents not ready - please wait for initialization", severity="warning", timeout=3)
            return
        
        # Set the demo description
        description_input = self.query_one("#description-input", TextArea)
        description_input.text = "Two species, red and green, repel each other strongly."
        
        # Trigger generation
        self.log_message("🎓 Tutorial: Auto-generating demo sketch...")
        self.generate_sketch_worker()
        
        # Advance tutorial if active
        if self.tutorial_active and self.tutorial_step == 1:
            self.set_timer(2.0, self.action_next_tutorial_step)
    
    def tutorial_apply_color_refinement(self):
        """Tutorial helper: Apply the color refinement automatically."""
        if not self.sketch_refiner or not self.current_sketch_code:
            self.notify("⚠️ No sketch available for refinement", severity="warning", timeout=3)
            return
        
        refinement_request = "Let's change the species colors to be a rust orange and teal please."
        
        # Add to chat history
        try:
            chat_container = self.query_one("#chat-history", ScrollableContainer)
            user_message = Static(f"Tutorial: {refinement_request}", classes="chat-message user-message")
            chat_container.mount(user_message)
            chat_container.scroll_end(animate=False)
        except Exception:
            pass
        
        # Apply refinement
        self.log_message("🎓 Tutorial: Applying color refinement...")
        self.apply_refinement_worker(refinement_request)
        
        # Advance tutorial if active
        if self.tutorial_active and self.tutorial_step == 4:
            self.set_timer(3.0, self.action_next_tutorial_step)
    
    
    def action_quit(self):
        """Quit the application."""
        if self.sketch_process:
            self.sketch_process.terminate()
        self.exit()


def main():
    """Main entry point."""
    app = TolveraTextualUI()
    app.run()


if __name__ == "__main__":
    main()
