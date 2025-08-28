"""Tölvera LLM synthesis system.

This module provides natural language synthesis of particle behaviors through
LLM-based code generation. It transforms descriptive language into executable
Taichi code for particle simulations.

The module includes:
    - BehaviorOrchestrator: Main orchestrator for the synthesis pipeline
    - CodeGenerator: Generates Taichi code from descriptions
    - BehaviorAnalyzer: Analyzes and decomposes complex behaviors
    - StateManager: Manages custom particle and global states
    - SketchRefiner: Refines sketches for architectural patterns
    - TemplateRenderer: Renders code templates

Example:
    Basic usage of the LLM synthesis system:
    
    ```python
    from tolvera import Tolvera
    from tolvera.llm import BehaviorOrchestrator
    
    # Initialize Tölvera
    tv = Tolvera(width=1920, height=1080, pn=1000, sn=4)
    
    # Create behavior orchestrator
    orchestrator = BehaviorOrchestrator(tv, model_name="gemini-2.0-flash")
    
    # Add behaviors
    await orchestrator.add_behavior("particles attract to center", weight=1.0)
    await orchestrator.add_behavior("red predators chase blue prey", weight=0.5)
    
    # Generate complete sketch
    sketch, path = orchestrator.generate_sketch(
        description="Ecosystem simulation",
        filename="ecosystem_sketch"
    )
    ```
"""

from .core.behavior_orchestrator import BehaviorOrchestrator
from .core.code_generator import CodeGenerator
from .core.behavior_analyzer import BehaviorAnalyzer
from .core.state_manager import StateManager
from .core.sketch_refiner import SketchRefiner
from .templates.template_renderer import TemplateRenderer

__all__ = [
    "BehaviorOrchestrator",
    "CodeGenerator",
    "BehaviorAnalyzer",
    "StateManager",
    "SketchRefiner",
    "TemplateRenderer",
]