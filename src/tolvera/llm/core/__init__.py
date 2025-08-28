"""Core LLM synthesis components.

This module provides the core functionality for synthesizing particle behaviors
from natural language descriptions. It includes components for code generation,
behavior analysis, state management, and orchestration of the synthesis pipeline.

The module exports the following main classes:
    - BehaviorOrchestrator: Main orchestrator coordinating the synthesis pipeline
    - CodeGenerator: Generates Taichi code from behavior descriptions
    - StateManager: Manages particle and global state definitions
    - BehaviorAnalyzer: Analyzes and decomposes complex behaviors
    - ExpertRegistry: Manages registered expert functions

Example:
    Basic usage of the core synthesis components:
    
    ```python
    from tolvera.llm.core import BehaviorOrchestrator
    
    # Initialize orchestrator
    orchestrator = BehaviorOrchestrator(tv, model_name="gemini-2.0-flash")
    
    # Add behaviors
    await orchestrator.add_behavior("particles attract to center", weight=1.0)
    
    # Generate complete sketch
    sketch, path = orchestrator.generate_sketch(
        description="Gravity simulation",
        filename="gravity_sketch"
    )
    ```
"""

from .data_models import (
    BehaviorSynthesisRequest,
    BehaviorSynthesisResponse,
    ExpertFunction,
    StateDefinition,
    SpeciesConfiguration,
    IntegrationKernel,
    TemporalUpdate,
    ForceComputation,
    VectorExpression
)
from .code_generator import CodeGenerator
from .state_manager import StateManager
from .behavior_orchestrator import BehaviorOrchestrator
from .behavior_analyzer import BehaviorAnalyzer, DecomposedBehavior, BehaviorComponent, SpeciesColorMapping
from .behavior_registry import ExpertRegistry, ExpertInfo

__all__ = [
    'BehaviorSynthesisRequest',
    'BehaviorSynthesisResponse',
    'ExpertFunction',
    'StateDefinition',
    'SpeciesConfiguration',
    'IntegrationKernel',
    'TemporalUpdate',
    'ForceComputation',
    'VectorExpression',
    'CodeGenerator',
    'StateManager',
    'BehaviorOrchestrator',
    'BehaviorAnalyzer',
    'DecomposedBehavior',
    'BehaviorComponent',
    'SpeciesColorMapping',
    'ExpertRegistry',
    'ExpertInfo'
]