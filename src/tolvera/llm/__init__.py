"""
Tölvera LLM Submodule

This submodule provides tools for interacting with Large Language Models (LLMs)
to generate Tölvera sketch configurations and Python code based on
natural language descriptions.

It includes both the legacy template-based system and the new PoE
(Product of Experts) behavior system.
"""

# Legacy imports (being phased out)
try:
    from .llm import LLM
    from .definitions import SketchConfig, ParticleShape, BackgroundBehavior
    from .codegen import generate_code_from_sketch_config
    _legacy_available = True
except ImportError:
    _legacy_available = False

# PoE system imports
from .poe_core import SimpleProgrammaticExpert, PoEBehaviorSystem
from .poe_integration import TolveraBehaviorAgent, AsyncTolveraBehaviorAgent
from .poe_synthesis import ExpertSynthesizer, SimpleSynthesizer
from .poe_experts import ExpertLibrary

__all__ = [
    # PoE system
    "SimpleProgrammaticExpert",
    "PoEBehaviorSystem", 
    "TolveraBehaviorAgent",
    "AsyncTolveraBehaviorAgent",
    "ExpertSynthesizer",
    "SimpleSynthesizer",
    "ExpertLibrary",
]

# Add legacy exports if available
if _legacy_available:
    __all__.extend([
        "LLM",
        "SketchConfig",
        "generate_code_from_sketch_config",
        "ParticleShape",
        "BackgroundBehavior",
    ])

__all__ = [name for name in __all__ if globals().get(name) is not None]