"""
Tölvera LLM Submodule

This submodule provides tools for interacting with Large Language Models (LLMs)
to generate Tölvera sketch configurations and Python code based on
natural language descriptions.

This module requires Ollama to be installed and running locally.
"""

from .llm import LLM
from .definitions import SketchConfig
from .codegen import generate_code_from_sketch_config
from .definitions import ParticleShape, BackgroundBehavior

__all__ = [
    "LLM",
    "SketchConfig",
    "generate_code_from_sketch_config",
    "ParticleShape",
    "BackgroundBehavior",
]

__all__ = [name for name in __all__ if globals().get(name) is not None]