"""Behavior analysis and decomposition.

This module provides functionality for analyzing natural language behavior
descriptions and decomposing them into implementable components.
"""

import time
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..prompts.prompt_loader import get_prompt_loader
from ..debug.tracing import get_collector


# ==============================================================================
# Pydantic Models for Structured Decomposition
# ==============================================================================

class SpeciesColorMapping(BaseModel):
    """RGBA color values for a species."""
    species_id: int = Field(description="Species ID (0-based)")
    rgba_values: List[float] = Field(description="RGBA color values [r, g, b, a] (0.0-1.0)")


class SpeciesNameMapping(BaseModel):
    """Mapping of species ID to its semantic name."""
    species_id: int = Field(description="Species ID (0-based)")
    name: str = Field(description="Name of the species")


class SpeciesColorDescriptionMapping(BaseModel):
    """Mapping of species ID to its color description."""
    species_id: int = Field(description="Species ID (0-based)")
    color_description: str = Field(description="Color description (e.g., 'red', 'blue', 'tangerine')")


class SpeciesConfiguration(BaseModel):
    """Complete species configuration with colors and names."""
    total_count: int = Field(description="Total number of species")
    species_names: List[SpeciesNameMapping] = Field(
        default_factory=list, 
        description="Species names with IDs"
    )
    species_color_descriptions: List[SpeciesColorDescriptionMapping] = Field(
        default_factory=list, 
        description="Species colors with IDs"
    )


class StateSpecification(BaseModel):
    """Specification for a custom state variable."""
    name: str = Field(description="State variable name (e.g., 'energy', 'grid_x')")
    category: Literal["global", "particle", "species"] = Field(
        description="State category: global (shared), particle (per-particle), or species (per-species)"
    )
    type: str = Field(description="Taichi type (e.g., 'ti.f32', 'ti.i32', 'ti.math.vec2')")
    min: float = Field(description="Minimum value for the state")
    max: float = Field(description="Maximum value for the state")


class BehaviorComponent(BaseModel):
    """A single behavior component for implementation."""
    expert_name: str = Field(description="Name of the expert function")
    expert_type: str = Field(description="Type of expert (force, interaction, state_update, etc.)")
    description: str = Field(description="Human-readable description of the behavior")
    implementation: str = Field(description="High-level implementation guidance")
    applies_to_species: Optional[List[int]] = Field(
        default=None, 
        description="Species IDs this applies to"
    )
    implementation_details: Optional[List[str]] = Field(
        default=None, 
        description="Step-by-step implementation"
    )
    parameters: Optional[List[str]] = Field(
        default=None, 
        description="Required parameters"
    )
    priority: Optional[float] = Field(
        default=1.0, 
        description="Priority weight"
    )
    force_formula: Optional[str] = Field(
        default=None, 
        description="Mathematical force formula"
    )
    required_states: Optional[List[StateSpecification]] = Field(
        default=None, 
        description="Required custom states with explicit field definitions"
    )
    is_temporal: Optional[bool] = Field(
        default=False, 
        description="Whether this is time-dependent"
    )


class DecomposedBehavior(BaseModel):
    """Complete result of behavior decomposition."""
    original_description: str = Field(description="Original behavior description")
    interpretation: str = Field(description="System's interpretation of the behavior")
    behavior_category: str = Field(description="Category: particle_system, pure_drawing, or hybrid")
    species_info: SpeciesConfiguration = Field(description="Species configuration")
    components: List[BehaviorComponent] = Field(description="List of behavior components")
    context: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Additional context information"
    )
    suggested_states: Optional[List[StateSpecification]] = Field(
        default=None, 
        description="Suggested custom states with explicit field definitions"
    )


# ==============================================================================
# Main Decomposer Class
# ==============================================================================

class BehaviorAnalyzer:
    """Analyzes and decomposes complex behavior descriptions.
    
    This class uses pydantic-ai to analyze natural language descriptions
    of particle behaviors and decompose them into specific, implementable
    components. It identifies species, required states, behavior types,
    and generates implementation guidance for synthesis.
    
    Attributes:
        model_name (str): Name of the LLM model being used.
        provider (str): LLM provider (gemini, anthropic, openai).
        model: LLM model instance from ModelFactory.
        decomposition_agent: Pydantic-ai agent for structured decomposition.
    """
    
    def __init__(
        self, 
        model_name: str, 
        provider: str = "gemini", 
        api_key: Optional[str] = None
    ):
        """Initialize the behavior analyzer.
        
        Args:
            model_name (str): Name of the LLM model to use.
            provider (str): Provider for the model. Defaults to "gemini".
            api_key (Optional[str]): API key for the provider. If None,
                attempts to load from environment variables.
        """
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        
        # Create the model using ModelFactory
        from .llm_factory import ModelFactory
        self.model = ModelFactory.create_model(model_name, api_key)
        
        # Create the decomposition agent
        self.decomposition_agent = self._create_decomposition_agent()
        
    def _create_decomposition_agent(self) -> Agent[None, DecomposedBehavior]:
        """Create the pydantic-ai agent for decomposition.
        
        Loads and combines prompt templates to create a structured
        decomposition agent.
        
        Returns:
            Agent[None, DecomposedBehavior]: Configured agent instance.
        """
        # Load and combine decomposer prompts
        loader = get_prompt_loader()
        
        system_prompt_parts = [
            loader.load_prompt("decomposition/behavior_decomposition_system.txt"),
            loader.load_prompt("decomposition/behavior_decomposition_examples.txt"),
            loader.load_prompt("decomposition/behavior_decomposition_criteria.txt")
        ]
        
        system_prompt = "\n\n".join(system_prompt_parts)
        
        # Create agent without dependencies (they weren't being used)
        agent = Agent(
            self.model,
            output_type=DecomposedBehavior,
            system_prompt=system_prompt
        )
        
        return agent
    
    async def analyze(self, description: str) -> DecomposedBehavior:
        """Analyze and decompose a behavior description.
        
        Breaks down complex behaviors into components, identifying species,
        required states, and implementation strategies.
        
        Args:
            description (str): Natural language description of the behavior.
            
        Returns:
            DecomposedBehavior: Structured decomposition containing:
                - Behavior components with implementation guidance
                - Species configuration
                - Required custom states
                - Implementation priorities
        """
        collector = get_collector()
        
        with collector.trace_node(
            "decompose_behavior", 
            "decomposition",
            description=description
        ) as node:
            # Load the user prompt template
            loader = get_prompt_loader()
            prompt = loader.load_prompt(
                "decomposition/decomposition_user.txt", 
                description=description
            )
            
            # Store prompt in trace metadata
            if node:
                node.metadata = {"prompt": prompt}
            
            # Run the agent
            start_time = time.time()
            result = await self.decomposition_agent.run(prompt)
            duration = time.time() - start_time
            
            # Log the LLM call for debugging
            collector.log_llm_call(
                model=self.model_name,
                system_prompt="Decomposition agent system prompt",
                user_prompt=prompt,
                response=str(result.output)
            )
            
            # Store result in trace node
            if node:
                node.output_data = {
                    "result": result.output.model_dump(),
                    "timing": {
                        "model": self.model_name,
                        "provider": self.provider,
                        "duration": duration,
                        "success": True
                    }
                }
            
            return result.output