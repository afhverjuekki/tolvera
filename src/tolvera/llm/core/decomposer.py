"""
Behavior decomposition for complex multi-agent simulations.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from .prompt_loader import get_prompt_loader

from ..debug.tracing import get_collector

logger = logging.getLogger(__name__)


# ... (keeping all the existing model definitions)
class SpeciesColorMapping(BaseModel):
    """RGBA color values for a species."""
    species_id: int = Field(description="Species ID (0-based)")
    rgba_values: List[float] = Field(description="RGBA color values [r, g, b, a] (0.0-1.0)")


class SpeciesNameMapping(BaseModel):
    """Mapping of species ID to name"""
    species_id: int = Field(description="Species ID (0-based)")
    name: str = Field(description="Name of the species")

class SpeciesColorMapping(BaseModel):
    """Mapping of species ID to color description"""
    species_id: int = Field(description="Species ID (0-based)")
    color_description: str = Field(description="Color description (e.g., 'red', 'blue', 'tangerine')")

class SpeciesConfiguration(BaseModel):
    """Species configuration with colors and names"""
    total_count: int = Field(description="Total number of species")
    species_names: List[SpeciesNameMapping] = Field(default_factory=list, description="Species names with IDs")
    species_color_descriptions: List[SpeciesColorMapping] = Field(default_factory=list, description="Species colors with IDs")


class BehaviorComponent(BaseModel):
    """A single behavior component for implementation"""
    expert_name: str = Field(description="Name of the expert function")
    expert_type: str = Field(description="Type of expert (force, interaction, state_update, etc.)")
    description: str = Field(description="Human-readable description of the behavior")
    implementation: str = Field(description="High-level implementation guidance")
    applies_to_species: Optional[List[int]] = Field(default=None, description="Species IDs this applies to")
    implementation_details: Optional[List[str]] = Field(default=None, description="Step-by-step implementation")
    parameters: Optional[List[str]] = Field(default=None, description="Required parameters")
    priority: Optional[float] = Field(default=1.0, description="Priority weight")
    force_formula: Optional[str] = Field(default=None, description="Mathematical force formula")
    required_states: Optional[List[Tuple[str, str, str, float, float]]] = Field(default=None, description="Required custom states")
    is_temporal: Optional[bool] = Field(default=False, description="Whether this is time-dependent")


class DecomposedBehavior(BaseModel):
    """Result of behavior decomposition"""
    original_description: str = Field(description="Original behavior description")
    interpretation: str = Field(description="System's interpretation of the behavior")
    behavior_category: str = Field(description="Category: particle_system, pure_drawing, or hybrid")
    species_info: SpeciesConfiguration = Field(description="Species configuration")
    components: List[BehaviorComponent] = Field(description="List of behavior components")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Shared context")
    suggested_states: Optional[List[Tuple[str, str, str, float, float]]] = Field(default=None, description="Suggested custom states")


class DecompositionDependencies(BaseModel):
    """Dependencies for decomposition agent"""
    context_patterns: Dict[str, str] = Field(description="Context patterns")
    alife_examples: List[str] = Field(description="Artificial life examples")


class BehaviorDecomposer:
    """Decomposes complex behavior descriptions into implementable components"""
    
    def __init__(self, model_name: str, provider: str = "gemini", prompt_builder=None, api_key: Optional[str] = None):
        """Initialize the decomposer with model configuration"""
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        
        # Pattern library for decomposition
        self.BEHAVIOR_PATTERNS = {
            "predator_prey": "Chasing and fleeing behaviors between species",
            "flocking": "Cohesion, alignment, and separation forces",
            "particle_life": "Species-based attraction and repulsion matrices",
            "cellular_automata": "Grid-based life and death rules",
            "foraging": "Resource seeking and consumption behaviors",
            "territory": "Spatial claiming and defense mechanisms",
        }
        
        logger.info(f"BehaviorDecomposer using provider '{self.provider}' with model '{model_name}'")
        self.prompt_builder = prompt_builder
        
        # Create the actual model using ModelFactory
        from .model_factory import ModelFactory
        self.model = ModelFactory.create_model(model_name, api_key)
        
        self.decomposition_agent = self._create_decomposition_agent()
        
    def _create_decomposition_agent(self) -> Agent[DecompositionDependencies, DecomposedBehavior]:
        # Load decomposer prompts using PromptLoader
        loader = get_prompt_loader()
        
        # Combine the decomposer prompts
        system_prompt_parts = [
            loader.load_prompt("decomposition/behavior_decomposition_system.txt"),
            loader.load_prompt("decomposition/behavior_decomposition_examples.txt"),
            loader.load_prompt("decomposition/behavior_decomposition_criteria.txt")
        ]
        
        system_prompt = "\n\n".join(system_prompt_parts)
        
        # Log the assembled system prompt
        logger.info(f"[DECOMPOSER] System Prompt assembled: {len(system_prompt)} chars")
        logger.debug(f"[DECOMPOSER] System prompt preview (first 500 chars): {system_prompt[:500]}")
        if logger.isEnabledFor(logging.DEBUG):
            # In debug mode, log the full prompt
            logger.debug(f"[DECOMPOSER] Full system prompt:\n{system_prompt}")
        
        agent = Agent(
            self.model,
            deps_type=DecompositionDependencies,
            result_type=DecomposedBehavior,
            system_prompt=system_prompt
        )
        
        return agent
    
    async def decompose(self, description: str) -> DecomposedBehavior:
        collector = get_collector()
        
        with collector.trace_node("decompose_behavior", "decomposition",
                                 description=description) as node:
            logger.info(f"Decomposing behavior: {description}")
            
            deps = DecompositionDependencies(
                context_patterns=self.BEHAVIOR_PATTERNS,
                alife_examples=self._get_relevant_examples(description)
            )
            
            prompt = f'''As an EXPERT IN ARTIFICIAL LIFE, analyze this behavior and create SCIENTIFICALLY-ACCURATE expert specifications: "{description}"
 
 RECOGNIZE KEY A-LIFE PATTERNS:
 - "different species interact" → PARTICLE LIFE with interaction matrix
 - "attraction and repulsion" → PARTICLE LIFE with species pairs
 - "forms clusters" → PARTICLE LIFE emergent behavior
 - "species X attracts/repels species Y" → PARTICLE LIFE matrix entry
 - "chase/hunt/flee" → PREDATOR-PREY dynamics
 - "together/grouping/schooling" → BOIDS flocking behaviors
 - "cells live or die" → GAME OF LIFE cellular automaton

 CRITICAL: Generate IMPLEMENTATION-READY components with EXACT mathematical formulas!
 Every component needs force calculations, step-by-step implementation, and parameter ranges!

 1. IDENTIFY SPECIES AND COLORS: Extract species and their colors
    - "small green fish" and "larger red predators" = 2 species
      → species_color_descriptions: [
          {{"species_id": 0, "color_description": "red"}},
          {{"species_id": 1, "color_description": "green"}}
      ]
    - For descriptions mentioning roles but no colors, use defaults:
      → predators: "red", prey: "green", neutral: "blue"
    - For "blue and teal species compete for green food":
      → 3 species: blue=0, teal=1, food=2

 2. GENERATE COMPONENTS FOR EACH BEHAVIOR: 
    - "predator hunts prey" needs TWO experts: predator_chase + prey_flee
    - "fish school together" needs THREE experts: cohesion + alignment + separation
    - "species compete for food" needs MULTIPLE experts for each species

 3. IMPLEMENTATION DETAILS: Include 3-5 step algorithmic instructions
    - "find nearest target particle within detection radius"
    - "calculate direction vector from current position to target"  
    - "apply force inversely proportional to distance with smooth falloff"
    - "clamp force magnitude to prevent instability"

 4. SPECIES TARGETING: ALWAYS set applies_to_species for specific behaviors
    - Species mentioned → applies_to_species: [species_id]
    - All particles → omit applies_to_species (defaults to all)
    
 5. PARAMETERS: Include realistic physical ranges
    - chase_strength: 100-500, detection_radius: 50-200
    - escape_strength: 200-600 (stronger than chase for drama)
    - interaction_radius: 30-150, separation_distance: 10-40'''
            
            # Store prompt info in metadata
            if node:
                node.metadata = {"prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt}
            
            # Log the user prompt being sent
            logger.info(f"[DECOMPOSER] Running decomposition with user prompt: {len(prompt)} chars")
            logger.debug(f"[DECOMPOSER] User prompt: {prompt}")
            
            start_time = time.time()
            result = await self.decomposition_agent.run(prompt, deps=deps)
            end_time = time.time()
            
            # Log the LLM call (simplified for now - TODO: fix logging conflicts)
            try:
                collector.log_llm_call(
                    model=self.model_name,
                    system_prompt="Decomposition agent system prompt",
                    user_prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    response=str(result.data)[:200] + "..." if len(str(result.data)) > 200 else str(result.data)
                )
            except Exception as e:
                logger.warning(f"Failed to log LLM call: {e}")
            # Store result in node output_data
            if node:
                node.output_data = {
                    "result": result.data.model_dump(),
                    "timing": {
                        "model": self.model_name,
                        "provider": self.provider,
                        "duration": end_time - start_time,
                        "success": True
                    }
                }
            
            logger.info(f"Decomposed into {len(result.data.components)} components")
            
            return result.data
    
    def _get_relevant_examples(self, description: str) -> List[str]:
        """Get relevant examples based on the description"""
        # Simple keyword matching for now
        examples = []
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["chase", "hunt", "predator", "prey"]):
            examples.append("predator_prey_example")
        if any(word in description_lower for word in ["flock", "school", "together", "group"]):
            examples.append("flocking_example")
        if any(word in description_lower for word in ["attract", "repel", "interact"]):
            examples.append("particle_life_example")
        if any(word in description_lower for word in ["live", "die", "cell", "neighbor"]):
            examples.append("cellular_automata_example")
            
        return examples