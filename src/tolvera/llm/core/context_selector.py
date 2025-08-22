"""
Intelligent LLM-powered context selector for dynamic prompt building.
Replaces static keyword-based context selection with AI-driven analysis.
"""

import logging
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from .model_factory import ModelFactory
from ..debug.tracing import get_collector

logger = logging.getLogger(__name__)


class ContextSelectionResponse(BaseModel):
    """Structured response from context selection LLM."""
    selected_contexts: List[str] = Field(
        description="List of context names relevant to the behavior description (maximum 3)"
    )
    reasoning: str = Field(
        description="Brief explanation of why these contexts were selected"
    )


class ContextSelector:
    """LLM-powered intelligent context selector for expert synthesis."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        """Initialize the context selector with a fast, lightweight model."""
        self.model_name = model_name
        self.model = ModelFactory.create_model(model_name, api_key)
        self.provider = ModelFactory.get_provider_for_model(model_name)
        
        # Available contexts from the existing prompts.py
        self.available_contexts = {
            'core_api': 'Tölvera core API functions and particle access patterns',
            'pixels_api': 'Pixel manipulation and drawing functions',
            'taichi_fundamentals': 'Core Taichi syntax, types, and GPU programming',
            'state_access': 'Custom state access patterns for particles, global, species',
            'boundaries': 'Boundary handling, wrapping, collision detection',
            'movement': 'Basic movement patterns (seek, flee, wander, orbital)',
            'flocking': 'Flocking behaviors (separation, alignment, cohesion)',
            'interaction': 'Particle-particle interactions (chase, repel, attraction)',
            'temporal': 'Time-based behaviors (cycles, energy, aging)',
            'cellular': 'Cellular automata patterns (Game of Life, grid behaviors)',
            'emergent': 'Emergent behaviors (slime mold, ant trails, fireflies)',
            'drawing': 'Visual effects and trail rendering',
            'drawing_api': 'Tölvera drawing API reference',
            'vera_patterns': 'Tölvera vera behavior library patterns',
            'vera_interactions': 'Multi-species interaction patterns',
            'species_interactions': 'Species-specific behavior patterns',
            'alife_patterns': 'Artificial life computational patterns',
            'iml_patterns': 'Interactive machine learning patterns',
            'evolution': 'Evolutionary algorithms and genetic patterns',
            'ecosystem': 'Ecosystem simulation patterns',
            'morphogenesis': 'Growth and development patterns',
            'swarm': 'Swarm intelligence algorithms',
            'initialization': 'Particle initialization patterns',
            'species_initialization': 'Multi-species initialization',
            'temporal_updates': 'Temporal state update patterns',
            'temporal_dynamics': 'Advanced temporal dynamics',
            'temporal_examples': 'Temporal behavior examples',
            'temporal_patterns_extended': 'Extended temporal patterns',
            'configuration': 'System configuration patterns'
        }
        
        # Create the selection agent
        self._create_selection_agent()
        
        logger.info(f"ContextSelector initialized with {len(self.available_contexts)} available contexts")
    
    def _create_selection_agent(self):
        """Create the pydantic-ai agent for context selection."""
        
        context_list = "\n".join([f"- {name}: {desc}" for name, desc in self.available_contexts.items()])
        
        system_prompt = f"""You are an expert context selector for Tölvera particle behavior synthesis.

Your task is to analyze behavior descriptions and intelligently select the most relevant contexts from the available library.

AVAILABLE CONTEXTS:
{context_list}

SELECTION GUIDELINES:

1. **Movement & Physics**: For basic forces, movement, gravity, springs
   - Select: core_api, taichi_fundamentals, movement
   - Add boundaries if mentioned: "bounce", "wrap", "edge"

2. **Flocking & Swarming**: For group behaviors, herding, schooling
   - Select: flocking, vera_patterns
   - Add swarm for advanced collective behaviors

3. **Species Interactions**: For multi-species, predator-prey, chase-flee
   - Select: interaction, species_interactions, vera_interactions
   - Always include core_api and taichi_fundamentals

4. **Temporal Behaviors**: For time-based, energy, cycles, aging
   - Select: temporal, temporal_dynamics, temporal_examples
   - Add temporal_patterns_extended for complex temporal logic

5. **Visual Effects**: For drawing, trails, glows, visual elements
   - Select: drawing, drawing_api, pixels_api
   - Include temporal_dynamics if animated effects

6. **Cellular Automata**: For grid-based, Game of Life, neighbors
   - Select: cellular, state_access
   - Add temporal for evolutionary rules

7. **Emergent Behaviors**: For slime mold, ant trails, pheromones
   - Select: emergent, vera_patterns
   - Add drawing_api for trail deposition

8. **Ecosystem Simulation**: For predator-prey, food webs, evolution
   - Select: ecosystem, interaction, species_interactions
   - Add evolution if genetic algorithms mentioned

SELECTION STRATEGY:
- Always include core_api and taichi_fundamentals for synthesis tasks
- Be selective: choose EXACTLY 3 contexts maximum for focused prompts
- Prioritize contexts that directly relate to the behavior
- Include state_access if custom states are needed
- Add drawing contexts only if visual effects are mentioned

IMPORTANT: Select contexts that will provide the most relevant patterns and examples for implementing the specific behavior described."""

        self.selection_agent = Agent(
            self.model,
            output_type=ContextSelectionResponse,
            system_prompt=system_prompt
        )
    
    async def select_contexts(
        self, 
        description: str, 
        expert_type: str = "force",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ContextSelectionResponse:
        """
        Select relevant contexts for a behavior description.
        
        Args:
            description: Natural language behavior description
            expert_type: Type of expert being synthesized (force, interaction, visual, etc.)
            additional_context: Optional additional context for selection
            
        Returns:
            ContextSelectionResponse with selected contexts and reasoning
        """
        logger.info(f"Selecting contexts for: {description} (type: {expert_type})")
        
        # Build the selection prompt
        prompt = f"""Analyze this behavior description and select the most relevant contexts:

BEHAVIOR: "{description}"
EXPERT TYPE: {expert_type}

Consider:
1. What type of behavior is this? (movement, interaction, visual, temporal, etc.)
2. What technical patterns will be needed? (forces, state access, drawing, etc.)
3. Are there specific Tölvera features required? (species, states, pixels, etc.)
4. What examples would be most helpful for implementation?

Select EXACTLY 3 most relevant contexts that will provide focused, useful patterns for implementing this specific behavior."""

        if additional_context:
            prompt += f"\n\nADDITIONAL CONTEXT:\n{additional_context}"
        
        try:
            # Create trace node for the LLM call
            collector = get_collector()
            
            with collector.trace_node("llm_context_selection", "llm_call",
                                     description=description,
                                     expert_type=expert_type,
                                     model=self.model_name) as node:
                result = await self.selection_agent.run(prompt)
                response = result.output
                
                if node:
                    # Log LLM call details
                    # Get system prompt properly from pydantic-ai Agent
                    try:
                        system_prompt_text = str(self.selection_agent._system_prompt) if hasattr(self.selection_agent, '_system_prompt') else "Context Selection System Prompt"
                    except Exception:
                        system_prompt_text = "Context Selection System Prompt"
                    
                    collector.log_llm_call(
                        model=self.model_name,
                        system_prompt=system_prompt_text,
                        user_prompt=prompt,
                        response=str(response),
                        parsed_response=response.model_dump() if hasattr(response, 'model_dump') else str(response)
                    )
            
            # Validate selected contexts exist
            valid_contexts = []
            invalid_contexts = []
            
            for context in response.selected_contexts:
                if context in self.available_contexts:
                    valid_contexts.append(context)
                else:
                    invalid_contexts.append(context)
            
            if invalid_contexts:
                logger.warning(f"Invalid contexts selected: {invalid_contexts}")
            
            # Update response with only valid contexts
            response.selected_contexts = valid_contexts
            
            logger.info(f"Selected {len(valid_contexts)} contexts: {valid_contexts}")
            logger.debug(f"Selection reasoning: {response.reasoning}")
            
            return response
            
        except Exception as e:
            logger.error(f"Context selection failed: {e}")
            # Fallback to basic contexts
            fallback_contexts = ['core_api', 'taichi_fundamentals', 'movement']
            return ContextSelectionResponse(
                selected_contexts=fallback_contexts,
                reasoning=f"Fallback selection due to error: {e}",
            )
    
    def select_contexts_for_refinement(
        self, 
        description: str, 
        sketch_code: str,
        refinement_type: str = "general"
    ) -> List[str]:
        """
        Synchronous context selection for sketch refinement.
        Uses pattern matching for speed in refinement scenarios.
        
        Args:
            description: Refinement description
            sketch_code: Current sketch code
            refinement_type: Type of refinement (error_correction, feature_addition, etc.)
            
        Returns:
            List of relevant context names
        """
        desc_lower = description.lower()
        code_lower = sketch_code.lower()
        
        # For implementation and error correction, provide comprehensive contexts
        if refinement_type == "implementation":
            contexts = ['taichi_fundamentals', 'movement', 'flocking']
        elif refinement_type == "error_correction":
            contexts = ['taichi_fundamentals', 'taichi_crashes']
        else:
            contexts = ['taichi_fundamentals']  # Basic fallback
        
        # Error correction gets crash fixes
        if refinement_type == "error_correction" or "error" in desc_lower:
            if 'taichi_crashes' not in contexts:
                contexts.append('taichi_crashes')
        
        # Analyze description and code for patterns
        if any(word in desc_lower + code_lower for word in ['flock', 'boid', 'align', 'cohesion', 'separation']):
            if 'flocking' not in contexts:
                contexts.append('flocking')
            if 'movement' not in contexts:
                contexts.append('movement')
        
        if any(word in desc_lower + code_lower for word in ['chase', 'flee', 'hunt', 'species', 'interaction']):
            contexts.extend(['interaction', 'species_interactions'])
        
        if any(word in desc_lower + code_lower for word in ['draw', 'trail', 'glow', 'visual', 'pixel']):
            contexts.extend(['drawing', 'drawing_api'])
        
        if any(word in desc_lower + code_lower for word in ['time', 'energy', 'phase', 'cycle', 'temporal']):
            contexts.extend(['temporal', 'temporal_dynamics'])
        
        if any(word in desc_lower + code_lower for word in ['move', 'drift', 'wander', 'gravity']):
            if 'movement' not in contexts:
                contexts.append('movement')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_contexts = [x for x in contexts if not (x in seen or seen.add(x))]
        
        # Limit to exactly 3 contexts (matching our LLM selector)
        if len(unique_contexts) > 3:
            unique_contexts = unique_contexts[:3]
        
        logger.info(f"Selected refinement contexts: {unique_contexts}")
        return unique_contexts
    
    def get_available_contexts(self) -> Dict[str, str]:
        """Get all available contexts with descriptions."""
        return self.available_contexts.copy()