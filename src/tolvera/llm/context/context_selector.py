"""
Intelligent LLM-powered context selector for dynamic prompt building.
Replaces static keyword-based context selection with AI-driven analysis.
"""

import os
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from ..core.llm_factory import ModelFactory
from ..debug.tracing import get_collector


class ContextSelectionResponse(BaseModel):
    """Structured response from context selection."""
    selected_contexts: List[str] = Field(
        description="List of SUPPLEMENTARY context names relevant to the behavior (0-3 contexts, often empty for simple behaviors)"
    )
    reasoning: str = Field(
        description="Brief explanation of why these supplementary contexts were selected (or why none were needed)"
    )


class ContextSelector:
    """LLM-powered intelligent context selector for expert synthesis."""
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the context selector with a fast, lightweight model."""
        if model_name is None:
            model_name = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
        self.model_name = model_name
        self.model = ModelFactory.create_model(model_name, api_key)
        self.provider = ModelFactory.get_provider_for_model(model_name)
        self.system_prompt = ""  # Store system prompt for tracing
        
        # Available contexts - SINGLE SOURCE OF TRUTH
        # Note: Base contexts (core_api, pixels_api, etc.) are always included via get_base_context()
        # The LLM selects from these as supplementary contexts to add
        self.available_contexts = {
            # Core/Base contexts (frequently needed, foundational)
            'core_api': 'Core Tölvera API and particle system fundamentals',
            'pixels_api': 'Pixel manipulation and drawing API',
            'taichi_fundamentals': 'Core Taichi programming patterns and API',
            'taichi_crashes': 'Common Taichi crash fixes and error patterns',
            'state_access': 'State field access patterns and utilities',
            # Behavior patterns
            'boundaries': 'Boundary handling, wrapping, collision detection',
            'movement': 'Basic movement patterns (seek, flee, wander, orbital)',
            'flocking': 'Flocking behaviors (separation, alignment, cohesion)',
            'interaction': 'Particle-particle interactions (chase, repel, attraction)',
            'temporal': 'Time-based behaviors (cycles, energy, aging)',
            'cellular': 'Cellular automata patterns (Game of Life, grid behaviors)',
            'emergent': 'Emergent behaviors (slime mold, ant trails, fireflies)',
            'drawing': 'Visual effects and trail rendering patterns (uses pixels API)',
            'drawing_api': 'Drawing API reference and pixel manipulation',
            # Tölvera-specific patterns
            'vera_patterns': 'Tölvera vera behavior library patterns',
            'vera_interactions': 'Multi-species interaction patterns',
            'species_interactions': 'Species-specific behavior patterns',
            # Artificial life patterns
            'alife_patterns': 'Artificial life computational patterns',
            'iml_patterns': 'Interactive machine learning patterns',
            'evolution': 'Evolutionary algorithms and genetic patterns',
            'ecosystem': 'Ecosystem simulation patterns',
            'morphogenesis': 'Growth and development patterns',
            'swarm': 'Swarm intelligence algorithms',
            # Initialization and configuration
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
        
    
    def _create_selection_agent(self):
        """Create the pydantic-ai agent for context selection."""
        
        # Load the system prompt from external file
        from ..prompts.prompt_loader import get_prompt_loader
        loader = get_prompt_loader()
        self.system_prompt = loader.load_prompt("context_selector/system.txt")

        self.selection_agent = Agent(
            self.model,
            output_type=ContextSelectionResponse,
            output_retries=2,
            system_prompt=self.system_prompt
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
        
        # Load the user prompt template
        from ..prompts.prompt_loader import get_prompt_loader
        loader = get_prompt_loader()
        
        # Build additional context section if needed
        additional_context_section = ""
        if additional_context:
            additional_context_section = f"\n\nADDITIONAL CONTEXT:\n{additional_context}"
        
        prompt = loader.load_prompt(
            "context_selector/user.txt",
            description=description,
            expert_type=expert_type,
            additional_context_section=additional_context_section
        )
        
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
                    # CRITICAL FIX: Set output_data on the LLM node so HTML report can find it
                    # This ensures the context count is available regardless of which node is processed
                    node.output_data = {
                        "selected_contexts": response.selected_contexts,
                        "reasoning": response.reasoning,
                        "context_count": len(response.selected_contexts)
                    }
                    
                    # Log LLM call details
                    # Use the stored system prompt for logging
                    collector.log_llm_call(
                        model=self.model_name,
                        system_prompt=self.system_prompt,
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
                pass
            
            # Update response with only valid contexts
            response.selected_contexts = valid_contexts
            
            
            return response
            
        except Exception as e:
            # Fallback to basic contexts
            fallback_contexts = ['core_api', 'taichi_fundamentals', 'movement']
            return ContextSelectionResponse(
                selected_contexts=fallback_contexts,
                reasoning=f"Fallback selection due to error: {e}",
            )
    
    async def select_contexts_for_refinement(
        self, 
        description: str, 
        sketch_code: str,
        refinement_type: str = "general"
    ) -> List[str]:
        """
        Context selection for sketch refinement using LLM intelligence.
        
        Args:
            description: Refinement description
            sketch_code: Current sketch code
            refinement_type: Type of refinement (error_correction, feature_addition, etc.)
            
        Returns:
            List of relevant context names selected by the LLM
        """
        # Build a more detailed description for the LLM to understand the refinement context
        enhanced_description = f"{description}"
        if refinement_type == "error_correction":
            enhanced_description = f"Error correction: {description}"
        elif refinement_type == "implementation":
            enhanced_description = f"Implementation refinement: {description}"
        
        # Add code snippet context if available
        additional_context = {
            "refinement_type": refinement_type,
            "code_snippet": sketch_code[:500] if sketch_code else None  # First 500 chars of code
        }
        
        # Use the LLM to intelligently select contexts
        result = await self.select_contexts(
            description=enhanced_description,
            expert_type="refinement",
            additional_context=additional_context
        )
        
        # Ensure we only return valid context keys
        valid_contexts = []
        for context in result.selected_contexts:
            if context in self.available_contexts:
                valid_contexts.append(context)
        
        return valid_contexts
    
    def get_available_contexts(self) -> Dict[str, str]:
        """Get all available contexts with descriptions."""
        return self.available_contexts.copy()