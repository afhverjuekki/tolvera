
import os
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

from .models import (
    BehaviorSynthesisResponse,
    ExpertFunction,
    StateDefinition,
    SpeciesConfiguration,
    ForceComputation,
    VectorExpression,
    IntegrationKernel
)
from ..debug.tracing import get_collector, LLMCallData
from .species_analyzer import SpeciesAnalyzer
from .species_manager import SpeciesManager
from .prompts import ContextAwarePromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class Expert:
    name: str
    description: str
    code: str
    is_interaction: bool = False
    weight: float = 1.0
    
    def to_code(self) -> str:
        return self.code


class TaichiCodeResponse(BaseModel):
    name: str
    description: str
    code: str
    is_interaction: bool = False


class StateField(BaseModel):
    name: str = Field(description="Name of the state field")
    type: str = Field(description="Taichi type (e.g., 'ti.f32', 'ti.i32', 'ti.math.vec2')")
    min: Optional[float] = Field(None, description="Minimum value for numeric types")
    max: Optional[float] = Field(None, description="Maximum value for numeric types")
    description: str = Field(description="What this state represents")
    initial: Optional[float] = Field(None, description="Initial value")


class TemporalUpdateInfo(BaseModel):
    state_name: str = Field(description="Name of the state to update")
    update_expression: str = Field(description="Taichi expression for updating the state over time")
    description: str = Field(description="Description of the temporal behavior")


class StateAnalysisResponse(BaseModel):
    needs_states: bool = Field(description="Whether this behavior requires custom states")
    global_states: List[StateField] = Field(default_factory=list, description="Global states needed")
    particle_states: List[StateField] = Field(default_factory=list, description="Per-particle states needed")
    species_states: List[StateField] = Field(default_factory=list, description="Per-species states needed")
    temporal_updates: List[TemporalUpdateInfo] = Field(default_factory=list, description="Temporal state updates needed")


class Synthesizer:
    
    def __init__(
        self, 
        model_name: str = "gemini-2.0-flash-exp", 
        tolvera_instance=None,
        api_key: Optional[str] = None
    ):
        # Load environment variables
        self._load_env()
        
        # Get API key
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")
        
        self.model_name = model_name
        self.tv = tolvera_instance
        
        # Set key for Gemini
        os.environ['GEMINI_API_KEY'] = api_key
        self.model = GeminiModel(model_name)
        
        # Initialize species components
        self.species_analyzer = SpeciesAnalyzer()
        self.species_manager = SpeciesManager(tolvera_instance) if tolvera_instance else None
        
        # Initialize prompt builder
        self.prompt_builder = ContextAwarePromptBuilder()
        
        # Keep track of experts
        self.experts = []
        
    def _load_env(self):
        env_paths = [
            Path.cwd() / ".env",
            Path(__file__).parent.parent.parent.parent.parent / ".env",
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                break
    
    async def analyze_states_needed(
        self,
        description: str
    ) -> Dict[str, Any]:
        logger.info(f"Analyzing states needed for: {description}")
        
        # Get trace collector
        collector = get_collector()
        
        # Build state analysis prompt
        prompt = self.prompt_builder.build_state_analysis_prompt(description)
        
        # Create structured agent for state analysis
        agent = Agent(
            self.model,
            output_type=StateAnalysisResponse,
            system_prompt="""You are an expert at analyzing particle behaviors and determining what states they need.

CRITICAL: The following properties are ALREADY AVAILABLE on every particle and must NOT be recreated:
- pos, vel (position, velocity) - ti.math.vec2
- mass - ti.f32  
- size - ti.f32
- speed - ti.f32
- species - ti.i32
- active - ti.f32
- ppos, pvel (previous position/velocity) - ti.math.vec2

DO NOT create states for any of these existing properties!

Analyze the behavior and determine:
1. What global states are needed (system-wide parameters like gravity strength, time of day)
2. What particle states are needed (per-particle data like energy, home position - but NOT mass, pos, vel, etc.)
3. What species states are needed (per-species configuration)
4. What temporal updates are needed (states that change over time)

For physics-related states, use proper ranges:
- Gravity: 0.0 to 1000.0 (initial: 300.0)
- Forces: 0.0 to 1000.0 
- Energy: 0.0 to 100.0 (initial: 80.0)
- Time/phase: 0.0 to 1.0

For temporal behaviors like "particles lose energy over time", "fade gradually", or "day/night cycles":
- Create the appropriate state (e.g., energy as particle state, time_of_day as global state)
- Include a temporal_update entry describing how it changes over time

For temporal updates, the update_expression should be a valid Taichi expression that computes the new value.
Examples:
- For energy decay: "energy * 0.99" (multiplies current energy by 0.99)
- For time cycles: "(frame / 3600.0) % 1.0" (cycles every 3600 frames)  
- For linear decrease: "energy - 0.01" (decreases by 0.01 per frame)
- For clamped increase: "min(1.0, charge + 0.1)" (increases but caps at 1.0)

IMPORTANT: The update expression should be the RIGHT-HAND SIDE of an assignment only.
Do NOT include the full path like "tv.s.llm_particle.field[i].energy"
Just provide the expression like "energy * 0.99" or "(frame / 100.0) % 1.0"

Return a structured response with the states organized by category.
For each state, specify the Taichi type, min/max values if applicable, and a clear description.
For temporal updates, provide the state name and a VALID update expression that can be assigned."""
        )
        
        # Create trace node for state analysis
        with collector.trace_node("state_analysis", "state_analysis", 
                                 description=description) as analysis_node:
            try:
                # Create nested LLM call node
                with collector.trace_node("llm_state_analysis", "llm_call",
                                         model=self.model_name) as llm_node:
                    # Run state analysis with structured output
                    result = await agent.run(prompt)
                    state_analysis = result.output
                    
                    # Log LLM call details
                    if llm_node:
                        # Create LLM data
                        from ..debug.tracing import LLMCallData
                        if hasattr(agent, '_system_prompt'):
                            system_prompt = agent._system_prompt
                        elif hasattr(agent, 'system_prompt'):
                            system_prompt = str(agent.system_prompt)
                        else:
                            system_prompt = "State analysis prompt"
                        full_prompt = f"{system_prompt}\n\n{prompt}"
                        llm_data = LLMCallData(
                            model=self.model_name,
                            provider="gemini",
                            system_prompt=system_prompt,
                            user_prompt=prompt,
                            full_prompt=full_prompt,
                            raw_response=str(result.output),
                            parsed_response={
                                "needs_states": state_analysis.needs_states,
                                "global_states": [state.model_dump() for state in state_analysis.global_states],
                                "particle_states": [state.model_dump() for state in state_analysis.particle_states],
                                "species_states": [state.model_dump() for state in state_analysis.species_states],
                                "temporal_updates": [update.model_dump() for update in state_analysis.temporal_updates]
                            }
                        )
                        llm_node.llm_call = llm_data
                
                logger.info(f"State analysis: needs_states={state_analysis.needs_states}, "
                           f"global={len(state_analysis.global_states)}, "
                           f"particle={len(state_analysis.particle_states)}, "
                           f"species={len(state_analysis.species_states)}")
                
                # Update trace node with results
                if analysis_node:
                    analysis_node.output_data = {
                        "needs_states": state_analysis.needs_states,
                        "global_states": len(state_analysis.global_states),
                        "particle_states": len(state_analysis.particle_states),
                        "species_states": len(state_analysis.species_states),
                        "temporal_updates": len(state_analysis.temporal_updates),
                        "temporal_update_details": [
                            {
                                "state": update.state_name,
                                "expression": update.update_expression,
                                "description": update.description
                            }
                            for update in state_analysis.temporal_updates
                        ]
                    }
            
                # Convert to StateDefinition objects
                states_dict = {
                    'global': {},
                    'particle': {},
                    'species': {},
                    'temporal_updates': []
                }
                
                # Process global states
                for state_info in state_analysis.global_states:
                    states_dict['global'][state_info.name] = StateDefinition(
                        name=state_info.name,
                        category='global',
                        type=state_info.type,
                        min=state_info.min if state_info.min is not None else 0.0,
                        max=state_info.max if state_info.max is not None else 1.0,
                        description=state_info.description,
                        initial=state_info.initial
                    )
                
                # Process particle states, filtering out built-in properties
                BUILTIN_PARTICLE_PROPS = {'pos', 'vel', 'mass', 'size', 'speed', 'species', 'active', 'ppos', 'pvel'}
                for state_info in state_analysis.particle_states:
                    if state_info.name.lower() in BUILTIN_PARTICLE_PROPS:
                        logger.warning(f"Skipping built-in particle property '{state_info.name}' from state analysis")
                        continue
                    states_dict['particle'][state_info.name] = StateDefinition(
                        name=state_info.name,
                        category='particle',
                        type=state_info.type,
                        min=state_info.min if state_info.min is not None else 0.0,
                        max=state_info.max if state_info.max is not None else 1.0,
                        description=state_info.description,
                        initial=state_info.initial
                    )
                
                # Process species states
                for state_info in state_analysis.species_states:
                    states_dict['species'][state_info.name] = StateDefinition(
                        name=state_info.name,
                        category='species',
                        type=state_info.type,
                        min=state_info.min if state_info.min is not None else 0.0,
                        max=state_info.max if state_info.max is not None else 1.0,
                        description=state_info.description,
                        initial=state_info.initial
                    )
                
                # Process temporal updates
                states_dict['temporal_updates'] = state_analysis.temporal_updates
                
                logger.info(f"States needed - Global: {len(states_dict['global'])}, "
                           f"Particle: {len(states_dict['particle'])}, "
                           f"Species: {len(states_dict['species'])}, "
                           f"Temporal: {len(states_dict['temporal_updates'])}")
                
                return states_dict
                
            except Exception as e:
                logger.warning(f"State analysis failed: {e}, proceeding without custom states")
                return {'global': {}, 'particle': {}, 'species': {}, 'temporal_updates': []}
    
    async def synthesize_behavior(
        self,
        description: str,
        available_states: Optional[Dict[str, List[str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        expert_name: Optional[str] = None
    ) -> BehaviorSynthesisResponse:
        logger.info(f"Synthesizing: {description}")
        
        # Get trace collector
        collector = get_collector()
        
        # First, analyze what states are needed
        states_analysis = await self.analyze_states_needed(description)
        
        # Flatten state definitions for the response
        states_needed_list = []
        for category in ['global', 'particle', 'species']:
            if category in states_analysis:
                for state_def in states_analysis[category].values():
                    states_needed_list.append(state_def)
        
        # Update available_states to include both existing and new states
        if available_states is None:
            available_states = {'global': [], 'particle': [], 'species': []}
        else:
            # Ensure all categories exist
            for category in ['global', 'particle', 'species']:
                if category not in available_states:
                    available_states[category] = []
        
        # Add analyzed states to available states for prompt
        for category in ['global', 'particle', 'species']:
            if category in states_analysis:
                for state_name in states_analysis[category].keys():
                    if state_name not in available_states[category]:
                        available_states[category].append(state_name)
        
        # Analyze species from description
        species_info = self.species_analyzer.analyze(description)
        logger.info(f"Detected {species_info.total_count} species from description")
        
        # Get species context for prompt
        species_context = ""
        if self.species_manager:
            species_context = self.species_manager.get_species_context_for_prompts(species_info)
        
        # Log available states for debugging
        logger.info(f"Available states for synthesis: {available_states}")
        
        # Build comprehensive prompt using ContextAwarePromptBuilder
        base_prompt = self.prompt_builder.build_synthesis_prompt(
            description=description,
            available_states=available_states,
            constrained=False  # We want direct code generation
        )
        
        # Add species context if available
        if species_context:
            system_prompt = base_prompt + f"\n\n## SPECIES CONTEXT\n{species_context}\n"
        else:
            system_prompt = base_prompt
        
        # Add specific instructions for expert functions
        system_prompt += f"""
## EXPERT FUNCTION REQUIREMENTS
Generate a complete Taichi expert function.
Force experts: (pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2
Interaction experts: (p1: ti.template(), p2: ti.template()) -> ti.math.vec2

Species IDs range from 0 to {species_info.total_count - 1}.

## CRITICAL TAICHI RULE - NEVER RETURN INSIDE CONDITIONALS:
❌ THIS WILL CRASH:
if species != 1:
    return ti.math.vec2(0.0, 0.0)  # CRASH: "Return inside non-static if"

✅ ALWAYS DO THIS INSTEAD:
result = ti.math.vec2(0.0, 0.0)  # Declare result FIRST
if species == 1:
    # Calculate behavior for species 1
    result = calculated_force
# Single return at END
return result"""
        
        # Add specific expert name requirement if provided
        if expert_name:
            system_prompt += f"\n\nIMPORTANT: The function MUST be named '{expert_name}' exactly."
        
        # Add context constraints if provided
        if context and context.get('constraints'):
            system_prompt += "\n\n## CONTEXT CONSTRAINTS\n"
            system_prompt += "The following constraints MUST be respected in your implementation:\n"
            for constraint in context['constraints']:
                system_prompt += f"- {constraint}\n"
        
        # Add previous expert information if provided
        if context and context.get('previous_experts'):
            system_prompt += "\n\n## PREVIOUS EXPERTS\n"
            system_prompt += "The following experts have already been implemented:\n"
            for prev in context['previous_experts']:
                system_prompt += f"- {prev['name']} ({prev['type']}): {prev.get('context', {})}\n"
        
        # Add explicit state usage instructions when states are available
        if any(available_states.values()):
            system_prompt += """

## IMPORTANT: USE THE AVAILABLE STATES
The following states are available and MUST be used in your implementation:
"""
            if available_states.get('global'):
                system_prompt += f"\nGlobal states: {', '.join(available_states['global'])}"
                system_prompt += "\nAccess with: tv.s.llm_global.field[0].state_name"
            if available_states.get('particle'):
                system_prompt += f"\nParticle states: {', '.join(available_states['particle'])}"
                system_prompt += "\nAccess with: tv.s.llm_particle.field[particle_idx].state_name"
            if available_states.get('species'):
                system_prompt += f"\nSpecies states: {', '.join(available_states['species'])}"
                system_prompt += "\nAccess with: tv.s.llm_species.field[species].state_name"
            
            system_prompt += "\n\nYour implementation MUST use these states to implement the requested behavior."""
        
        # Create agent for this synthesis
        agent = Agent(
            self.model,
            output_type=TaichiCodeResponse,
            system_prompt=system_prompt
        )
        
        # User prompt
        if expert_name:
            user_prompt = f'Generate expert function named "{expert_name}" for: "{description}"'
        else:
            user_prompt = f'Generate expert function for: "{description}"'
        
        # Create LLM trace node
        with collector.trace_node("llm_synthesis", "llm_call", 
                                 description=description,
                                 expert_name=expert_name if expert_name else "",
                                 model=self.model_name) as llm_node:
            
            # Create LLM call data
            llm_data = LLMCallData(
                model=self.model_name,
                provider="gemini",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                full_prompt=f"System: {system_prompt}\n\nUser: {user_prompt}"
            )
            
            try:
                # Time the API call
                start_time = time.time()
                result = await agent.run(user_prompt)
                api_duration = (time.time() - start_time) * 1000
                
                expert_data = result.output
                
                # Capture the response
                llm_data.api_call_ms = api_duration
                llm_data.raw_response = str(result)
                llm_data.parsed_response = {
                    "name": expert_data.name,
                    "description": expert_data.description,
                    "code": expert_data.code,
                    "is_interaction": expert_data.is_interaction
                }
                
                # Update trace node if it exists
                if llm_node:
                    llm_node.llm_call = llm_data
                    llm_node.output_data = {
                        "expert_name": expert_data.name,
                        "code_length": len(expert_data.code),
                        "is_interaction": expert_data.is_interaction
                    }
                    # Update metadata for console tracer
                    llm_node.metadata.update({
                        "model": self.model_name,
                        "prompt_length": len(system_prompt + user_prompt),
                        "response_length": len(str(result))
                    })
                
                # Create minimal expert
                expert = Expert(
                    name=expert_data.name,
                    description=expert_data.description,
                    code=expert_data.code,
                    is_interaction=expert_data.is_interaction
                )
                
                # Store it
                self.experts.append(expert)
                
                # Create proper response structure
                # We need to create the ExpertFunction with proper computation
                force_expr = VectorExpression(x="0.0", y="0.0")
                computation = ForceComputation(
                    base_force=force_expr,
                    description=expert.description if hasattr(expert, 'description') else ""
                )
                
                # Create ExpertFunction with custom to_code
                class CustomExpertFunction(ExpertFunction):
                    def __init__(self, expert: Expert):
                        super().__init__(
                            name=expert.name,
                            description=expert.description,
                            is_interaction=expert.is_interaction,
                            computation=computation,
                            weight=expert.weight
                        )
                        self._expert = expert
                    
                    def to_code(self) -> str:
                        return self._expert.code
                
                expert_func = CustomExpertFunction(expert)
                
                # Create integration kernel
                kernel = IntegrationKernel(
                    single_experts=[expert.name] if not expert.is_interaction else [],
                    interaction_experts=[expert.name] if expert.is_interaction else []
                )
                
                # Create species configuration with analyzed data
                species_config = SpeciesConfiguration(
                    species_ids=species_info.species_ids if species_info.species_ids else list(range(species_info.total_count)),
                    species_names=species_info.species_names,
                    interaction_pairs=species_info.interaction_pairs,
                    species_behaviors=species_info.species_behaviors,
                    requires_all_species=species_info.requires_all_species,
                    colors={i: self.species_analyzer.get_color_for_species(i, species_info) 
                           for i in range(species_info.total_count)}
                )
                
                # Create temporal update kernel if needed
                temporal_update = None
                if states_analysis.get('temporal_updates'):
                    from .models import TemporalUpdate
                    # Convert temporal updates to frame updates
                    frame_updates = {}
                    for update in states_analysis['temporal_updates']:
                        frame_updates[update.state_name] = update.update_expression
                    temporal_update = TemporalUpdate(frame_updates=frame_updates)
                
                return BehaviorSynthesisResponse(
                    experts=[expert_func],
                    states_needed=states_needed_list,
                    species_config=species_config,
                    integration_kernel=kernel,
                    temporal_update=temporal_update
                )
                
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                # Update trace node with error if it exists
                if llm_node:
                    llm_node.error = str(e)
                    llm_node.status = "error"
                # Return default
                return self._default_response()
    
    def _default_response(self) -> BehaviorSynthesisResponse:
        
        code = """@ti.func
def drift(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    t = ti.cast(tv.ctx.i[None], ti.f32) * 0.01
    return ti.math.vec2(ti.sin(t) * 30.0, ti.cos(t) * 30.0)"""
        
        expert = Expert(
            name="drift",
            description="Simple drift",
            code=code
        )
        
        # Create proper structure
        force_expr = VectorExpression(x="0.0", y="0.0")
        computation = ForceComputation(expression=force_expr, description="drift")
        
        class CustomExpertFunction(ExpertFunction):
            def __init__(self, expert: Expert):
                super().__init__(
                    name=expert.name,
                    description=expert.description,
                    is_interaction=False,
                    computation=computation,
                    weight=1.0
                )
                self._expert = expert
            
            def to_code(self) -> str:
                return self._expert.code
        
        # Create integration kernel
        kernel = IntegrationKernel(
            single_experts=[expert.name],
            interaction_experts=[]
        )
        
        return BehaviorSynthesisResponse(
            experts=[CustomExpertFunction(expert)],
            states_needed=[],
            species_config=SpeciesConfiguration(species_ids=[0]),
            integration_kernel=kernel
        )
    
