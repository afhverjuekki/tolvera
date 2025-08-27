import time
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .data_models import (
    BehaviorSynthesisResponse,
    StateDefinition,
    SpeciesConfiguration,
    IntegrationKernel
)
from ..debug.tracing import get_collector, LLMCallData
from .species_manager import SpeciesManager, SpeciesInfo
from .color_resolver import ColorResolver
from ..prompts.prompt_loader import get_prompt_loader


class TaichiCodeResponse(BaseModel):
    """Response structure for Taichi code generation."""
    name: str
    description: str
    code: str
    is_interaction: bool = False
    helper_functions: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Helper functions generated alongside the expert"
    )


class UtilityExpertResponse(BaseModel):
    """Response structure for utility expert generation."""
    name: str = Field(description="Name of the utility function")
    description: str = Field(description="What this utility function does")
    code: str = Field(description="Complete @ti.func code without particle parameters")
    utility_type: str = Field(description="Type: temporal_update, state_update, helper, etc.")
    returns_value: bool = Field(default=False, description="Whether the function returns a value")


class StateField(BaseModel):
    """State field specification for analysis."""
    name: str = Field(description="Name of the state field")
    category: str = Field(description="Category: 'global', 'particle', or 'species'")
    type: str = Field(description="Taichi type (e.g., 'ti.f32', 'ti.i32', 'ti.math.vec2')")
    min: Optional[float] = Field(default=None, description="Minimum value for numeric types")
    max: Optional[float] = Field(default=None, description="Maximum value for numeric types")
    description: str = Field(description="What this state represents")
    initial: Optional[float] = Field(default=None, description="Initial value")


class StateAnalysisResponse(BaseModel):
    """Response structure for state analysis."""
    needs_states: bool = Field(description="Whether this behavior requires custom states")
    states: List[StateField] = Field(default_factory=list, description="List of states needed")


class CodeGenerator:
    """
    Generates Taichi code from natural language behavior descriptions.
    
    This class coordinates between various specialized components to transform
    natural language into executable Taichi code for particle behaviors.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        tolvera_instance=None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the synthesizer.
        
        Args:
            model_name: Name of the LLM model to use
            tolvera_instance: Tolvera instance for particle system access
            api_key: Optional API key for the model provider
        """
        self._load_env()
        
        self.model_name = model_name
        self.tv = tolvera_instance
        
        # Initialize model
        from .llm_factory import ModelFactory
        self.model = ModelFactory.create_model(model_name, api_key)
        self.provider = ModelFactory.get_provider_for_model(model_name)
        
        # Initialize components
        self.species_manager = SpeciesManager(tolvera_instance) if tolvera_instance else None
        self.color_resolver = ColorResolver()
        self.prompt_loader = get_prompt_loader()
        
        # Initialize template renderer for code generation
        from ..templates.template_renderer import TemplateRenderer
        self.template_renderer = TemplateRenderer()
    
    def _load_env(self):
        """Load environment variables from .env file."""
        from pathlib import Path
        from dotenv import load_dotenv
        
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
        description: str,
        expert_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze what states are needed for a behavior.
        
        Args:
            description: Natural language behavior description
            expert_type: Type of expert being analyzed
            
        Returns:
            Dictionary of states organized by category
        """
        
        collector = get_collector()
        
        # Get expert type guidance from prompt loader
        expert_type_guidance = self.prompt_loader.get_expert_type_guidance(expert_type)
        
        # Load and assemble prompts
        prompt = self.prompt_loader.load_prompt(
            "synthesis/state_analysis_user.txt",
            description=description,
            expert_type=expert_type,
            expert_type_guidance=expert_type_guidance
        )
        
        system_prompt_parts = [
            self.prompt_loader.load_prompt("synthesis/state_analysis_system.txt"),
            self.prompt_loader.load_prompt("synthesis/state_analysis_examples.txt"),
            self.prompt_loader.load_prompt("synthesis/state_analysis_criteria.txt", 
                                          expert_type_guidance=expert_type_guidance)
        ]
        system_prompt = "\n\n".join(system_prompt_parts)
        
        # Create agent for state analysis
        agent = Agent(
            self.model,
            output_type=StateAnalysisResponse,
            system_prompt=system_prompt
        )
        
        # Perform analysis with tracing
        with collector.trace_node("state_analysis", "state_analysis", 
                                 description=description) as analysis_node:
            try:
                with collector.trace_node("llm_state_analysis", "llm_call",
                                         model=self.model_name) as llm_node:
                    result = await agent.run(prompt)
                    state_analysis = result.output
                    
                    # Log LLM call details
                    if llm_node:
                        llm_data = self._create_llm_data(
                            system_prompt, prompt, result.output,
                            {
                                "needs_states": state_analysis.needs_states,
                                "states": [state.model_dump() for state in state_analysis.states]
                            }
                        )
                        llm_node.llm_call = llm_data
                
                # Process results
                states_dict = self._process_state_analysis(state_analysis)
                
                # Update trace node
                if analysis_node:
                    analysis_node.output_data = {
                        "needs_states": state_analysis.needs_states,
                        "global_states": len(states_dict['global']),
                        "particle_states": len(states_dict['particle']),
                        "species_states": len(states_dict['species'])
                    }
                
                return states_dict
                
            except Exception:
                return {'global': {}, 'particle': {}, 'species': {}}
    
    def _process_state_analysis(self, state_analysis: StateAnalysisResponse) -> Dict[str, Any]:
        """
        Process state analysis response into structured dictionary.
        
        Args:
            state_analysis: Analysis response from LLM
            
        Returns:
            Dictionary of states by category
        """
        states_dict = {'global': {}, 'particle': {}, 'species': {}}
        
        if not state_analysis.needs_states or not state_analysis.states:
            return states_dict
        
        # Built-in particle properties to skip
        BUILTIN_PROPS = {'pos', 'vel', 'mass', 'size', 'speed', 'species', 'active', 'ppos', 'pvel'}
        
        for state_info in state_analysis.states:
            category = state_info.category
            state_name = state_info.name
            
            # Skip built-in particle properties
            if category == 'particle' and state_name.lower() in BUILTIN_PROPS:
                continue
            
            states_dict[category][state_name] = StateDefinition(
                name=state_name,
                category=category,
                type=state_info.type,
                min=state_info.min if state_info.min is not None else 0.0,
                max=state_info.max if state_info.max is not None else 1.0,
                description=state_info.description,
                initial=state_info.initial
            )
        
        
        return states_dict
    
    async def synthesize_behavior(
        self,
        description: str,
        available_states: Optional[Dict[str, List[str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        expert_name: Optional[str] = None,
        skip_state_analysis: bool = False
    ) -> BehaviorSynthesisResponse:
        """
        Synthesize particle behavior from natural language description.
        
        Args:
            description: Natural language behavior description
            available_states: Dictionary of available states by category
            context: Additional context for synthesis
            expert_name: Optional specific name for the expert function
            skip_state_analysis: Whether to skip state analysis
            
        Returns:
            BehaviorSynthesisResponse with synthesized expert code
        """
        
        # Prepare synthesis
        states_analysis = await self._prepare_synthesis(
            description, skip_state_analysis, context
        )
        available_states = self._update_available_states(
            available_states, states_analysis
        )
        # Get species info from context or analyze if not provided
        species_info = context.get('species_info') if context else None
        if not species_info and self.species_manager:
            # Fallback only if orchestrator didn't provide it
            species_info = self.species_manager.analyze_description(description)
        
        # Build prompts
        system_prompt, user_prompt = await self._build_prompts(
            description, available_states, context, expert_name, species_info
        )
        
        # Execute synthesis
        expert_data = await self._execute_synthesis(
            system_prompt, user_prompt, description, expert_name
        )
        
        # Process response
        return self._create_response(
            expert_data, species_info, states_analysis, context
        )
    
    async def _prepare_synthesis(
        self,
        description: str,
        skip_state_analysis: bool,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare for synthesis by analyzing states if needed."""
        if skip_state_analysis or (context and context.get('states_already_created')):
            return {'global': {}, 'particle': {}, 'species': {}}
        
        return await self.analyze_states_needed(description)
    
    def _update_available_states(
        self,
        available_states: Optional[Dict[str, List[str]]],
        states_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Update available states with analyzed states."""
        if available_states is None:
            available_states = {'global': [], 'particle': [], 'species': [], 'temporal': []}
        else:
            # Ensure all categories exist
            for category in ['global', 'particle', 'species', 'temporal']:
                if category not in available_states:
                    available_states[category] = []
        
        # Add analyzed states
        for category in ['global', 'particle', 'species']:
            if category in states_analysis:
                for state_name in states_analysis[category].keys():
                    if state_name not in available_states[category]:
                        available_states[category].append(state_name)
        
        return available_states
    
    
    async def _build_prompts(
        self,
        description: str,
        available_states: Dict[str, List[str]],
        context: Optional[Dict[str, Any]],
        expert_name: Optional[str],
        species_info: Optional[SpeciesInfo]
    ) -> tuple[str, str]:
        """Build system and user prompts for synthesis."""
        # Get species context
        species_context = ""
        if self.species_manager and species_info:
            species_context = self.species_manager.get_species_context_for_prompts(species_info)
        
        # Build base prompt with dynamic context selection
        base_prompt = await self.prompt_loader.build_prompt_with_dynamic_context(
            description=description,
            expert_type=context.get('expert_type', 'force') if context else 'force',
            available_states=available_states,
            additional_context=context
        )
        
        # Add species context
        system_prompt = base_prompt
        if species_context:
            system_prompt += f"\n\n## SPECIES CONTEXT\n{species_context}\n"
        
        # Add requirements and rules
        species_count = species_info.total_count - 1 if species_info else 0
        expert_requirements = self.prompt_loader.load_prompt(
            "synthesis/expert_synthesis_requirements.txt",
            species_count=species_count
        )
        taichi_rules = self.prompt_loader.load_prompt("synthesis/taichi_critical_rules.txt")
        system_prompt += f"\n\n{expert_requirements}\n\n{taichi_rules}"
        
        # Add specific expert name requirement if provided
        if expert_name:
            system_prompt += f"\n\nIMPORTANT: The function MUST be named '{expert_name}' exactly."
        
        # Build user prompt
        user_prompt = self._build_user_prompt(description, expert_name, context)
        
        # Apply context enhancements
        system_prompt = self._enhance_prompt_with_context(
            system_prompt, context, available_states, species_info
        )
        
        return system_prompt, user_prompt
    
    def _build_user_prompt(
        self,
        description: str,
        expert_name: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the user prompt for synthesis."""
        if expert_name:
            user_prompt = f'Generate expert function named "{expert_name}" for: "{description}"'
        else:
            user_prompt = f'Generate expert function for: "{description}"'
        
        # Add behavioral guidance if provided
        if context:
            if context.get('component_behavioral_guidance'):
                user_prompt += f"\n\nBehavioral guidance: {context['component_behavioral_guidance']}"
            elif context.get('component_implementation'):
                user_prompt += f"\n\nBehavioral guidance: {context['component_implementation']}"
        
        return user_prompt
    
    def _enhance_prompt_with_context(
        self,
        system_prompt: str,
        context: Optional[Dict[str, Any]],
        available_states: Dict[str, List[str]],
        species_info: Optional[SpeciesInfo]
    ) -> str:
        """Enhance prompt with additional context information."""
        if not context:
            return system_prompt
        
        prompt_additions = []
        
        # Check for visual expert
        component = context.get('component')
        if component and hasattr(component, 'expert_type') and component.expert_type == 'visual':
            # Load visual expert additions template
            visual_params = self._get_visual_expert_prompt_params(available_states)
            visual_additions = self.prompt_loader.load_prompt(
                "synthesis/visual_expert_additions.txt",
                **visual_params
            )
            prompt_additions.append(visual_additions)
        
        # Handle species-specific experts
        applies_to = self._get_species_application(context, species_info)
        if applies_to is not None:
            # Load species-specific additions template
            if len(applies_to) == 1:
                species_msg = f"CRITICAL: This expert is specifically for species {applies_to[0]}.\nDO NOT check species inside the expert function!"
            else:
                species_msg = f"DETECTED: This behavior is for species {applies_to} specifically."
            
            species_additions = self.prompt_loader.load_prompt(
                "synthesis/species_specific_additions.txt",
                species_specific_message=species_msg,
                species_detection_message=''
            )
            prompt_additions.append(species_additions)
        
        # Add context elements
        context_params = self._get_context_prompt_params(context, available_states)
        context_additions = self.prompt_loader.load_prompt(
            "synthesis/context_additions.txt",
            **context_params
        )
        prompt_additions.append(context_additions)
        
        return system_prompt + '\n\n'.join(prompt_additions)
    
    def _get_visual_expert_prompt_params(self, available_states: Dict[str, List[str]]) -> Dict[str, str]:
        """Get parameters for visual expert prompt template."""
        from ..context.drawing_patterns import DRAWING_API_REFERENCE
        
        # Build state access instructions
        state_lines = []
        if available_states.get('global'):
            state_lines.append(f"\nGlobal states: {', '.join(available_states['global'])}")
            state_lines.append("Access with: tv.s.llm_global.field[0].STATE_NAME")
        if available_states.get('particle'):
            state_lines.append(f"\nParticle states: {', '.join(available_states['particle'])}")
            state_lines.append("Access with: tv.s.llm_particle.field[particle_idx].STATE_NAME")
        if not any(available_states.values()):
            state_lines.append("\nNO CUSTOM STATES AVAILABLE - use frame counter for animations")
        
        return {
            'drawing_api_reference': DRAWING_API_REFERENCE,
            'state_access_instructions': '\n'.join(state_lines)
        }
    
    def _get_species_application(
        self,
        context: Optional[Dict[str, Any]],
        species_info: Optional[SpeciesInfo]
    ) -> Optional[List[int]]:
        """Determine which species an expert applies to."""
        if not context:
            return None
        
        # Check component specification
        component = context.get('component')
        if component and hasattr(component, 'applies_to_species'):
            return component.applies_to_species
        
        # The decomposer should be the single source of truth for species application
        return None
    
    def _get_context_prompt_params(
        self,
        context: Dict[str, Any],
        available_states: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """Get parameters for context additions prompt template."""
        sections = {}
        
        # Helper functions section
        if context.get('synthesized_helpers'):
            helper_parts = ["## AVAILABLE HELPER FUNCTIONS FROM PREVIOUS EXPERTS",
                          "The following helper functions have already been synthesized and are available:"]
            for helper_name, helper_code in context['synthesized_helpers'].items():
                helper_parts.append(f"\n### {helper_name}:\n```python\n{helper_code}\n```")
            helper_parts.append("\nYou can CALL these existing helpers in your expert.")
            sections['helper_functions_section'] = '\n'.join(helper_parts)
        else:
            sections['helper_functions_section'] = ''
        
        # Pattern type section
        if context.get('pattern_type'):
            sections['pattern_type_section'] = f"\n## PATTERN TYPE\nThis behavior is part of a {context['pattern_type']} pattern."
        else:
            sections['pattern_type_section'] = ''
        
        # Constraints section  
        if context.get('constraints'):
            constraint_lines = ["## CONTEXT CONSTRAINTS", "The following constraints MUST be respected:"]
            constraint_lines.extend(f"- {c}" for c in context['constraints'])
            sections['constraints_section'] = '\n'.join(constraint_lines)
        else:
            sections['constraints_section'] = ''
        
        # Previous experts section
        if context.get('previous_experts'):
            prev_lines = ["## PREVIOUS EXPERTS", "The following experts have already been implemented:"]
            prev_lines.extend(f"- {prev['name']} ({prev['type']})" for prev in context['previous_experts'])
            sections['previous_experts_section'] = '\n'.join(prev_lines)
        else:
            sections['previous_experts_section'] = ''
        
        # Shared parameters section
        if context.get('shared_parameters'):
            param_lines = ["## SHARED PARAMETERS", "Use these parameter values consistently:"]
            param_lines.extend(f"- {param}: {value}" for param, value in context['shared_parameters'].items())
            sections['shared_parameters_section'] = '\n'.join(param_lines)
        else:
            sections['shared_parameters_section'] = ''
        
        # State usage section
        if any(available_states.values()):
            state_lines = ["## IMPORTANT: USE THE AVAILABLE STATES",
                          "The following states are available and MUST be used:"]
            if available_states.get('global'):
                state_lines.append(f"\nGlobal states: {', '.join(available_states['global'])}")
                state_lines.append("Access with: tv.s.llm_global.field[0].STATE_NAME")
            if available_states.get('particle'):
                state_lines.append(f"\nParticle states: {', '.join(available_states['particle'])}")
                state_lines.append("Access with: tv.s.llm_particle.field[particle_idx].STATE_NAME")
            if available_states.get('species'):
                state_lines.append(f"\nSpecies states: {', '.join(available_states['species'])}")
                state_lines.append("Access with: tv.s.llm_species.field[species].STATE_NAME")
            sections['state_usage_section'] = '\n'.join(state_lines)
        else:
            sections['state_usage_section'] = ''
        
        return sections
    
    async def _execute_synthesis(
        self,
        system_prompt: str,
        user_prompt: str,
        description: str,
        expert_name: Optional[str]
    ) -> TaichiCodeResponse:
        """Execute the synthesis with the LLM."""
        collector = get_collector()
        
        # Create agent
        agent = Agent(
            self.model,
            output_type=TaichiCodeResponse,
            system_prompt=system_prompt
        )
        
        # Execute with tracing
        with collector.trace_node("llm_synthesis", "llm_call", 
                                 description=description,
                                 expert_name=expert_name or "",
                                 model=self.model_name) as llm_node:
            
            # Create LLM call data
            llm_data = LLMCallData(
                model=self.model_name,
                provider=self.provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                full_prompt=f"System: {system_prompt}\n\nUser: {user_prompt}"
            )
            
            try:
                start_time = time.time()
                result = await agent.run(user_prompt)
                api_duration = (time.time() - start_time) * 1000
                
                expert_data = result.output
                
                # Update trace
                llm_data.api_call_ms = api_duration
                llm_data.raw_response = str(result)
                llm_data.parsed_response = {
                    "name": expert_data.name,
                    "description": expert_data.description,
                    "code": expert_data.code,
                    "is_interaction": getattr(expert_data, 'is_interaction', False)
                }
                
                if llm_node:
                    llm_node.llm_call = llm_data
                    llm_node.output_data = {
                        "expert_name": expert_data.name,
                        "code_length": len(expert_data.code),
                        "is_interaction": getattr(expert_data, 'is_interaction', False)
                    }
                
                return expert_data
                
            except Exception as e:
                if llm_node:
                    llm_node.error = str(e)
                    llm_node.status = "error"
                raise
    
    def _create_response(
        self,
        expert_data: TaichiCodeResponse,
        species_info: Optional[SpeciesInfo],
        states_analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> BehaviorSynthesisResponse:
        """Create the synthesis response."""
        # Create expert function
        expert_func = self._create_expert_function(
            name=expert_data.name,
            description=expert_data.description,
            code=expert_data.code,
            is_interaction=getattr(expert_data, 'is_interaction', False),
            weight=1.0,
            applies_to_species=self._get_species_application(context, species_info)
        )
        
        # Create integration kernel
        kernel = IntegrationKernel(
            single_experts=[expert_data.name] if not expert_func.is_interaction else [],
            interaction_experts=[expert_data.name] if expert_func.is_interaction else []
        )
        
        # Create species configuration
        species_config = self._create_species_config(species_info)
        
        # Create states list
        states_needed_list = []
        for category in ['global', 'particle', 'species']:
            if category in states_analysis:
                for state_def in states_analysis[category].values():
                    states_needed_list.append(state_def)
        
        # Create response
        response = BehaviorSynthesisResponse(
            experts=[expert_func],
            states_needed=states_needed_list,
            species_config=species_config,
            integration_kernel=kernel,
            temporal_update=None
        )
        
        # Add helper functions if generated
        if hasattr(expert_data, 'helper_functions') and expert_data.helper_functions:
            response.helper_functions = expert_data.helper_functions
        
        return response
    
    def _create_expert_function(
        self,
        name: str,
        description: str,
        code: str,
        is_interaction: bool = False,
        weight: float = 1.0,
        applies_to_species: Optional[List[int]] = None
    ) -> Any:
        """Create an ExpertFunction with pre-generated code."""
        from .data_models import ExpertFunction, ForceComputation, VectorExpression
        
        force_expr = VectorExpression(x="0.0", y="0.0")
        computation = ForceComputation(
            base_force=force_expr,
            description=description
        )
        
        return ExpertFunction(
            name=name,
            description=description,
            is_interaction=is_interaction,
            computation=computation,
            weight=weight,
            applies_to_species=applies_to_species,
            code=code  # Store the pre-generated code
        )
    
    def _create_species_config(self, species_info: Optional[SpeciesInfo]) -> SpeciesConfiguration:
        """Create species configuration from species info."""
        from .data_models import SpeciesConfiguration, SpeciesNameMapping, SpeciesBehaviorMapping, SpeciesColorMapping
        
        # Handle None species_info
        if not species_info:
            return SpeciesConfiguration(species_ids=[0])
        
        # Convert species names
        species_names_list = None
        if species_info.species_names:
            species_names_list = [
                SpeciesNameMapping(species_id=sid, name=name)
                for sid, name in species_info.species_names.items()
            ]
        
        # Convert species behaviors
        species_behaviors_list = None
        if species_info.species_behaviors:
            species_behaviors_list = [
                SpeciesBehaviorMapping(species_id=sid, behaviors=behaviors)
                for sid, behaviors in species_info.species_behaviors.items()
            ]
        
        # Create colors
        colors_list = [
            SpeciesColorMapping(
                species_id=i,
                rgba=self.color_resolver.get_species_color(i, species_info)
            )
            for i in range(species_info.total_count)
        ]
        
        return SpeciesConfiguration(
            species_ids=species_info.species_ids if species_info.species_ids else list(range(species_info.total_count)),
            species_names=species_names_list,
            interaction_pairs=species_info.interaction_pairs,
            species_behaviors=species_behaviors_list,
            requires_all_species=species_info.requires_all_species,
            colors=colors_list
        )
    
    def _create_llm_data(
        self,
        system_prompt: str,
        user_prompt: str,
        raw_response: Any,
        parsed_response: Dict[str, Any]
    ) -> LLMCallData:
        """Create LLM call data for tracing."""
        return LLMCallData(
            model=self.model_name,
            provider=self.provider,
            system_prompt=system_prompt.strip(),
            user_prompt=user_prompt,
            full_prompt=f"{system_prompt}\n\n{user_prompt}",
            raw_response=str(raw_response),
            parsed_response=parsed_response
        )
    
    
    async def synthesize_utility_expert(
        self,
        description: str,
        available_states: Dict[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
        expert_type: str = 'utility'
    ) -> BehaviorSynthesisResponse:
        """
        Synthesize a utility expert function that doesn't take particle parameters.
        
        Args:
            description: Natural language description of the utility
            available_states: Dictionary of available states
            context: Additional context for synthesis
            expert_type: Type of utility expert
            
        Returns:
            BehaviorSynthesisResponse with synthesized utility code
        """
        
        # Format available states
        state_lines = []
        if 'particle' in available_states and available_states['particle']:
            state_lines.append(f"Particle states: {', '.join(available_states['particle'])}")
        if 'global' in available_states and available_states['global']:
            state_lines.append(f"Global states: {', '.join(available_states['global'])}")
        if 'species' in available_states and available_states['species']:
            state_lines.append(f"Species states: {', '.join(available_states['species'])}")
        state_info = "\n".join(state_lines) if state_lines else "No custom states available."
        
        # Get expert-type specific guidance
        expert_guidance = self.prompt_loader.get_expert_type_guidance(expert_type)
        
        # Load prompts
        system_prompt = self.prompt_loader.load_prompt(
            "synthesis/utility_expert_system.txt",
            expert_type=expert_type,
            expert_guidance=expert_guidance,
            state_info=state_info
        )
        
        user_prompt = self.prompt_loader.load_prompt(
            "synthesis/utility_expert_user.txt",
            description=description
        )
        
        # Create agent for utility expert synthesis
        agent = Agent(
            self.model,
            output_type=UtilityExpertResponse,
            system_prompt=system_prompt
        )
        
        # Synthesize utility expert
        result = await agent.run(user_prompt)
        utility_response = result.output
        
        # Convert UtilityExpertResponse to BehaviorSynthesisResponse
        # Create a dummy expert function for compatibility
        expert_func = self._create_expert_function(
            name=utility_response.name,
            description=utility_response.description,
            code=utility_response.code,
            is_interaction=False,
            weight=1.0
        )
        
        return BehaviorSynthesisResponse(
            experts=[expert_func],
            states_needed=[],
            species_config=SpeciesConfiguration(species_ids=[0]),
            integration_kernel=IntegrationKernel(single_experts=[], interaction_experts=[]),
            temporal_update=None
        )