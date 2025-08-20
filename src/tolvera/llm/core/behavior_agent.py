
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .synthesizer import Synthesizer, Expert
from .state_manager import StateManager
from .decomposer import BehaviorDecomposer
from .behavior_requirements import BehaviorRequirementsAnalyzer, BehaviorRequirements
from ..debug.tracing import get_collector
from ..generation.kernel import IntegrationKernelGenerator
from ..generation.sketch import SketchGenerator
from .species_manager import SpeciesManager

logger = logging.getLogger(__name__)


@dataclass
class ExpertInfo:
    name: str
    description: str
    weight: float
    expert_type: str  # 'single', 'interaction', 'drawing', 'drawing_interaction', 'utility'
    code: str
    draw_order: Optional[str] = None  # 'pre' or 'post' for drawing experts
    applies_to_species: Optional[List[int]] = None  # Species IDs this expert applies to


class BehaviorAgent:
    
    def _fix_expert_function_name(self, code: str, expected_name: str) -> str:
        import re
        
        pattern = r'@ti\.func\s*\n\s*def\s+(\w+)\s*\('
        
        match = re.search(pattern, code)
        if match:
            current_name = match.group(1)
            if current_name != expected_name:
                code = re.sub(
                    r'(@ti\.func\s*\n\s*def\s+)' + re.escape(current_name) + r'(\s*\()',
                    r'\g<1>' + expected_name + r'\g<2>',
                    code
                )
                logger.info(f"Fixed function name: {current_name} -> {expected_name}")
        
        return code
    
    def __init__(
        self, 
        tolvera_instance, 
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None
    ):
        self.tv = tolvera_instance
        self.model_name = model_name
        
        # Use model factory to determine provider
        from .model_factory import ModelFactory
        self.provider = ModelFactory.get_provider_for_model(model_name)
        logger.info(f"BehaviorAgent using provider '{self.provider}' with model '{model_name}'")
        
        self.synthesizer = Synthesizer(model_name, tolvera_instance, api_key)
        self.state_manager = StateManager(tolvera_instance)
        self.species_manager = SpeciesManager(tolvera_instance)
        self.requirements_analyzer = BehaviorRequirementsAnalyzer()
        
        from .prompts import ContextAwarePromptBuilder
        prompt_builder = ContextAwarePromptBuilder()
        self.decomposer = BehaviorDecomposer(model_name, self.provider, prompt_builder, api_key)
        
        self.kernel_generator = IntegrationKernelGenerator()
        self.sketch_generator = SketchGenerator()
        
        # Initialize SketchRefiner for architectural enhancement
        from .sketch_refiner import SketchRefiner
        self.sketch_refiner = SketchRefiner(model_name, api_key)
        
        self.experts: List[ExpertInfo] = []
        self.expert_weights: Dict[str, float] = {}
        
        self.current_species_config = None
        self.detected_particle_count: Optional[int] = None  # Store particle count from decomposer
        
        self.temporal_updates: List[Any] = []
        
        # Track behavior requirements for holistic synthesis
        self.current_behavior_requirements: Optional[BehaviorRequirements] = None
        self.synthesized_helpers: Dict[str, str] = {}  # Helper functions synthesized by experts
        
        # Track synthesized components (from new LLM synthesis)
        self.synthesized_initialization = None
        self.synthesized_temporal_update = None
        self.synthesized_configuration = None
        
    async def _analyze_states_for_component(self, component) -> List[Dict[str, Any]]:
        """
        Analyze what states a component needs before synthesis.
        
        Args:
            component: BehaviorComponent to analyze
            
        Returns:
            List of state specifications
        """
        # Build a description that captures what this component does
        description = f"{component.description}. {component.implementation}"
        
        # Pass expert type for context-aware state analysis
        # This helps the LLM understand what kind of states are appropriate
        states_analysis = await self.synthesizer.analyze_states_needed(
            description, 
            expert_type=component.expert_type
        )
        
        state_specs = []
        
        # Convert the analysis to state specs
        for category in ['global', 'particle', 'species']:
            if category in states_analysis and states_analysis[category]:
                for state_name, state_def in states_analysis[category].items():
                    spec = {category: {state_name: state_def}}
                    state_specs.append(spec)
                    
                    # Handle temporal updates (now all go to their original category)
                    if hasattr(state_def, 'temporal_update') and state_def.temporal_update:
                        # Register temporal update in the state's own category
                        self.state_manager.register_temporal_update(
                            category,
                            state_name,
                            state_def.temporal_update
                        )
        
        # Process temporal_updates from the analysis
        # Important: Temporal updates are applied to existing states, NOT creating new ones
        if 'temporal_updates' in states_analysis and states_analysis['temporal_updates']:
            for temporal_update in states_analysis['temporal_updates']:
                state_name = temporal_update.state_name
                # Find which category this state belongs to
                found_state = False
                for category in ['global', 'particle', 'species']:
                    if category in states_analysis and state_name in states_analysis[category]:
                        # Register the temporal update for this state in its original category
                        # DO NOT create a duplicate state in temporal category!
                        self.state_manager.register_temporal_update(
                            category,  # Keep in original category
                            state_name,
                            temporal_update.update_expression
                        )
                        found_state = True
                        break
                
                # Check if it's already in temporal (for states that are truly temporal-only)
                if not found_state and 'temporal' in states_analysis and state_name in states_analysis['temporal']:
                    self.state_manager.register_temporal_update(
                        'temporal',
                        state_name,
                        temporal_update.update_expression
                    )
                    found_state = True
                
                if not found_state:
                    # Only create a new temporal state if the state doesn't exist anywhere else
                    # This would be for temporal-only states like counters or timers
                    from ..core.behavior_requirements import StateRequirement
                    temporal_state = StateRequirement(
                        name=state_name,
                        category='temporal',
                        type='ti.f32',
                        min=0.0,
                        max=1.0,
                        description=temporal_update.description,
                        temporal_update={'expression': temporal_update.update_expression} if temporal_update.update_expression else None
                    )
                    spec = {'temporal': {state_name: temporal_state}}
                    state_specs.append(spec)
                    self.state_manager.register_temporal_update(
                        'temporal',
                        state_name,
                        temporal_update.update_expression
                    )
        
        return state_specs
    
    async def synthesize_complete_behavior(self, description: str, weight: float = 1.0, decomposition=None) -> Dict[str, Any]:
        """
        Holistic synthesis method that analyzes requirements upfront and synthesizes with shared context.
        
        Args:
            description: Natural language behavior description
            weight: Weight for this behavior
            decomposition: Optional pre-computed decomposition to avoid double decomposition
            
        Returns:
            Dictionary with synthesis results
        """
        collector = get_collector()
        
        with collector.trace_node("synthesize_complete_behavior", "synthesis", description=description) as node:
            logger.info(f"Starting holistic synthesis for: {description}")
            
            # 1. Use provided decomposition or decompose if not provided
            decomposed = decomposition
            if decomposed is None and self.decomposer:
                try:
                    decomposed = await self.decomposer.decompose(description)
                    logger.info(f"Decomposed behavior: components={len(decomposed.components)}")
                    
                    # Extract particle count if detected
                    if decomposed and hasattr(decomposed, 'particle_count') and decomposed.particle_count:
                        self.detected_particle_count = decomposed.particle_count
                        logger.info(f"Detected particle count: {self.detected_particle_count}")
                    
                    # Extract species configuration from decomposition
                    if decomposed and hasattr(decomposed, 'species_info') and decomposed.species_info:
                        from .models import SpeciesConfiguration
                        from .color_resolver import ColorResolver
                        species_info = decomposed.species_info
                        
                        # Resolve color descriptions to RGBA values
                        resolved_colors = {}
                        if species_info.species_color_descriptions and species_info.species_color_descriptions is not None:
                            # Pass the synthesizer's model to ColorResolver for LLM color resolution
                            color_resolver = ColorResolver(llm_client=self.synthesizer.model)
                            # Now it's a list of SpeciesColorMapping objects or dicts
                            for color_mapping in species_info.species_color_descriptions:
                                # Handle both object and dict formats
                                if hasattr(color_mapping, 'species_id'):
                                    species_id = color_mapping.species_id
                                    color_desc = color_mapping.color_description if hasattr(color_mapping, 'color_description') else 'gray'
                                elif isinstance(color_mapping, dict):
                                    species_id = color_mapping.get('species_id', 0)
                                    color_desc = color_mapping.get('color_description', 'gray')
                                else:
                                    continue  # Skip invalid items
                                    
                                try:
                                    # Use async color resolution if available
                                    resolved_colors[species_id] = await color_resolver.resolve_color_name(color_desc)
                                    logger.info(f"Resolved color '{color_desc}' for species {species_id}")
                                except Exception as e:
                                    logger.warning(f"Failed to resolve color '{color_desc}': {e}")
                                    # Use fallback colors
                                    resolved_colors[species_id] = color_resolver.get_default_species_colors(1)[0]
                        elif hasattr(species_info, 'species_colors') and species_info.species_colors:
                            # Use directly provided RGBA colors if available
                            for color_mapping in species_info.species_colors:
                                resolved_colors[color_mapping.get('species_id', 0)] = color_mapping.get('rgba_values', [0.7, 0.7, 0.7, 1.0])
                        else:
                            # Generate default colors
                            color_resolver = ColorResolver()
                            resolved_colors = color_resolver.get_default_species_colors(species_info.total_count)
                        
                        # Convert species names from list to dict
                        species_names_dict = {}
                        if species_info.species_names and species_info.species_names is not None:
                            for item in species_info.species_names:
                                # Handle both object and dict formats
                                if hasattr(item, 'species_id'):
                                    species_id = item.species_id
                                    name = item.name if hasattr(item, 'name') else 'unnamed'
                                elif isinstance(item, dict):
                                    species_id = item.get('species_id', 0)
                                    name = item.get('name', 'unnamed')
                                else:
                                    continue  # Skip invalid items
                                species_names_dict[species_id] = name

                        # Convert to SpeciesConfiguration format
                        self.current_species_config = SpeciesConfiguration(
                            species_ids=list(range(species_info.total_count)),
                            species_names=species_names_dict,
                            interaction_pairs=getattr(species_info, 'interaction_pairs', None) or [],
                            species_behaviors=None,
                            requires_all_species=False,
                            colors=resolved_colors
                        )
                        logger.info(f"Extracted species config from decomposition: {species_info.total_count} species")
                except Exception as e:
                    logger.warning(f"Decomposition failed: {e}")
                    decomposed = None
            
            # 2. Check for pure drawing behavior from decomposition
            # NOTE: We should NOT bypass synthesis for pure drawing - we still need to create proper experts
            # Pure drawing behaviors should still generate @ti.func experts that return zero force
            if decomposed and hasattr(decomposed, 'behavior_category') and decomposed.behavior_category == 'pure_drawing':
                logger.info("Detected pure drawing behavior - will synthesize drawing expert with zero force")
            
            # 3. Check for uniform speed specification from decomposition
            if decomposed and hasattr(decomposed, 'speed_spec'):
                self.current_speed_spec = decomposed.speed_spec
                logger.info(f"Detected speed specification: uniform={decomposed.speed_spec.uniform}, magnitude={decomposed.speed_spec.magnitude}")
            
            # 4. Analyze requirements with decomposition info
            if decomposed:
                self.current_behavior_requirements = self.requirements_analyzer.analyze(description, decomposed)
            else:
                self.current_behavior_requirements = self.requirements_analyzer.analyze(description)
            logger.info(f"Pattern detected: {self.current_behavior_requirements.pattern_type}")
            
            # 3. Create ALL states upfront (from requirements, decomposer, AND component analysis)
            all_states_specs = []
            temporal_count = 0
            
            # Collect states from requirements analyzer
            if self.current_behavior_requirements.state_requirements:
                for state_req in self.current_behavior_requirements.state_requirements:
                    # Create state spec
                    spec = {state_req.category: {state_req.name: state_req}}
                    all_states_specs.append(spec)
                    
                    # Register temporal update if present
                    if hasattr(state_req, 'temporal_update') and state_req.temporal_update:
                        self.state_manager.register_temporal_update(
                            state_req.category,
                            state_req.name,
                            state_req.temporal_update
                        )
                        temporal_count += 1
                        logger.info(f"Registered temporal update for {state_req.category}.{state_req.name}")
            
            # Collect states from decomposer's suggested_states
            if decomposed and hasattr(decomposed, 'suggested_states') and decomposed.suggested_states:
                logger.info(f"Processing {len(decomposed.suggested_states)} states from decomposer")
                for state_tuple in decomposed.suggested_states:
                    # Each state is (name, category, type, min, max)
                    if len(state_tuple) >= 5:
                        name, category, type_str, min_val, max_val = state_tuple[:5]
                        # Create a StateRequirement-like object
                        from ..core.behavior_requirements import StateRequirement
                        state_req = StateRequirement(
                            name=name,
                            category=category,
                            type=type_str,
                            min=min_val,
                            max=max_val,
                            description=f"State from decomposer: {name}"
                        )
                        spec = {category: {name: state_req}}
                        all_states_specs.append(spec)
                        logger.info(f"Added state from decomposer: {category}.{name} ({type_str}, {min_val}-{max_val})")
            
            # Analyze states needed for individual components BEFORE synthesis
            if decomposed and hasattr(decomposed, 'components'):
                logger.info(f"Analyzing states for {len(decomposed.components)} components upfront")
                for component in decomposed.components:
                    # First, add states from component.required_states
                    if component.required_states:
                        for state_tuple in component.required_states:
                            if len(state_tuple) >= 5:
                                name, category, type_str, min_val, max_val = state_tuple[:5]
                                from ..core.behavior_requirements import StateRequirement
                                state_req = StateRequirement(
                                    name=name,
                                    category=category,
                                    type=type_str,
                                    min=min_val,
                                    max=max_val,
                                    description=f"State for {component.expert_name}"
                                )
                                spec = {category: {name: state_req}}
                                all_states_specs.append(spec)
                                logger.info(f"Added required state from {component.expert_name}: {category}.{name}")
                    
                    # Analyze the component description for extra states
                    # Do this for ALL expert types to get comprehensive state requirements
                    component_states = await self._analyze_states_for_component(component)
                    if component_states:
                        all_states_specs.extend(component_states)
                        logger.info(f"Added {len(component_states)} analyzed states for {component.expert_name} ({component.expert_type})")
            
            # Create all states
            if all_states_specs:
                self.state_manager.collect_and_create_states(all_states_specs)
                logger.info(f"Created {len(all_states_specs)} state specs upfront")
                if temporal_count > 0:
                    logger.info(f"Registered {temporal_count} temporal updates")
            
            # 4. Helper functions will be synthesized by experts as needed
            # No pre-generated templates - each expert creates what it needs
            
            # 5. Build shared context for synthesis
            shared_context = {
                "behavior_requirements": self.current_behavior_requirements,
                "available_states": self.state_manager.get_available_states(),
                "synthesized_helpers": self.synthesized_helpers.copy(),  # Pass already synthesized helpers
                "existing_experts": [e.name for e in self.experts],
                "pattern_type": self.current_behavior_requirements.pattern_type,
                "pattern_confidence": self.current_behavior_requirements.pattern_confidence,
                "shared_parameters": self.current_behavior_requirements.shared_parameters,
                "constraints": self.current_behavior_requirements.constraints,
                "pixel_field": self.current_behavior_requirements.pixel_field,
                "temporal": self.current_behavior_requirements.temporal,
                "states_already_created": True  # Signal states are created upfront
            }
            
            # 6. Synthesize ALL components from decomposer
            if decomposed and decomposed.components:
                logger.info(f"Processing behavior with {len(decomposed.components)} component(s)")
                
                # Synthesize each component with shared context
                components_results = []
                for component in decomposed.components:
                    comp_result = await self._synthesize_component_with_context(
                        component, weight, shared_context
                    )
                    if comp_result is None:
                        logger.error(f"Component synthesis returned None for: {component.expert_name}")
                        comp_result = {
                            'success': False,
                            'experts_added': 0,
                            'expert_names': [],
                            'states_created': 0
                        }
                    components_results.append({
                        'expert_name': component.expert_name,
                        'description': component.description,
                        'result': comp_result
                    })
                
                # Safely calculate total experts, handling None results
                total_experts = 0
                for r in components_results:
                    if r.get('result') and isinstance(r['result'], dict):
                        total_experts += r['result'].get('experts_added', 0)
                
                # Temporal updates are now handled by utility experts - no legacy kernel generation needed
                
                # Generate pixel kernels if needed
                if self.current_behavior_requirements.pixel_field:
                    pixel_kernels = self._generate_pixel_kernels(self.current_behavior_requirements.pixel_field)
                    if pixel_kernels:
                        self.pixel_kernels = pixel_kernels
                        logger.info(f"Generated {len(pixel_kernels)} pixel kernels")
                
                return {
                    'success': True,
                    'pattern_type': self.current_behavior_requirements.pattern_type if self.current_behavior_requirements else 'unknown',
                    'experts_added': total_experts,
                    'components': components_results,
                    'states_created': len(self.current_behavior_requirements.state_requirements) if self.current_behavior_requirements and self.current_behavior_requirements.state_requirements else 0,
                    'helpers_synthesized': len(self.synthesized_helpers),
                    'requirements': self.current_behavior_requirements,
                    'expert_names': [c['expert_name'] for c in components_results]
                }
            else:
                # This should never happen since decomposer always returns at least 1 component
                logger.error(f"Decomposer returned no components for: {description}")
                return {
                    'success': False,
                    'pattern_type': 'unknown',
                    'experts_added': 0,
                    'expert_names': [],
                    'states_created': 0,
                    'helpers_synthesized': 0,
                    'requirements': self.current_behavior_requirements
                }
    
    async def _synthesize_drawing_function(
        self,
        function_name: str,
        description: str, 
        implementation: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Synthesize a drawing function for visual components.
        
        Args:
            function_name: Name for the drawing function
            description: What to draw
            implementation: Implementation guidance
            context: Synthesis context
            
        Returns:
            Taichi drawing function code
        """
        # Use the synthesizer's utility expert capability for drawing
        # This generates @ti.func code suitable for drawing
        drawing_response = await self.synthesizer.synthesize_utility_expert(
            f"{description}. Implementation: {implementation}",
            available_states=context.get('available_states', {}),
            context=context,
            expert_type='drawing'  # Mark as drawing type
        )
        
        if drawing_response and hasattr(drawing_response, 'code'):
            # Ensure the function name matches what we expect
            code = drawing_response.code
            # Replace the function name if needed
            import re
            code = re.sub(r'@ti\.func\s+def\s+\w+\(', f'@ti.func\ndef {function_name}(', code)
            return code
        
        # Fallback: generate simple drawing function
        return f"""@ti.func
def {function_name}():
    \"\"\"Drawing function: {description}\"\"\"
    # Drawing implementation for: {implementation}
    pass  # Placeholder - actual drawing code would go here
"""
    
    async def _synthesize_component_with_context(
        self, 
        component: Any, 
        weight: float,
        shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize a component with shared context."""
        # Add component-specific context with FULL details
        synthesis_context = dict(shared_context)
        synthesis_context['component'] = component
        synthesis_context['component_description'] = component.description
        synthesis_context['component_behavioral_guidance'] = component.implementation
        
        # Add the new detailed fields from enhanced decomposer
        if hasattr(component, 'implementation_details'):
            synthesis_context['implementation_details'] = component.implementation_details
        if hasattr(component, 'parameters'):
            synthesis_context['parameters'] = component.parameters
        
        # Create an enhanced description that includes all implementation guidance
        enhanced_description = f"{component.description}. Behavior: {component.implementation}"
        if hasattr(component, 'implementation_details') and component.implementation_details:
            enhanced_description += f" Implementation details: {'; '.join(component.implementation_details)}"
        
        # Add ALL components to context so each expert knows about others
        if 'all_components' in shared_context:
            synthesis_context['other_components'] = [
                {
                    'name': c.expert_name,
                    'type': c.expert_type,
                    'description': c.description,
                    'implementation': c.implementation,
                    'implementation_details': getattr(c, 'implementation_details', []),
                    'depends_on': getattr(c, 'depends_on', [])
                }
                for c in shared_context['all_components'] 
                if c.expert_name != component.expert_name
            ]
        
        # Add previous experts to context for coordination
        if len(self.experts) > 0:
            synthesis_context['previous_experts'] = [
                {'name': e.name, 'type': e.expert_type, 'description': e.description}
                for e in self.experts
            ]
        
        # Route to appropriate synthesizer method based on expert type
        if component.expert_type == 'visual':
            # Visual components need special handling - generate drawing functions
            from .models import BehaviorSynthesisResponse, ExpertFunction, ForceComputation, VectorExpression, IntegrationKernel, SpeciesConfiguration
            
            # Create drawing function code directly
            drawing_code = await self._synthesize_drawing_function(
                component.expert_name,
                component.description,
                component.implementation,
                synthesis_context
            )
            
            # Create wrapper for drawing function as an expert
            class DrawingExpert:
                def __init__(self, name, description, code):
                    self.name = name
                    self.description = description
                    self.code = code
                    self.is_interaction = False
                    self.weight = 1.0
                
                def to_code(self) -> str:
                    return self.code
            
            drawing_expert = DrawingExpert(
                component.expert_name,
                component.description,
                drawing_code
            )
            
            # Create response structure compatible with expert registration
            force_expr = VectorExpression(x="0.0", y="0.0")
            computation = ForceComputation(base_force=force_expr, description=drawing_expert.description)
            
            class CustomExpertFunction(ExpertFunction):
                def __init__(self, expert):
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
            
            response = BehaviorSynthesisResponse(
                experts=[CustomExpertFunction(drawing_expert)],
                states_needed=[],
                species_config=SpeciesConfiguration(species_ids=[0]),
                integration_kernel=IntegrationKernel(single_experts=[], interaction_experts=[])
            )
            
        elif component.expert_type in ['temporal_update', 'state_update', 'utility']:
            # Use utility expert synthesis for temporal/state updates
            utility_response = await self.synthesizer.synthesize_utility_expert(
                component.implementation,  # Use implementation directly for utility experts
                available_states=shared_context['available_states'],
                context=synthesis_context,
                expert_type=component.expert_type  # Pass the specific expert type!
            )
            
            # States for utility experts are created upfront by decomposer
            
            # Convert utility response to behavior synthesis response format
            from .models import BehaviorSynthesisResponse, ExpertFunction, ForceComputation, VectorExpression, IntegrationKernel, SpeciesConfiguration
            
            # Create minimal expert wrapper for utility response
            class UtilityExpert:
                def __init__(self, utility_resp):
                    self.name = utility_resp.name
                    self.description = utility_resp.description
                    self.code = utility_resp.code
                    self.is_interaction = False
                    self.weight = 1.0
                
                def to_code(self) -> str:
                    return self.code
            
            utility_expert = UtilityExpert(utility_response)
            
            # Create minimal response structure
            force_expr = VectorExpression(x="0.0", y="0.0")
            computation = ForceComputation(base_force=force_expr, description=utility_expert.description)
            
            class CustomExpertFunction(ExpertFunction):
                def __init__(self, expert):
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
            
            response = BehaviorSynthesisResponse(
                experts=[CustomExpertFunction(utility_expert)],
                states_needed=[],
                species_config=SpeciesConfiguration(species_ids=[0]),
                integration_kernel=IntegrationKernel(single_experts=[], interaction_experts=[])
            )
        else:
            # Use regular behavior synthesis for force/interaction experts
            response = await self.synthesizer.synthesize_behavior(
                enhanced_description,  # Use enhanced description with implementation guidance
                shared_context['available_states'],
                context=synthesis_context,
                expert_name=component.expert_name,
                skip_state_analysis=True  # States already created
            )
        
        # Check if synthesis succeeded
        if response is None:
            logger.error(f"Synthesizer returned None for component: {component.expert_name}")
            return {
                'success': False,
                'experts_added': 0,
                'expert_names': [],
                'states_created': 0
            }
        
        if not hasattr(response, 'experts') or not response.experts:
            logger.warning(f"No experts generated for component: {component.expert_name}")
            logger.warning(f"Component description: {component.description}")
            logger.warning(f"Component implementation hint: {component.implementation}")
            # Return empty result
            return {
                'success': False,
                'experts_added': 0,
                'expert_names': [],
                'states_created': 0
            }
        
        # Register expert (no state creation - already done upfront)
        experts_added = []
        for expert in response.experts:
            expert.name = component.expert_name
            code = expert.to_code() if hasattr(expert, 'to_code') else ""
            code = self._fix_expert_function_name(code, component.expert_name)
            
            # Determine expert type based on component
            if component.expert_type == 'visual':
                expert_type = 'drawing'  # Use 'drawing' for consistency with kernel generator
            elif component.expert_type in ['temporal_update', 'state_update', 'utility']:
                expert_type = 'utility'
            elif expert.is_interaction:
                expert_type = 'interaction'
            else:
                expert_type = 'single'
            
            expert_info = ExpertInfo(
                name=component.expert_name,
                description=component.description,
                weight=weight,
                expert_type=expert_type,
                code=code,
                applies_to_species=component.applies_to_species,
                draw_order='post' if expert_type == 'drawing' else None  # Default draw_order for visual experts
            )
            self.experts.append(expert_info)
            self.expert_weights[component.expert_name] = weight
            experts_added.append(component.expert_name)
        
        # Regenerate drawing kernel if we just added a drawing expert
        if expert_type == 'drawing':
            await self._regenerate_drawing_kernel()
            logger.info(f"Regenerated drawing kernel after adding visual component: {component.expert_name}")
        
        return {
            'success': True,
            'experts_added': len(experts_added),
            'expert_names': experts_added,
            'states_created': 0  # States already created upfront
        }
    
    def _merge_species_configs(self, existing, new):
        if not existing:
            return new
        if not new:
            return existing
            
        from .models import SpeciesConfiguration
        
        # Build a mapping of color/name to species ID for existing config
        color_to_id = {}
        if existing.species_names:
            for sid, name in existing.species_names.items():
                # Extract color from name if present
                color = self._extract_color_from_name(name)
                if color:
                    color_to_id[color] = sid
                else:
                    color_to_id[name] = sid
        
        # Process new species
        final_names = dict(existing.species_names) if existing.species_names else {}
        final_behaviors = dict(existing.species_behaviors) if existing.species_behaviors else {}
        max_id = max(existing.species_ids) if existing.species_ids else -1
        new_species_ids = list(existing.species_ids)
        
        if new.species_names:
            for item in new.species_names:
                sid = item.get('species_id', 0)
                name = item.get('name', 'unnamed')
                color = self._extract_color_from_name(name)
                lookup_key = color if color else name
                
                if lookup_key in color_to_id:
                    # This species already exists, update its info
                    existing_id = color_to_id[lookup_key]
                    # Keep the more descriptive name
                    if len(name) > len(final_names.get(existing_id, '')):
                        final_names[existing_id] = name
                else:
                    # This is a new species
                    max_id += 1
                    new_species_ids.append(max_id)
                    final_names[max_id] = name
                    color_to_id[lookup_key] = max_id
                    
                    # Copy behaviors if any
                    if new.species_behaviors and sid in new.species_behaviors:
                        final_behaviors[max_id] = new.species_behaviors[sid]
        
        # Update interaction pairs with remapped IDs
        remapped_pairs = list(existing.interaction_pairs) if existing.interaction_pairs else []
        # (We'd need to implement pair remapping logic here)
        
        # Sort species IDs
        new_species_ids = sorted(list(set(new_species_ids)))
        
        # Merge colors - preserve existing and add new
        final_colors = {}
        if existing.colors:
            final_colors.update(existing.colors)
        if new.colors:
            # Map new colors to the correct species IDs
            for sid, color in new.colors.items():
                if new.species_names:
                    name = ""
                    for item in new.species_names:
                        if item.get('species_id', 0) == sid:
                            name = item.get('name', 'unnamed')
                            break
                    if name:
                        color_key = self._extract_color_from_name(name)
                        lookup_key = color_key if color_key else name
                        if lookup_key in color_to_id:
                            final_colors[color_to_id[lookup_key]] = color
        
        return SpeciesConfiguration(
            species_ids=new_species_ids,
            species_names=final_names if final_names else None,
            interaction_pairs=remapped_pairs,
            species_behaviors=final_behaviors if final_behaviors else None,
            requires_all_species=existing.requires_all_species or new.requires_all_species,
            colors=final_colors if final_colors else None
        )
    
    def _extract_color_from_name(self, name: str) -> Optional[str]:
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'pink', 'brown', 'gray']
        name_lower = name.lower()
        for color in colors:
            if color in name_lower:
                return color
        return None
        
    def get_expert_info(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': e.name,
                'description': e.description,
                'weight': e.weight,
                'expert_type': e.expert_type
            }
            for e in self.experts
        ]

    async def _synthesize_component(
        self,
        component,
        weight: float,
        synthesis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize a single component with context awareness.
        
        Args:
            component: BehaviorComponent to synthesize
            weight: Weight for this component
            synthesis_context: Context from previous components
            
        Returns:
            Dictionary with synthesis results
        """
        # Get current available states
        available_states = self.state_manager.get_available_states()
        
        # Create a synthesis description that includes high-level guidance
        description = f"{component.description}. Behavior: {component.implementation}"
        
        # Add context from previous experts if any
        if synthesis_context.get("previous_experts"):
            description += "\n\nContext from previous experts:"
            for prev in synthesis_context["previous_experts"]:
                description += f"\n- {prev['name']}: {prev.get('context', {})}"
        
        # Add constraints from shared context
        shared_context = synthesis_context.get("shared_context")
        if shared_context and hasattr(shared_context, 'constraints') and shared_context.constraints:
            description += "\n\nConstraints to maintain:"
            for constraint in shared_context.constraints:
                description += f"\n- {constraint}"
        
        # Synthesize with updated description, context, and expert name
        response = await self.synthesizer.synthesize_behavior(
            description,
            available_states,
            context=synthesis_context,
            expert_name=component.expert_name
        )
        
        # Create states if needed
        if response.states_needed:
            states_spec = {'global': {}, 'particle': {}, 'species': {}}
            for state_def in response.states_needed:
                states_spec[state_def.category][state_def.name] = state_def
            self.state_manager.create_states_from_spec(states_spec)
        
        # Register expert with the specific name from decomposition
        experts_added = []
        for expert in response.experts:
            # Override the name with the one from decomposition
            expert.name = component.expert_name
            
            code = expert.to_code() if hasattr(expert, 'to_code') else ""
            
            code = self._fix_expert_function_name(code, component.expert_name)
            
            # Determine expert type based on component
            if component.expert_type == 'visual':
                expert_type = 'drawing'  # Use 'drawing' for consistency with kernel generator
            elif component.expert_type in ['temporal_update', 'state_update', 'utility']:
                expert_type = 'utility'
            elif expert.is_interaction:
                expert_type = 'interaction'
            else:
                expert_type = 'single'
            
            expert_info = ExpertInfo(
                name=component.expert_name,
                description=component.description,
                weight=weight,
                expert_type=expert_type,
                code=code,
                applies_to_species=component.applies_to_species,
                draw_order='post' if expert_type == 'drawing' else None  # Default draw_order for visual experts
            )
            self.experts.append(expert_info)
            self.expert_weights[component.expert_name] = weight
            experts_added.append(component.expert_name)
            logger.info(f"Registered component expert: {component.expert_name} (weight={weight})")
        
        return {
            'success': True,
            'experts_added': len(experts_added),
            'expert_names': experts_added,
            'states_created': len(response.states_needed) if response.states_needed else 0
        }

    async def add_behavior(
        self, 
        description: str,
        weight: float = 1.0,
        skip_decomposition: bool = False
    ) -> Dict[str, Any]:
        """
        Add a new behavior from natural language description.
        Always uses decomposition followed by holistic synthesis for consistency.
        
        Args:
            description: Natural language behavior description
            weight: Weight for this behavior in integration
            skip_decomposition: Skip decomposition check (for already decomposed components)
            
        Returns:
            Dictionary with synthesis results
        """
        collector = get_collector()
        
        with collector.trace_node("add_behavior", "synthesis", description=description) as node:
            logger.info(f"Adding behavior: {description}")
            
            # Always decompose first (unless explicitly skipped)
            decomposed = None
            if not skip_decomposition and self.decomposer:
                try:
                    decomposed = await self.decomposer.decompose(description)
                    logger.info(f"Decomposition complete: components={len(decomposed.components)}")
                    
                    # Extract particle count if detected
                    if decomposed and hasattr(decomposed, 'particle_count') and decomposed.particle_count:
                        self.detected_particle_count = decomposed.particle_count
                        logger.info(f"Detected particle count: {self.detected_particle_count}")
                    
                    # Extract species configuration from decomposition
                    if decomposed and hasattr(decomposed, 'species_info') and decomposed.species_info:
                        from .models import SpeciesConfiguration
                        from .color_resolver import ColorResolver
                        species_info = decomposed.species_info
                        
                        # Resolve color descriptions to RGBA values
                        resolved_colors = {}
                        if species_info.species_color_descriptions and species_info.species_color_descriptions is not None:
                            # Pass the synthesizer's model to ColorResolver for LLM color resolution
                            color_resolver = ColorResolver(llm_client=self.synthesizer.model)
                            # Now it's a list of SpeciesColorMapping objects or dicts
                            for color_mapping in species_info.species_color_descriptions:
                                # Handle both object and dict formats
                                if hasattr(color_mapping, 'species_id'):
                                    species_id = color_mapping.species_id
                                    color_desc = color_mapping.color_description if hasattr(color_mapping, 'color_description') else 'gray'
                                elif isinstance(color_mapping, dict):
                                    species_id = color_mapping.get('species_id', 0)
                                    color_desc = color_mapping.get('color_description', 'gray')
                                else:
                                    continue  # Skip invalid items
                                    
                                try:
                                    # Use async color resolution if available
                                    resolved_colors[species_id] = await color_resolver.resolve_color_name(color_desc)
                                    logger.info(f"Resolved color '{color_desc}' for species {species_id}")
                                except Exception as e:
                                    logger.warning(f"Failed to resolve color '{color_desc}': {e}")
                                    # Use fallback colors
                                    resolved_colors[species_id] = color_resolver.get_default_species_colors(1)[0]
                        elif hasattr(species_info, 'species_colors') and species_info.species_colors:
                            # Use directly provided RGBA colors if available
                            for color_mapping in species_info.species_colors:
                                resolved_colors[color_mapping.get('species_id', 0)] = color_mapping.get('rgba_values', [0.7, 0.7, 0.7, 1.0])
                        else:
                            # Generate default colors
                            color_resolver = ColorResolver()
                            resolved_colors = color_resolver.get_default_species_colors(species_info.total_count)
                        
                        # Convert species names from list to dict
                        species_names_dict = {}
                        if species_info.species_names and species_info.species_names is not None:
                            for item in species_info.species_names:
                                # Handle both object and dict formats
                                if hasattr(item, 'species_id'):
                                    species_id = item.species_id
                                    name = item.name if hasattr(item, 'name') else 'unnamed'
                                elif isinstance(item, dict):
                                    species_id = item.get('species_id', 0)
                                    name = item.get('name', 'unnamed')
                                else:
                                    continue  # Skip invalid items
                                species_names_dict[species_id] = name

                        # Convert to SpeciesConfiguration format
                        self.current_species_config = SpeciesConfiguration(
                            species_ids=list(range(species_info.total_count)),
                            species_names=species_names_dict,
                            interaction_pairs=getattr(species_info, 'interaction_pairs', None) or [],
                            species_behaviors=None,
                            requires_all_species=False,
                            colors=resolved_colors
                        )
                        logger.info(f"Extracted species config from decomposition: {species_info.total_count} species")
                except Exception as e:
                    logger.warning(f"Decomposition failed: {e}")
                    decomposed = None
            
            # Always use holistic synthesis with decomposition result
            logger.info("Using holistic synthesis pipeline")
            result = await self.synthesize_complete_behavior(description, weight, decomposition=decomposed)
            
            # Handle case where result could be None
            if result is None:
                logger.error(f"synthesize_complete_behavior returned None for: {description}")
                result = {
                    'success': False,
                    'experts_added': 0,
                    'expert_names': [],
                    'states_created': 0,
                    'pattern_type': 'unknown',
                    'pattern_confidence': 0.0
                }
            
            # Format result consistently
            expert_names = result.get('expert_names', [])
            if not expert_names and 'components' in result:
                for comp in result['components']:
                    if isinstance(comp, dict):
                        if 'result' in comp and 'expert_names' in comp['result']:
                            expert_names.extend(comp['result']['expert_names'])
                        elif 'expert_name' in comp:
                            expert_names.append(comp['expert_name'])
            
            # Build final result
            final_result = {
                'success': result.get('success', True),
                'experts_added': result.get('experts_added', 0),
                'expert_names': expert_names,
                'states_created': result.get('states_created', 0),
                'species_count': len(self.current_species_config.species_ids) if self.current_species_config else 1,
                'pattern_type': result.get('pattern_type', 'unknown'),
                'pattern_confidence': result.get('pattern_confidence', 0.0)
            }
            
            # Regenerate integration kernel if we have experts
            if self.experts:
                await self._regenerate_kernel()
            
            # Update trace node with results
            if node:
                node.output_data = final_result
            
            return final_result
    
    async def _regenerate_kernel(self):
        # Separate experts by type
        single_experts = [e for e in self.experts if e.expert_type == 'single']
        interaction_experts = [e for e in self.experts if e.expert_type == 'interaction']
        visual_experts = [e for e in self.experts if e.expert_type == 'visual']
        utility_experts = [e for e in self.experts if e.expert_type == 'utility']
        
        # Build species conditions mapping from expert info
        species_conditions = {}
        for expert in self.experts:
            if expert.applies_to_species is not None:
                species_conditions[expert.name] = expert.applies_to_species
        
        # Generate kernel code with species configuration and conditions
        kernel_code = self.kernel_generator.generate(
            single_expert_names=[e.name for e in single_experts],
            interaction_expert_names=[e.name for e in interaction_experts],
            expert_weights=self.expert_weights,
            tolvera_instance=self.tv,
            species_config=self.current_species_config,  # Pass species config for proper mapping
            species_conditions=species_conditions,  # Pass explicit species conditions
            visual_expert_names=[e.name for e in visual_experts]  # Pass visual experts separately
        )
        
        # Compile and register the kernel
        # This would integrate with Tölvera's kernel system
        logger.info("Regenerated integration kernel with all experts")
    
    async def _regenerate_drawing_kernel(self):
        from ..generation.drawing import DrawingKernelGenerator
        
        # Separate drawing experts by type and order
        # Include both 'drawing' and 'visual' expert types
        pre_draw = [e for e in self.experts if e.expert_type in ['drawing', 'visual'] and getattr(e, 'draw_order', None) == 'pre']
        post_draw = [e for e in self.experts if e.expert_type in ['drawing', 'visual'] and getattr(e, 'draw_order', 'post') == 'post']
        interaction_draw = [e for e in self.experts if e.expert_type == 'drawing_interaction']
        
        # Generate drawing kernel code
        drawing_generator = DrawingKernelGenerator()
        kernel_code = drawing_generator.generate(
            pre_draw_experts=[e.name for e in pre_draw],
            post_draw_experts=[e.name for e in post_draw],
            interaction_draw_experts=[e.name for e in interaction_draw],
            expert_weights=self.expert_weights,
            tolvera_instance=self.tv
        )
        
        # Store the generated kernel code for later use in sketch generation
        self.drawing_kernel_code = kernel_code
        
        # Compile and register the drawing kernel
        logger.info("Regenerated drawing kernel with all drawing experts")
    
    async def _regenerate_utility_kernel(self):
        """Regenerate the utility kernel with all utility experts."""
        # Get all utility experts
        utility_experts = [e for e in self.experts if e.expert_type == 'utility']
        
        if not utility_experts:
            return
        
        # Generate utility kernel code
        kernel_code = self.kernel_generator.generate_utility_kernel(
            utility_expert_names=[e.name for e in utility_experts],
            function_name="update_utilities"
        )
        
        # Store the generated kernel code (will be included in sketch generation)
        self.utility_kernel_code = kernel_code
        
        logger.info(f"Regenerated utility kernel with {len(utility_experts)} utility experts")
    
    async def _create_states_for_component(
        self, 
        description: str, 
        states_needed: List[str],
        component_type: str
    ):
        from .models import StateDefinition
        
        # Map of common state names to their definitions
        state_templates = {
            # Grid-based states (cellular automata)
            "grid_x": StateDefinition(
                name="grid_x", category="particle", type="ti.i32",
                min=0, max=100, description="Grid X position"
            ),
            "grid_y": StateDefinition(
                name="grid_y", category="particle", type="ti.i32",
                min=0, max=100, description="Grid Y position"
            ),
            "is_alive": StateDefinition(
                name="is_alive", category="particle", type="ti.i32",
                min=0, max=1, description="Cell alive state"
            ),
            "neighbor_count": StateDefinition(
                name="neighbor_count", category="particle", type="ti.i32",
                min=0, max=8, description="Number of alive neighbors"
            ),
            "next_state": StateDefinition(
                name="next_state", category="particle", type="ti.i32",
                min=0, max=1, description="Next generation state"
            ),
            
            # Pheromone/trail states
            "heading": StateDefinition(
                name="heading", category="particle", type="ti.f32",
                min=0.0, max=6.28319, description="Movement heading angle"
            ),
            "sensor_angle": StateDefinition(
                name="sensor_angle", category="particle", type="ti.f32",
                min=0.0, max=1.57, description="Angle between sensors"
            ),
            "sensor_distance": StateDefinition(
                name="sensor_distance", category="particle", type="ti.f32",
                min=5.0, max=50.0, description="Sensor detection distance"
            ),
            "turn_speed": StateDefinition(
                name="turn_speed", category="particle", type="ti.f32",
                min=0.0, max=3.0, description="Turning rate"
            ),
            
            # Morphogenetic states
            "age": StateDefinition(
                name="age", category="particle", type="ti.f32",
                min=0.0, max=1000.0, description="Particle age"
            ),
            "cell_type": StateDefinition(
                name="cell_type", category="particle", type="ti.i32",
                min=0, max=10, description="Cell differentiation type"
            ),
            "growth_direction": StateDefinition(
                name="growth_direction", category="particle", type="ti.f32",
                min=0.0, max=6.28319, description="Growth angle"
            ),
            
            # Oscillator states
            "phase": StateDefinition(
                name="phase", category="particle", type="ti.f32",
                min=0.0, max=6.28319, description="Oscillator phase"
            ),
            "frequency": StateDefinition(
                name="frequency", category="particle", type="ti.f32",
                min=0.1, max=10.0, description="Oscillation frequency"
            ),
            
            # Resource/energy states
            "energy": StateDefinition(
                name="energy", category="particle", type="ti.f32",
                min=0.0, max=100.0, description="Energy level", initial=80.0
            ),
            "depletion_rate": StateDefinition(
                name="depletion_rate", category="particle", type="ti.f32",
                min=0.0, max=1.0, description="Energy depletion rate per frame", initial=0.01
            ),
            "threshold": StateDefinition(
                name="threshold", category="particle", type="ti.f32",
                min=0.0, max=100.0, description="Energy threshold for returning home", initial=20.0
            ),
            "origin": StateDefinition(
                name="origin", category="particle", type="ti.math.vec2",
                min=0.0, max=1.0, description="Home position to return to"
            ),
            "home_pos": StateDefinition(
                name="home_pos", category="particle", type="ti.math.vec2",
                min=0.0, max=1.0, description="Home position"
            ),
            "is_tired": StateDefinition(
                name="is_tired", category="particle", type="ti.i32",
                min=0, max=1, description="Whether particle is tired"
            ),
            "resource_consumption_rate": StateDefinition(
                name="resource_consumption_rate", category="species", type="ti.f32",
                min=0.0, max=1.0, description="Resource usage rate"
            ),
            "return_force": StateDefinition(
                name="return_force", category="species", type="ti.f32",
                min=0.0, max=1000.0, description="Force strength for returning home", initial=100.0
            ),
            
            # Chemical field states
            "u": StateDefinition(
                name="u", category="particle", type="ti.f32",
                min=0.0, max=1.0, description="Chemical activator concentration"
            ),
            "v": StateDefinition(
                name="v", category="particle", type="ti.f32",
                min=0.0, max=1.0, description="Chemical inhibitor concentration"
            ),
            
            # Genetic states
            "gene1": StateDefinition(
                name="gene1", category="particle", type="ti.f32",
                min=0.0, max=1.0, description="Genetic trait 1"
            ),
            "gene2": StateDefinition(
                name="gene2", category="particle", type="ti.f32",
                min=0.0, max=1.0, description="Genetic trait 2"
            ),
            "fitness": StateDefinition(
                name="fitness", category="particle", type="ti.f32",
                min=0.0, max=100.0, description="Evolutionary fitness"
            ),
        }
        
        # Collect states to create
        states_to_create = {'global': {}, 'particle': {}, 'species': {}}
        
        for state_name in states_needed:
            if state_name in state_templates:
                state_def = state_templates[state_name]
                states_to_create[state_def.category][state_name] = state_def
            else:
                # Create a generic state if not in templates
                logger.warning(f"Unknown state '{state_name}' requested, creating generic float state")
                state_def = StateDefinition(
                    name=state_name, category="particle", type="ti.f32",
                    min=0.0, max=1.0, description=f"Custom state for {component_type}"
                )
                states_to_create["particle"][state_name] = state_def
        
        # Create the states
        if any(states_to_create.values()):
            self.state_manager.create_states_from_spec(states_to_create)
            logger.info(f"Created states for {component_type}: {states_needed}")
    
    # Legacy _generate_temporal_update_kernel method removed - utility experts now handle temporal updates
    
    def _has_grid_states(self, available_states: Dict[str, List[str]]) -> bool:
        return any(s in available_states.get('particle', []) 
                  for s in ['grid_x', 'grid_y', 'is_alive', 'neighbor_count'])
    
    def _has_oscillator_states(self, available_states: Dict[str, List[str]]) -> bool:
        return any(s in available_states.get('particle', []) 
                  for s in ['phase', 'frequency'])
    
    def _has_growth_states(self, available_states: Dict[str, List[str]]) -> bool:
        return any(s in available_states.get('particle', []) 
                  for s in ['age', 'cell_type'])
    
    # Legacy temporal kernel headers removed - utility experts now handle temporal updates
    
    def _generate_grid_update_code(self, available_states: Dict[str, List[str]]) -> List[str]:
        particle_states = available_states.get('particle', [])
        has_all_grid_states = all(s in particle_states for s in ['grid_x', 'grid_y', 'is_alive', 'neighbor_count'])
        
        if not has_all_grid_states:
            return []
        
        return [
            "    # Step 1: Count neighbors for grid-based patterns",
            "    grid_size = ti.cast(ti.sqrt(ti.cast(tv.pn, ti.f32)), ti.i32)",
            "    for i in range(tv.pn):",
            "        if tv.p.field[i].active == 0:",
            "            continue",
            "        gx = tv.s.llm_particle.field[i].grid_x",
            "        gy = tv.s.llm_particle.field[i].grid_y",
            "        count = 0",
            "        for dx in range(-1, 2):",
            "            for dy in range(-1, 2):",
            "                if dx == 0 and dy == 0:",
            "                    continue",
            "                nx = (gx + dx + grid_size) % grid_size",
            "                ny = (gy + dy + grid_size) % grid_size",
            "                for j in range(tv.pn):",
            "                    if (tv.s.llm_particle.field[j].grid_x == nx and",
            "                        tv.s.llm_particle.field[j].grid_y == ny and",
            "                        tv.s.llm_particle.field[j].is_alive == 1):",
            "                        count += 1",
            "                        break",
            "        tv.s.llm_particle.field[i].neighbor_count = count",
            "    ",
            "    # Step 2: Apply rules and update states",
            "    for i in range(tv.pn):",
            "        if tv.p.field[i].active == 0:",
            "            continue",
            "        alive = tv.s.llm_particle.field[i].is_alive",
            "        neighbors = tv.s.llm_particle.field[i].neighbor_count",
            "        next_state = 0",
            "        if alive == 1:",
            "            if neighbors == 2 or neighbors == 3:",
            "                next_state = 1",
            "        else:",
            "            if neighbors == 3:",
            "                next_state = 1",
            "        tv.s.llm_particle.field[i].next_state = next_state",
            "    ",
            "    # Step 3: Commit state changes",
            "    for i in range(tv.pn):",
            "        tv.s.llm_particle.field[i].is_alive = tv.s.llm_particle.field[i].next_state",
            "        if tv.s.llm_particle.field[i].is_alive == 1:",
            "            tv.p.field[i].size = 5.0",
            "            tv.p.field[i].active = 1.0",
            "        else:",
            "            tv.p.field[i].size = 1.0",
            "            tv.p.field[i].active = 0.3",
            ""
        ]
    
    def _generate_oscillator_update_code(self, available_states: Dict[str, List[str]]) -> List[str]:
        particle_states = available_states.get('particle', [])
        if not ('phase' in particle_states and 'frequency' in particle_states):
            return []
        
        return [
            "    # Update oscillator phases",
            "    for i in range(tv.pn):",
            "        if tv.p.field[i].active > 0:",
            "            frequency = tv.s.llm_particle.field[i].frequency",
            "            tv.s.llm_particle.field[i].phase += frequency * dt * 2 * 3.14159",
            "            if tv.s.llm_particle.field[i].phase > 2 * 3.14159:",
            "                tv.s.llm_particle.field[i].phase -= 2 * 3.14159",
            "            brightness = (ti.sin(tv.s.llm_particle.field[i].phase) + 1.0) * 0.5",
            "            tv.p.field[i].size = 2.0 + brightness * 3.0",
            ""
        ]
    
    def _generate_growth_update_code(self, available_states: Dict[str, List[str]]) -> List[str]:
        particle_states = available_states.get('particle', [])
        if 'age' not in particle_states:
            return []
        
        return [
            "    # Update age and growth",
            "    for i in range(tv.pn):",
            "        if tv.p.field[i].active > 0:",
            "            tv.s.llm_particle.field[i].age += dt",
            ""
        ]
    
    def _generate_energy_update_code(self) -> List[str]:
        return [
            "    # Energy dynamics",
            "    for i in range(tv.pn):",
            "        if tv.p.field[i].active > 0:",
            "            tv.s.llm_particle.field[i].energy *= 0.995",
            "            if tv.s.llm_particle.field[i].energy < 0.1:",
            "                tv.p.field[i].active = 0.0",
            ""
        ]
    
    def _generate_synthesis_temporal_updates(self, available_states: Dict[str, List[str]]) -> List[str]:
        parts = ["    # Synthesis-specified temporal updates"]
        for temporal_update in self.temporal_updates:
            for state_name, update_expr in temporal_update.frame_updates.items():
                if state_name in available_states.get('global', []):
                    expr = update_expr.replace('frame', 'ti.cast(frame, ti.f32)')
                    parts.append(f"    tv.s.llm_global.field[0].{state_name} = {expr}")
                elif state_name in available_states.get('particle', []):
                    parts.extend([
                        "    for i in range(tv.pn):",
                        "        if tv.p.field[i].active > 0:",
                        f"            tv.s.llm_particle.field[i].{state_name} = {update_expr.replace(state_name, f'tv.s.llm_particle.field[i].{state_name}')}"
                    ])
        return parts
    
    def _generate_pixel_kernels(self, pixel_field_req) -> str:
        """Generate pixel field kernels for pheromone/trail operations."""
        kernels = []
        
        if pixel_field_req.needs_decay:
            kernels.append("""
@ti.kernel
def decay_pheromones():
    \"\"\"Decay pheromone trails over time.\"\"\"
    evaporation_rate = 0.99  # Could be from global state
    for i, j in ti.ndrange(tv.x, tv.y):
        tv.px.px.rgba[i, j][0] *= evaporation_rate
        tv.px.px.rgba[i, j][1] *= evaporation_rate
        tv.px.px.rgba[i, j][2] *= evaporation_rate
""")
        
        if pixel_field_req.needs_diffusion:
            kernels.append("""
@ti.kernel
def diffuse_pheromones():
    \"\"\"Diffuse pheromone trails to neighboring pixels.\"\"\"
    diffusion_rate = 0.1
    temp = ti.field(dtype=ti.f32, shape=(tv.x, tv.y, 3))
    
    # Copy to temp and apply diffusion
    for i, j in ti.ndrange(tv.x, tv.y):
        center_val = tv.px.px.rgba[i, j]
        sum_val = ti.math.vec3(0.0, 0.0, 0.0)
        count = 0
        
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                ni = (i + di) % tv.x
                nj = (j + dj) % tv.y
                sum_val += tv.px.px.rgba[ni, nj][:3]
                count += 1
        
        avg_val = sum_val / count
        temp[i, j, 0] = center_val[0] * (1 - diffusion_rate) + avg_val[0] * diffusion_rate
        temp[i, j, 1] = center_val[1] * (1 - diffusion_rate) + avg_val[1] * diffusion_rate
        temp[i, j, 2] = center_val[2] * (1 - diffusion_rate) + avg_val[2] * diffusion_rate
    
    # Copy back
    for i, j in ti.ndrange(tv.x, tv.y):
        tv.px.px.rgba[i, j][0] = temp[i, j, 0]
        tv.px.px.rgba[i, j][1] = temp[i, j, 1]
        tv.px.px.rgba[i, j][2] = temp[i, j, 2]
""")
        
        if pixel_field_req.needs_deposition:
            kernels.append("""
@ti.kernel
def deposit_trails():
    \"\"\"Deposit pheromone trails from particles.\"\"\"
    deposit_amount = 0.1
    
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            pos = tv.p.field[i].pos
            x = ti.cast(pos[0], ti.i32) % tv.x
            y = ti.cast(pos[1], ti.i32) % tv.y
            
            # Deposit based on species
            species = tv.p.field[i].species
            if species == 0:
                tv.px.px.rgba[x, y][0] += deposit_amount
                tv.px.px.rgba[x, y][0] = ti.min(1.0, tv.px.px.rgba[x, y][0])
            elif species == 1:
                tv.px.px.rgba[x, y][1] += deposit_amount
                tv.px.px.rgba[x, y][1] = ti.min(1.0, tv.px.px.rgba[x, y][1])
            elif species == 2:
                tv.px.px.rgba[x, y][2] += deposit_amount
                tv.px.px.rgba[x, y][2] = ti.min(1.0, tv.px.px.rgba[x, y][2])
""")
        
        return "\n".join(kernels) if kernels else ""
    
    async def add_drawing_behavior(
        self,
        description: str,
        weight: float = 1.0,
        draw_order: str = "post"
    ) -> Dict[str, Any]:
        logger.info(f"Adding drawing behavior: {description}")
        
        # Import drawing classifier
        from ..generation.drawing import DrawingClassifier
        
        # Classify the drawing behavior
        classifier = DrawingClassifier()
        classification = classifier.classify(description)
        
        if not classification["is_drawing"]:
            # Fall back to regular behavior synthesis
            return await self.add_behavior(description, weight)
        
        # Get current available states
        available_states = self.state_manager.get_available_states()
        
        # Synthesize the drawing behavior
        response = await self.synthesizer.synthesize_behavior(
            description,
            available_states
        )
        
        # Register drawing experts
        experts_added = []
        for expert in response.experts:
            expert_info = ExpertInfo(
                name=expert.name,
                description=expert.description,
                weight=weight * expert.weight,
                expert_type='drawing_interaction' if expert.is_interaction else 'drawing',
                code=expert.to_code(),
                draw_order=draw_order
            )
            self.experts.append(expert_info)
            self.expert_weights[expert.name] = expert_info.weight
            experts_added.append(expert.name)
            logger.info(f"Registered drawing expert: {expert.name} (order={draw_order})")
        
        # Regenerate drawing kernel if we have drawing experts
        if any(e.expert_type.startswith('drawing') for e in self.experts):
            await self._regenerate_drawing_kernel()
        
        return {
            'success': True,
            'experts_added': len(experts_added),
            'expert_names': experts_added,
            'draw_order': draw_order,
            'confidence': classification["confidence"]
        }
    
    async def add_complex_behavior(
        self,
        description: str,
        weight: float = 1.0,
        decompose: bool = True,
        auto_embellish: bool = True,
        _decomposed=None
    ) -> Dict[str, Any]:
        collector = get_collector()
        
        with collector.trace_node("add_complex_behavior", "synthesis", 
                                      description=description,
                                      decompose=decompose,
                                      auto_embellish=auto_embellish) as node:
            logger.info(f"Adding complex behavior: {description}")
            
            if not self.decomposer:
                # Fall back to regular synthesis if decomposer not available
                return await self.add_behavior(description, weight, skip_decomposition=True)
            
            # Use decomposer to break down the behavior (if not already done)
            if _decomposed:
                decomposed = _decomposed
            elif decompose:
                decomposed = await self.decomposer.decompose(description)
                
                # Extract particle count if detected
                if decomposed and hasattr(decomposed, 'particle_count') and decomposed.particle_count:
                    self.detected_particle_count = decomposed.particle_count
                    logger.info(f"Detected particle count: {self.detected_particle_count}")
            else:
                # If decompose=False and no _decomposed, fall back to simple synthesis
                return await self.add_behavior(description, weight, skip_decomposition=True)
            
            # Create synthesis context from decomposition
            synthesis_context = {
                "shared_context": decomposed.context,
                "previous_experts": [],
                "species_config": self.current_species_config,
                "tolvera_context": {
                    "pn": self.tv.pn,
                    "sn": self.tv.sn,
                    "width": self.tv.x,
                    "height": self.tv.y
                }
            }
            
            # Synthesize each component
            results = {
                'interpretation': decomposed.interpretation,
                'components': [],
                'embellishments': [],
                'total_experts': 0,
                'implementation_notes': decomposed.implementation_notes
            }
            
            # Process each decomposed component
            for component in decomposed.components:
                try:
                    # Adjust weight based on priority
                    component_weight = weight * component.priority
                    
                    # Check if component needs states
                    if component.required_states:
                        state_names = [state[0] for state in component.required_states]
                        await self._create_states_for_component(
                            component.description, 
                            state_names,
                            component.expert_type
                        )
                    
                    # Create a synthesis description that includes high-level behavioral guidance
                    synthesis_description = f"{component.description}. Behavior: {component.implementation}"
                    
                    # Pass context requirements to synthesizer
                    synthesis_context.update({
                        "current_component": component.expert_name,
                        "constraints": component.context_requirements or [],
                        "previous_experts": synthesis_context["previous_experts"]
                    })
                    
                    # Route to appropriate synthesis method based on type
                    result = await self._synthesize_component(
                        component,
                        component_weight,
                        synthesis_context
                    )
                    
                    # Track this expert for context threading
                    synthesis_context["previous_experts"].append({
                        "name": component.expert_name,
                        "type": component.expert_type,
                        "context": component.context_requirements
                    })
                    
                    # Track results
                    results['components'].append({
                        'expert_name': component.expert_name,
                        'description': component.description,
                        'type': component.expert_type,
                        'result': result
                    })
                    
                    results['total_experts'] += result.get('experts_added', 0)
                    
                except Exception as e:
                    logger.warning(f"Failed to synthesize component '{component.expert_name}': {e}")
                    # Continue with other components
            
            # Update trace node with results
            if node:
                node.output_data = results
            
            return results
    
    async def _synthesize_component(
        self, 
        component,
        weight: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize a single component based on its type."""
        
        if component.expert_type in ['force', 'interaction', 'state_update']:
            # Use existing synthesis methods
            synthesis_description = f"{component.description}. Behavior: {component.implementation}"
            result = await self.add_behavior(
                synthesis_description,
                weight=weight
            )
            return result
            
        elif component.expert_type == 'initialization':
            # Synthesize initialization kernel
            init_response = await self.synthesizer.synthesize_initialization(
                component.implementation,
                species_config=self.current_species_config,
                context=context
            )
            
            # Store initialization code
            self.synthesized_initialization = init_response.code
            
            return {
                'type': 'initialization',
                'code': init_response.code,
                'description': init_response.description
            }
            
        elif component.expert_type in ['temporal_update', 'state_update', 'utility']:
            # Synthesize utility expert (temporal updates, state updates, etc.)
            available_states = self.state_manager.get_available_states()
            utility_response = await self.synthesizer.synthesize_utility_expert(
                component.implementation,
                available_states=available_states,
                context=context,
                expert_type=component.expert_type  # Pass the specific expert type!
            )
            
            # States for utility experts are created upfront by decomposer
            states_created = 0
            
            # Create Expert object from utility response
            expert = Expert(
                name=utility_response.name,
                description=utility_response.description,
                code=utility_response.code,
                is_interaction=False,
                weight=weight
            )
            
            # Register as utility expert
            expert_info = ExpertInfo(
                name=expert.name,
                description=expert.description,
                weight=weight,
                expert_type='utility',
                code=expert.code
            )
            self.experts.append(expert_info)
            self.expert_weights[expert.name] = weight
            logger.info(f"Registered utility expert: {expert.name}")
            
            # Regenerate utility kernel
            await self._regenerate_utility_kernel()
            
            return {
                'type': 'utility',
                'expert_name': expert.name,
                'code': utility_response.code,
                'description': utility_response.description,
                'experts_added': 1,
                'states_created': states_created
            }
            
        elif component.expert_type == 'configuration':
            # Synthesize configuration
            config_response = await self.synthesizer.synthesize_configuration(
                component.implementation,
                species_config=self.current_species_config
            )
            
            # Store configuration code
            self.synthesized_configuration = config_response.config_code
            
            return {
                'type': 'configuration',
                'code': config_response.config_code,
                'species_count': config_response.species_count,
                'particle_count': config_response.particle_count
            }
            
        elif component.expert_type == 'visual':
            # Use existing drawing behavior synthesis
            synthesis_description = f"{component.description}. Behavior: {component.implementation}"
            result = await self.add_drawing_behavior(
                synthesis_description,
                weight=weight
            )
            return result
            
        else:
            logger.warning(f"Unknown component type: {component.expert_type}")
            return {}
    
    async def _synthesize_pure_drawing(self, description: str, weight: float, decomposed) -> Dict[str, Any]:
        """
        Synthesize a pure drawing behavior without particle systems.
        
        Args:
            description: Natural language description
            weight: Weight (not used for pure drawing)
            decomposed: Decomposition result
            
        Returns:
            Dictionary with synthesis results
        """
        logger.info(f"Synthesizing pure drawing behavior: {description}")
        
        # Synthesize drawing code using the synthesizer
        # We'll use the drawing behavior synthesis which should generate appropriate drawing code
        try:
            # Create a simple drawing kernel
            drawing_code = f"""
# Draw based on description: {description}
# This is a placeholder - actual synthesis would generate proper drawing code
width = tv.x // 2
height = tv.y // 2
x = tv.x // 2 - width // 2
y = tv.y // 2 - height // 2
tv.px.rect(x, y, width, height, ti.Vector([1.0, 0.0, 0.0, 1.0]))
"""
            
            # Store the drawing code
            self.pure_drawing_code = drawing_code
            self.is_pure_drawing = True
            
            return {
                'success': True,
                'pattern_type': 'pure_drawing',
                'experts_added': 0,  # No experts for pure drawing
                'drawing_functions': 1,
                'description': description,
                'behavior_category': 'pure_drawing'
            }
        except Exception as e:
            logger.error(f"Failed to synthesize pure drawing behavior: {e}")
            return {
                'success': False,
                'pattern_type': 'pure_drawing',
                'error': str(e)
            }
    
    async def generate_sketch_async(
        self, 
        description: str,
        filename: Optional[str] = None,
        use_timestamp: bool = True,
        validate: bool = True,
        auto_fix: bool = True
    ) -> tuple[str, str]:
        # Check if this is a pure drawing behavior
        if hasattr(self, 'is_pure_drawing') and self.is_pure_drawing:
            logger.info("Generating pure drawing sketch")
            sketch = self.sketch_generator.generate_pure_drawing(
                description=description,
                drawing_code=self.pure_drawing_code if hasattr(self, 'pure_drawing_code') else "# Drawing code placeholder"
            )
            
            # Save if filename provided or use_timestamp is True
            if filename or use_timestamp:
                from pathlib import Path
                from datetime import datetime
                
                # Create generated_sketches directory if it doesn't exist
                sketch_dir = Path('examples/generated_sketches')
                sketch_dir.mkdir(parents=True, exist_ok=True)
                
                if filename and not use_timestamp:
                    # Use provided filename as-is
                    file_path = Path(filename)
                    if not file_path.parent.name or file_path.parent == Path('.'):
                        file_path = sketch_dir / file_path.name
                else:
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = filename.replace('.py', '') if filename else 'pure_drawing_sketch'
                    file_path = sketch_dir / f"{base_name}_{timestamp}.py"
                
                with open(file_path, 'w') as f:
                    f.write(sketch)
                logger.info(f"Saved pure drawing sketch to: {file_path}")
                
                # Validate if requested
                if validate:
                    from .sketch_validator import SketchValidator
                    validator = SketchValidator()
                    success, fixed_sketch, error = await validator.validate_sketch(
                        sketch, auto_fix=auto_fix, verbose=False
                    )
                    if success and fixed_sketch != sketch:
                        # Update the saved file with the fixed version
                        with open(file_path, 'w') as f:
                            f.write(fixed_sketch)
                        logger.info(f"Sketch validated and fixed: {file_path}")
                        return fixed_sketch, str(file_path)
                    elif not success:
                        logger.warning(f"Sketch validation failed: {error}")
                
                return sketch, str(file_path)
            return sketch, ""
        
        # Regular particle system sketch generation
        # Collect synthesized helper functions
        helper_code = ""
        if self.synthesized_helpers:
            helper_parts = ["# Helper Functions"]
            for name, code in self.synthesized_helpers.items():
                helper_parts.append(f"\n{code}\n")
            helper_code = "\n".join(helper_parts)
        
        # Separate expert code by type - only particle force experts go in expert_code
        force_experts = [e for e in self.experts if e.expert_type in ['single', 'interaction']]
        expert_code = "\n\n".join(e.code for e in force_experts)
        
        # Combine helper functions and force experts only
        if helper_code:
            combined_expert_code = f"{helper_code}\n\n{expert_code}"
        else:
            combined_expert_code = expert_code
        
        # Use synthesized initialization if available, otherwise fall back to template
        if hasattr(self, 'synthesized_initialization') and self.synthesized_initialization:
            init_code = self.synthesized_initialization
        else:
            init_code = self._generate_init_code()
        
        # Generate state code
        state_code = self._generate_state_code()
        
        # Generate kernel (simplified for now)
        kernel_code = self._generate_simple_kernel()
        
        # Temporal updates are now handled by utility experts - no legacy kernel code needed
        temporal_kernel_code = ""
        
        # Add pixel kernels if generated
        if hasattr(self, 'pixel_kernels') and self.pixel_kernels:
            temporal_kernel_code = f"{temporal_kernel_code}\n\n{self.pixel_kernels}"
        
        # Use synthesized configuration if available, otherwise generate based on species
        if hasattr(self, 'synthesized_configuration') and self.synthesized_configuration:
            config_code = self.synthesized_configuration
        elif self.current_species_config or self.detected_particle_count:
            species_count = len(self.current_species_config.species_ids) if self.current_species_config else 1
            particle_count = self.detected_particle_count if self.detected_particle_count else 1000
            
            # Override kwargs to use detected species count and particle count
            # Using correct Tölvera parameter names: 'species' and 'particles'
            config_code = f"""# Override default Tölvera parameters
    # Detected {species_count} species from description
    import sys
    if 'species' not in kwargs:
        kwargs['species'] = {species_count}  # Use detected species count
    if 'particles' not in kwargs:
        kwargs['particles'] = {particle_count}  # Use detected particle count
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080"""
        else:
            config_code = ""
        
        # Generate drawing kernel and code if we have visual experts
        # Include both 'visual' and 'drawing' types as they're the same concept
        visual_experts = [e for e in self.experts if e.expert_type in ['visual', 'drawing', 'drawing_interaction']]
        drawing_kernel_code = ""
        drawing_expert_code = ""
        if visual_experts:
            # Use the stored drawing kernel if available, otherwise generate a new one
            if hasattr(self, 'drawing_kernel_code') and self.drawing_kernel_code:
                drawing_kernel_code = self.drawing_kernel_code
            else:
                drawing_kernel_code = self.kernel_generator.generate_drawing_kernel_from_experts(
                    visual_expert_names=[e.name for e in visual_experts],
                    function_name="draw"
                )
            # Include the actual visual expert function code
            drawing_expert_code = "\n\n".join(e.code for e in visual_experts)
        
        # Check if we have non-visual experts (for particle physics)
        single_experts = [e for e in self.experts if e.expert_type == 'single']
        interaction_experts = [e for e in self.experts if e.expert_type == 'interaction']
        utility_experts = [e for e in self.experts if e.expert_type == 'utility']
        has_non_visual_experts = len(single_experts) > 0 or len(interaction_experts) > 0
        
        # Get utility expert code and kernel
        utility_expert_code = "\n\n".join([e.code for e in utility_experts])
        utility_kernel_code = ""
        if utility_experts:
            utility_kernel_code = self.kernel_generator.generate_utility_kernel(
                utility_expert_names=[e.name for e in utility_experts],
                function_name="update_utilities"
            )
        
        # Generate complete sketch
        sketch = self.sketch_generator.generate(
            description=description,
            experts=[combined_expert_code],  # Now includes helpers
            kernel=kernel_code,
            init_code=init_code,
            state_code=state_code,
            temporal_code=temporal_kernel_code,
            config_code=config_code,
            utility_code=utility_expert_code,
            utility_kernel=utility_kernel_code,
            drawing_code=drawing_expert_code,  # Include visual expert function code
            drawing_kernel=drawing_kernel_code,
            has_non_visual_experts=has_non_visual_experts
        )
        
        # STAGE 2: Apply architectural refinement to transform simple sketches into sophisticated simulations
        logger.info("Applying architectural refinement to enhance sketch sophistication")
        
        # Get collector for tracing
        from ..debug.tracing import get_collector
        collector = get_collector()
        
        # Wrap the entire architectural refinement in a trace node
        with collector.trace_node("architectural_refinement", "refinement", 
                                  description=description,
                                  stage="sketch_enhancement") as refinement_node:
            try:
                # Detect if this sketch would benefit from architectural enhancement
                pattern_info = self.sketch_refiner.detect_architectural_pattern(description, sketch)
                
                # Add pattern detection metadata to the trace
                if refinement_node:
                    refinement_node.metadata = {
                        'pattern': pattern_info['primary_pattern'],
                        'confidence': pattern_info['confidence'],
                        'needs_refinement': pattern_info['needs_refinement']
                    }
                
                if pattern_info['needs_refinement'] and pattern_info['primary_pattern']:
                    logger.info(f"Detected {pattern_info['primary_pattern']} pattern (confidence: {pattern_info['confidence']:.2f})")
                    logger.info("Applying architectural refinement to enhance sophistication")
                    
                    refinement_result = await self.sketch_refiner.refine_to_architecture(
                        sketch_code=sketch,
                        description=description,
                        pattern=pattern_info['primary_pattern']
                    )
                    
                    if refinement_result['success']:
                        sketch = refinement_result['refined_code']
                        logger.info(f"Architectural refinement applied: {refinement_result['changes_made']}")
                        if refinement_result.get('warnings'):
                            logger.warning(f"Refinement warnings: {refinement_result['warnings']}")
                        
                        # Update trace with success information
                        if refinement_node:
                            refinement_node.output_data = {
                                'changes_made': refinement_result['changes_made'],
                                'warnings': refinement_result.get('warnings'),
                                'refined': True
                            }
                    else:
                        logger.warning(f"Architectural refinement failed: {refinement_result.get('error', 'Unknown error')}")
                        if refinement_node:
                            refinement_node.set_error(refinement_result.get('error', 'Unknown error'))
                else:
                    logger.info("No architectural enhancement needed - sketch is already appropriate")
                    if refinement_node:
                        refinement_node.output_data = {
                            'refined': False,
                            'reason': 'No enhancement needed'
                        }
                    
            except Exception as e:
                logger.warning(f"Architectural refinement failed: {e}")
                if refinement_node:
                    refinement_node.set_error(str(e))
                # Continue with original sketch if refinement fails
        
        # Save if filename provided or use_timestamp is True
        if filename or use_timestamp:
            from pathlib import Path
            from datetime import datetime
            
            # Create generated_sketches directory if it doesn't exist
            sketch_dir = Path('examples/generated_sketches')
            sketch_dir.mkdir(parents=True, exist_ok=True)
            
            if filename and not use_timestamp:
                # Use provided filename as-is
                file_path = Path(filename)
                if not file_path.parent.name or file_path.parent == Path('.'):
                    file_path = sketch_dir / file_path.name
            else:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = filename.replace('.py', '') if filename else 'generated_sketch'
                file_path = sketch_dir / f"{base_name}_{timestamp}.py"
            
            with open(file_path, 'w') as f:
                f.write(sketch)
            logger.info(f"Saved sketch to: {file_path}")
            
            # Validate if requested
            if validate:
                from .sketch_validator import SketchValidator
                validator = SketchValidator()
                success, fixed_sketch, error = await validator.validate_sketch(
                    sketch, auto_fix=auto_fix, verbose=False
                )
                if success and fixed_sketch != sketch:
                    # Update the saved file with the fixed version
                    with open(file_path, 'w') as f:
                        f.write(fixed_sketch)
                    logger.info(f"Sketch validated and fixed: {file_path}")
                    return fixed_sketch, str(file_path)
                elif not success:
                    logger.warning(f"Sketch validation failed: {error}")
            
            return sketch, str(file_path)
        
        return sketch, ""
    
    def generate_sketch(
        self, 
        description: str,
        filename: Optional[str] = None,
        use_timestamp: bool = True,
        validate: bool = True,
        auto_fix: bool = True
    ) -> tuple[str, str]:
        """
        Synchronous wrapper for generate_sketch_async.
        Generates and optionally validates a complete sketch.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, can't use run_until_complete
                # Fall back to non-validated version
                logger.debug("In async context - using sync generation")
                return self._generate_sketch_sync(description, filename, use_timestamp)
            else:
                return loop.run_until_complete(
                    self.generate_sketch_async(description, filename, use_timestamp, validate, auto_fix)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self.generate_sketch_async(description, filename, use_timestamp, validate, auto_fix)
            )
    
    def _generate_sketch_sync(self, description: str, filename: Optional[str] = None, use_timestamp: bool = True) -> tuple[str, str]:
        """Synchronous sketch generation without validation (fallback)."""
        # This is the original generate_sketch logic without validation
        # Used when we can't run async validation
        
        # Check if this is a pure drawing behavior
        if hasattr(self, 'is_pure_drawing') and self.is_pure_drawing:
            logger.info("Generating pure drawing sketch")
            sketch = self.sketch_generator.generate_pure_drawing(
                description=description,
                drawing_code=self.pure_drawing_code if hasattr(self, 'pure_drawing_code') else "# Drawing code placeholder"
            )
            
            # Save if filename provided or use_timestamp is True
            if filename or use_timestamp:
                from pathlib import Path
                from datetime import datetime
                
                # Create generated_sketches directory if it doesn't exist
                sketch_dir = Path('examples/generated_sketches')
                sketch_dir.mkdir(parents=True, exist_ok=True)
                
                if filename and not use_timestamp:
                    # Use provided filename as-is
                    file_path = Path(filename)
                    if not file_path.parent.name or file_path.parent == Path('.'):
                        file_path = sketch_dir / file_path.name
                else:
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = filename or "pure_drawing_sketch"
                    file_path = sketch_dir / f"{base_name}_{timestamp}.py"
                
                with open(file_path, 'w') as f:
                    f.write(sketch)
                logger.info(f"Saved pure drawing sketch to: {file_path}")
                
                return sketch, str(file_path)
            return sketch, ""
        
        # Regular particle system sketch generation
        # Collect synthesized helper functions
        helper_code = ""
        if hasattr(self, 'synthesized_helpers'):
            helper_parts = ["# Helper Functions"]
            for name, code in self.synthesized_helpers.items():
                helper_parts.append(f"\n{code}\n")
            helper_code = "\n".join(helper_parts)
        
        # Separate expert code by type
        force_experts = [e for e in self.experts if e.expert_type in ['single', 'interaction']] if hasattr(self, 'experts') else []
        # Include both 'visual' and 'drawing' types as they're the same concept
        visual_experts = [e for e in self.experts if e.expert_type in ['visual', 'drawing', 'drawing_interaction']] if hasattr(self, 'experts') else []
        utility_experts = [e for e in self.experts if e.expert_type == 'utility'] if hasattr(self, 'experts') else []
        
        expert_code = "\n\n".join(e.code for e in force_experts)
        
        # Combine helper functions and force experts
        if helper_code:
            combined_expert_code = f"{helper_code}\n\n{expert_code}"
        else:
            combined_expert_code = expert_code if expert_code else "# No experts defined"
        
        # Get drawing code from visual experts
        drawing_expert_code = "\n\n".join(e.code for e in visual_experts) if visual_experts else ""
        
        # Generate drawing kernel if we have visual experts
        drawing_kernel_code = ""
        if visual_experts:
            # Use kernel_generator to create an actual kernel that calls the drawing functions
            drawing_kernel_code = self.kernel_generator.generate_drawing_kernel_from_experts(
                visual_expert_names=[e.name for e in visual_experts],
                function_name="draw"
            )
        
        # Generate utility kernel if we have utility experts
        utility_expert_code = "\n\n".join(e.code for e in utility_experts) if utility_experts else ""
        utility_kernel_code = ""
        if utility_experts:
            utility_kernel_code = self.kernel_generator.generate_utility_kernel(
                utility_expert_names=[e.name for e in utility_experts],
                function_name="update_utilities"
            )
        
        # Get initialization code
        if hasattr(self, 'synthesized_initialization') and self.synthesized_initialization:
            init_code = self.synthesized_initialization
        else:
            init_code = self._generate_init_code()
        
        # Get state code
        state_code = self._generate_state_code()
        
        # Generate kernel
        kernel_code = self._generate_simple_kernel()
        
        # Config code
        config_code = ""
        if self.current_species_config or (hasattr(self, 'detected_particle_count') and self.detected_particle_count):
            # Include species configuration
            kwargs = self.tv.kwargs.copy() if hasattr(self.tv, 'kwargs') else {}
            if self.current_species_config:
                kwargs['species'] = len(self.current_species_config.species_ids)
            if hasattr(self, 'detected_particle_count') and self.detected_particle_count:
                kwargs['pn'] = self.detected_particle_count
            
            config_code = f"""# Override default Tölvera parameters
    # Detected {len(self.current_species_config.species_ids) if self.current_species_config else 1} species from description
    import sys
    if 'species' not in kwargs:
        kwargs['species'] = {kwargs.get('species', 1)}  # Use detected species count
    if 'particles' not in kwargs:
        kwargs['particles'] = {kwargs.get('pn', 1000)}  # Use detected particle count
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080"""
        
        # Check if we have force experts (single/interaction) that actually affect particle movement
        # Only render particles if we have experts that generate forces
        has_non_visual_experts = bool(force_experts)
        
        # Generate complete sketch using the correct signature
        sketch = self.sketch_generator.generate(
            description=description,
            experts=[combined_expert_code],  # List of expert code strings
            kernel=kernel_code,
            init_code=init_code,
            state_code=state_code,
            temporal_code="",  # Legacy, now handled by utility experts
            config_code=config_code,
            utility_code=utility_expert_code,
            utility_kernel=utility_kernel_code,
            drawing_code=drawing_expert_code,
            drawing_kernel=drawing_kernel_code,
            has_non_visual_experts=has_non_visual_experts
        )
        
        # Save if filename provided or use_timestamp is True
        if filename or use_timestamp:
            from pathlib import Path
            from datetime import datetime
            
            # Create generated_sketches directory if it doesn't exist
            sketch_dir = Path('examples/generated_sketches')
            sketch_dir.mkdir(parents=True, exist_ok=True)
            
            if filename and not use_timestamp:
                # Use provided filename as-is
                file_path = Path(filename)
                if not file_path.parent.name or file_path.parent == Path('.'):
                    file_path = sketch_dir / file_path.name
            else:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = filename or "sketch"
                file_path = sketch_dir / f"{base_name}_{timestamp}.py"
            
            with open(file_path, 'w') as f:
                f.write(sketch)
            logger.info(f"Saved sketch to: {file_path}")
            
            return sketch, str(file_path)
        
        return sketch, ""
    
    def _generate_init_code(self) -> str:
        if self.current_species_config:
            # Use species manager to generate initialization
            init_type = "random"
            grid_size = None
            
            # Check if any behavior description suggests grid init
            behavior_descriptions = [e.description for e in self.experts]
            combined_desc = " ".join(behavior_descriptions)
            use_grid, suggested_size = self.species_manager.should_use_grid_init(combined_desc)
            
            if use_grid:
                init_type = "grid"
                grid_size = suggested_size
            # For ecosystem patterns, use clustered initialization
            elif self.current_behavior_requirements and self.current_behavior_requirements.pattern_type == "ecosystem":
                init_type = "clustered"
                logger.info("Using clustered initialization for ecosystem pattern")
            
            # Generate species-aware initialization
            from .species_analyzer import SpeciesInfo
            species_info = SpeciesInfo(
                species_ids=self.current_species_config.species_ids,
                species_names=self.current_species_config.species_names or {},
                interaction_pairs=self.current_species_config.interaction_pairs,
                species_behaviors=self.current_species_config.species_behaviors or {},
                requires_all_species=self.current_species_config.requires_all_species,
                total_count=len(self.current_species_config.species_ids),
                color_hints={}
            )
            
            # Pass the species config and speed_spec for proper initialization
            speed_spec = getattr(self, 'current_speed_spec', None)
            return self.species_manager.get_initialization_code(
                species_info,
                init_type=init_type,
                grid_size=grid_size,
                species_config=self.current_species_config,
                speed_spec=speed_spec
            )
        else:
            # Fallback to simple initialization
            # Important: We need to explicitly set species to 0 for single-species behaviors
            # Otherwise tv.p.randomise() will assign random species values
            
            # Check for speed specification
            speed_spec = getattr(self, 'current_speed_spec', None)
            if speed_spec and speed_spec.uniform:
                # Generate uniform speed initialization
                speed_val = 100.0  # Default
                if speed_spec.value:
                    speed_val = speed_spec.value
                elif speed_spec.magnitude:
                    magnitude_map = {"slow": 50.0, "medium": 100.0, "fast": 200.0, "very_fast": 300.0}
                    speed_val = magnitude_map.get(speed_spec.magnitude, 100.0)
                
                init_code = f"""# Initialize particles with uniform speed
@ti.kernel  
def init_particles():
    for i in range(tv.pn):
        tv.p.field[i].active = 1.0
        tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
        # Uniform speed with random directions
        angle = ti.random() * 2 * 3.14159
        tv.p.field[i].vel = ti.Vector([
            ti.cos(angle) * {speed_val},
            ti.sin(angle) * {speed_val}
        ])
        tv.p.field[i].size = 5.0
        tv.p.field[i].mass = 1.0
        tv.p.field[i].species = 0  # All particles are species 0"""
            else:
                # Default random velocities
                init_code = """# Initialize particles
@ti.kernel  
def init_particles():
    for i in range(tv.pn):
        tv.p.field[i].active = 1.0
        tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
        tv.p.field[i].vel = ti.Vector([
            (ti.random() - 0.5) * 100.0,
            (ti.random() - 0.5) * 100.0
        ])
        tv.p.field[i].size = 5.0
        tv.p.field[i].mass = 1.0
        tv.p.field[i].species = 0  # All particles are species 0

init_particles()

# Initialize species colors
"""
            # Check if we have detected species to initialize
            if hasattr(self, 'current_species_config') and self.current_species_config:
                # Use species analyzer to get colors for all detected species
                from .species_analyzer import SpeciesInfo, SpeciesAnalyzer
                analyzer = SpeciesAnalyzer()
                
                # Create a minimal species info
                species_info = SpeciesInfo(
                    species_ids=list(range(self.tv.sn)),
                    species_names={},
                    interaction_pairs=[],
                    species_behaviors={},
                    requires_all_species=False,
                    total_count=self.tv.sn,
                    color_hints={}
                )
                
                # Get default colors for all species
                for i in range(self.tv.sn):
                    color = analyzer.get_color_for_species(i, species_info)
                    init_code += f"tv.s.species.field[{i}].rgba = {color}\n"
            else:
                # Just initialize first species if no config
                init_code += "tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]  # Red"
            
            return init_code
    
    def _generate_state_code(self) -> str:
        # Use the StateManager's proper code generation
        container_code = self.state_manager.generate_state_initialization_code()
        value_code = self.state_manager.generate_state_value_initialization_code()
        
        # Combine both parts
        if container_code and value_code and value_code != "# No state values to initialize":
            return f"{container_code}\n{value_code}"
        else:
            return container_code
    
    def _generate_simple_kernel(self) -> str:
        # Use the proper kernel generator with species conditions
        single_experts = [e for e in self.experts if e.expert_type == 'single']
        interaction_experts = [e for e in self.experts if e.expert_type == 'interaction']
        visual_experts = [e for e in self.experts if e.expert_type == 'visual']
        utility_experts = [e for e in self.experts if e.expert_type == 'utility']
        
        # Build species conditions mapping from expert info
        species_conditions = {}
        for expert in self.experts:
            if expert.applies_to_species is not None:
                species_conditions[expert.name] = expert.applies_to_species
        
        # Generate kernel code with species configuration and conditions
        kernel_code = self.kernel_generator.generate(
            single_expert_names=[e.name for e in single_experts],
            interaction_expert_names=[e.name for e in interaction_experts],
            expert_weights=self.expert_weights,
            tolvera_instance=self.tv,
            species_config=self.current_species_config,
            species_conditions=species_conditions,
            visual_expert_names=[e.name for e in visual_experts]
        )
        
        return kernel_code