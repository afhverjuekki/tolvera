import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from .code_generator import CodeGenerator
from .state_manager import StateManager
from .behavior_analyzer import BehaviorAnalyzer
from .species_manager import SpeciesManager
from ..templates.template_renderer import TemplateRenderer
from .sketch_refiner import SketchRefiner
from .behavior_registry import ExpertInfo, ExpertRegistry
from ..debug.tracing import get_collector


class BehaviorOrchestrator:
    """
    Main orchestrator for synthesizing particle behaviors from natural language.
    
    This class coordinates the synthesis pipeline, delegating specific tasks to
    specialized components while maintaining the overall workflow.
    """
    
    def __init__(
        self,
        tolvera_instance,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None
    ):
        """
        Initialize the behavior agent.
        
        Args:
            tolvera_instance: Tölvera instance for particle system access
            model_name: Name of the LLM model to use
            api_key: Optional API key for the model
        """
        self.tv = tolvera_instance
        self.model_name = model_name
        
        # Determine provider
        from .llm_factory import ModelFactory
        self.provider = ModelFactory.get_provider_for_model(model_name)
        
        # Initialize core components
        self.code_generator = CodeGenerator(model_name, tolvera_instance, api_key)
        self.state_manager = StateManager(tolvera_instance)
        self.species_manager = SpeciesManager(tolvera_instance)
        self.behavior_analyzer = BehaviorAnalyzer(model_name, self.provider, api_key=api_key)
        self.template_renderer = TemplateRenderer()
        self.sketch_refiner = SketchRefiner(model_name, api_key)
        
        # Initialize expert registry
        self.expert_registry = ExpertRegistry()
        
        # Configuration tracking
        self.current_species_config = None
        self.synthesized_helpers = {}  # Track helper functions
    
    async def add_behavior(
        self,
        description: str,
        weight: float = 1.0,
        skip_decomposition: bool = False
    ) -> Dict[str, Any]:
        """
        Add a new behavior from natural language description.
        
        Args:
            description: Natural language behavior description
            weight: Weight for this behavior in integration
            skip_decomposition: Skip decomposition check
            
        Returns:
            Dictionary with synthesis results
        """
        collector = get_collector()
        
        with collector.trace_node("add_behavior", "synthesis", description=description) as node:
            
            # Analyze if needed
            analyzed = None
            if not skip_decomposition and self.behavior_analyzer:
                try:
                    analyzed = await self.behavior_analyzer.analyze(description)
                except Exception:
                    pass
            
            # Use comprehensive synthesis pipeline
            result = await self._synthesize_complete_behavior(description, weight, decomposition=analyzed)
            
            # Ensure result is valid
            if result is None:
                result = self._create_error_result()
            
            # Build final result
            final_result = self._format_synthesis_result(result)
            
            # Regenerate kernels if needed
            if self.expert_registry.experts:
                await self._regenerate_kernels()
            
            # Update trace
            if node:
                node.output_data = final_result
            
            return final_result
    
    def generate_sketch(
        self,
        description: str,
        filename: Optional[str] = None,
        use_timestamp: bool = True
    ) -> Tuple[str, str]:
        """
        Generate a complete runnable sketch.
        
        Args:
            description: Behavior description
            filename: Optional output filename
            use_timestamp: Whether to add timestamp to filename
            
        Returns:
            Tuple of (sketch_code, file_path)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context - need to handle this properly
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.generate_sketch_async(description, filename, use_timestamp)
                )
            else:
                # We can run async
                return loop.run_until_complete(
                    self.generate_sketch_async(description, filename, use_timestamp)
                )
        except RuntimeError:
            # No event loop - create one
            return asyncio.run(
                self.generate_sketch_async(description, filename, use_timestamp)
            )
    
    async def generate_sketch_async(
        self,
        description: str,
        filename: Optional[str] = None,
        use_timestamp: bool = True
    ) -> Tuple[str, str]:
        """
        Async sketch generation with all components.
        
        Args:
            description: Behavior description
            filename: Optional output filename
            use_timestamp: Whether to add timestamp to filename
            
        Returns:
            Tuple of (sketch_code, file_path)
        """
        # Collect all code elements
        code_elements = self._collect_code_elements()
        
        # Prepare metadata
        metadata = self._prepare_sketch_metadata(description)
        
        # Generate kernels
        kernels = self._generate_kernels()
        
        # Build complete sketch
        sketch = self.template_renderer.render_sketch(
            description=description,
            **code_elements,
            **kernels,
            **metadata
        )
        
        # Apply refinements if configured
        sketch = await self._apply_refinements(sketch, description)
        
        # Save to file
        return self._save_to_file(sketch, filename, use_timestamp)
    
    def get_expert_info(self) -> List[Dict[str, Any]]:
        """Get information about registered experts."""
        return self.expert_registry.get_expert_info_list()
    
    # ========== Core Synthesis Pipeline (Private) ==========
    
    async def _synthesize_complete_behavior(
        self,
        description: str,
        weight: float = 1.0,
        decomposition=None
    ) -> Dict[str, Any]:
        """
        Complete synthesis pipeline for a behavior.
        
        Args:
            description: Natural language behavior description
            weight: Weight for this behavior
            decomposition: Optional pre-computed decomposition
            
        Returns:
            Dictionary with synthesis results
        """
        collector = get_collector()
        
        with collector.trace_node("synthesize_complete_behavior", "synthesis", description=description):
            
            # Step 1: Decompose and analyze
            analyzed = await self._decompose_and_analyze(description, decomposition)
            
            # Step 2: Prepare states
            await self._prepare_states(analyzed)
            
            # Step 3: Build synthesis context
            synthesis_context = self._build_synthesis_context()
            
            # Step 4: Synthesize components
            return await self._synthesize_components(analyzed, weight, synthesis_context)
    
    async def _decompose_and_analyze(
        self,
        description: str,
        decomposition=None
    ) -> Any:
        """Analyze behavior and extract requirements."""
        if decomposition:
            analyzed = decomposition
        elif self.behavior_analyzer:
            try:
                analyzed = await self.behavior_analyzer.analyze(description)
            except Exception:
                return None
        else:
            analyzed = None
        
        # Extract metadata from analysis
        if analyzed:
            await self._extract_decomposition_metadata(analyzed)
        
        return analyzed
    
    async def _prepare_states(self, decomposed) -> None:
        """Prepare and create all required states."""
        if not decomposed:
            return
        
        all_state_specs = []
        
        # Collect states from decomposition
        if hasattr(decomposed, 'suggested_states') and decomposed.suggested_states:
            for state_spec in decomposed.suggested_states:
                from .data_models import StateDefinition
                state_def = StateDefinition(
                    name=state_spec.name,
                    category=state_spec.category,
                    type=state_spec.type,
                    min=state_spec.min,
                    max=state_spec.max,
                    description=f"State from decomposer: {state_spec.name}"
                )
                spec = {state_spec.category: {state_spec.name: state_def}}
                all_state_specs.append(spec)
        
        # Analyze states for components
        if hasattr(decomposed, 'components'):
            for component in decomposed.components:
                states = await self._analyze_component_states(component)
                all_state_specs.extend(states)
        
        # Create all states
        if all_state_specs:
            self.state_manager.collect_and_create_states(all_state_specs)
    
    def _build_synthesis_context(self) -> Dict[str, Any]:
        """Build shared context for synthesis."""
        # Get current species info if available
        species_info = None
        if self.current_species_config:
            # Build species_info from current configuration
            from .species_manager import SpeciesInfo
            species_info = SpeciesInfo(
                species_ids=self.current_species_config.species_ids,
                species_names={m.species_id: m.name for m in self.current_species_config.species_names} if self.current_species_config.species_names else {},
                interaction_pairs=self.current_species_config.interaction_pairs or [],
                species_behaviors={},  # Not stored in config
                requires_all_species=self.current_species_config.requires_all_species if hasattr(self.current_species_config, 'requires_all_species') else False,
                total_count=len(self.current_species_config.species_ids),
                color_hints={}  # Not stored in config
            )
        
        return {
            "available_states": self.state_manager.get_available_states(),
            "synthesized_helpers": self.synthesized_helpers.copy(),
            "existing_experts": [e.name for e in self.expert_registry.experts],
            "pattern_type": "unknown",
            "pattern_confidence": 0.0,
            "shared_parameters": {},
            "constraints": [],
            "states_already_created": True,
            "species_info": species_info  # Pass species info to code generator
        }
    
    async def _synthesize_components(
        self,
        decomposed,
        weight: float,
        synthesis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize all components from decomposition."""
        if not decomposed or not hasattr(decomposed, 'components') or not decomposed.components:
            return self._create_error_result()
        
        # Add all components to context for cross-referencing
        synthesis_context['all_components'] = decomposed.components
        
        # Synthesize each component
        results = []
        for component in decomposed.components:
            result = await self._synthesize_single_component(
                component, weight, synthesis_context
            )
            results.append({
                'expert_name': component.expert_name,
                'description': component.description,
                'result': result
            })
        
        # Aggregate results
        return self._aggregate_synthesis_results(results)
    
    async def _synthesize_single_component(
        self,
        component: Any,
        weight: float,
        synthesis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize a single component by delegating to CodeGenerator."""
        # Prepare component-specific context
        context = dict(synthesis_context)
        context['component'] = component
        context['component_description'] = component.description
        context['component_behavioral_guidance'] = component.implementation
        context['expert_type'] = component.expert_type
        
        # Add implementation details if available
        if hasattr(component, 'implementation_details'):
            context['implementation_details'] = component.implementation_details
        if hasattr(component, 'parameters'):
            context['parameters'] = component.parameters
        
        try:
            # Delegate all synthesis to CodeGenerator
            if component.expert_type in ['temporal_update', 'state_update', 'utility']:
                response = await self.code_generator.synthesize_utility_expert(
                    component.implementation,
                    synthesis_context.get('available_states', {}),
                    context,
                    component.expert_type
                )
            else:
                response = await self.code_generator.synthesize_behavior(
                    component.implementation,
                    synthesis_context.get('available_states', {}),
                    context,
                    component.expert_name,
                    skip_state_analysis=True
                )
            
            if response is None:
                return self._create_error_result()
            
            # Register the expert
            self._register_synthesized_expert(component, response, weight)
            
            # Handle type-specific post-processing
            await self._handle_expert_post_processing(component.expert_type)
            
            return {
                'success': True,
                'experts_added': 1,
                'expert_names': [component.expert_name],
                'states_created': 0
            }
            
        except Exception:
            return self._create_error_result()
    
    # ========== Helper Methods ==========
    
    # ========== Helper Methods ==========
    
    
    def _register_synthesized_expert(self, component, response, weight: float) -> None:
        """Register a synthesized expert."""
        if not response or not hasattr(response, 'experts') or not response.experts:
            return
        
        for expert in response.experts:
            # Get the code - either pre-generated or render it
            if hasattr(expert, 'code') and expert.code:
                code = expert.code
            else:
                code = self.template_renderer.render_expert_function(expert)
            
            # Determine expert type
            if component.expert_type == 'visual':
                expert_type = 'drawing'
            elif component.expert_type in ['temporal_update', 'state_update', 'utility']:
                expert_type = 'utility'
            elif expert.is_interaction:
                expert_type = 'interaction'
            else:
                expert_type = 'single'
            
            # Create and register expert info
            expert_info = ExpertInfo(
                name=component.expert_name,
                description=component.description,
                weight=weight,
                expert_type=expert_type,
                code=code,
                applies_to_species=component.applies_to_species if hasattr(component, 'applies_to_species') else None,
                draw_order='post' if expert_type == 'drawing' else None
            )
            self.expert_registry.register(expert_info)
    
    async def _extract_decomposition_metadata(self, decomposed) -> None:
        """Extract and store metadata from decomposition."""
        # Extract particle count
        if hasattr(decomposed, 'particle_count') and decomposed.particle_count:
            self.detected_particle_count = decomposed.particle_count
        
        # Extract species configuration
        if hasattr(decomposed, 'species_info') and decomposed.species_info:
            await self._process_species_info(decomposed.species_info)
        
        # Extract speed specification
        if hasattr(decomposed, 'speed_spec'):
            self.current_speed_spec = decomposed.speed_spec
    
    async def _process_species_info(self, species_info) -> None:
        """Process species information from decomposition."""
        from .data_models import SpeciesNameMapping
        
        # Convert decomposer's SpeciesNameMapping objects to models.py's SpeciesNameMapping
        species_names_list = None
        if species_info.species_names:
            species_names_list = []
            for item in species_info.species_names:
                if hasattr(item, 'species_id') and hasattr(item, 'name'):
                    species_names_list.append(SpeciesNameMapping(
                        species_id=item.species_id,
                        name=item.name
                    ))
                elif isinstance(item, dict):
                    species_names_list.append(SpeciesNameMapping(
                        species_id=item.get('species_id', 0),
                        name=item.get('name', 'unnamed')
                    ))
                else:
                    pass
        
        # Delegate to species_manager for configuration building
        self.current_species_config = await self.species_manager.build_species_configuration(
            species_info, species_names_list
        )
    
    async def _analyze_component_states(self, component) -> List[Dict[str, Any]]:
        """Analyze states needed for a component."""
        state_specs = []
        
        # Add required states from component
        if hasattr(component, 'required_states') and component.required_states:
            for state_spec in component.required_states:
                from .data_models import StateDefinition
                state_def = StateDefinition(
                    name=state_spec.name,
                    category=state_spec.category,
                    type=state_spec.type,
                    min=state_spec.min,
                    max=state_spec.max,
                    description=f"State for {component.expert_name}"
                )
                spec = {state_spec.category: {state_spec.name: state_def}}
                state_specs.append(spec)
        
        # Analyze description for additional states
        description = f"{component.description}. {component.implementation}"
        states_analysis = await self.code_generator.analyze_states_needed(
            description,
            expert_type=component.expert_type
        )
        
        # Convert analysis to specs
        for category in ['global', 'particle', 'species']:
            if category in states_analysis and states_analysis[category]:
                for state_name, state_def in states_analysis[category].items():
                    spec = {category: {state_name: state_def}}
                    state_specs.append(spec)
        
        return state_specs
    
    async def _handle_expert_post_processing(self, expert_type: str) -> None:
        """Handle type-specific post-processing after expert registration."""
        if expert_type == 'visual':
            # Regenerate drawing kernel for visual experts
            visual_experts = self.expert_registry.get_by_type('drawing')
            if visual_experts:
                self.template_renderer.render_drawing_kernel(
                    visual_expert_names=[e.name for e in visual_experts]
                )
    
    async def _regenerate_kernels(self) -> None:
        """Regenerate all kernels based on registered experts."""
        kernel_params = self.expert_registry.get_kernel_params()
        
        # Generate integration kernel
        if kernel_params['single_expert_names'] or kernel_params['interaction_expert_names']:
            integration_params = {
                'single_expert_names': kernel_params['single_expert_names'],
                'interaction_expert_names': kernel_params['interaction_expert_names'],
                'visual_expert_names': kernel_params['visual_expert_names'],
                'expert_weights': kernel_params['expert_weights'],
                'species_conditions': kernel_params['species_conditions']
            }
            self.template_renderer.render_integration_kernel(
                **integration_params,
                species_config=self.current_species_config
            )
        
        # Generate drawing kernel
        if kernel_params['visual_expert_names']:
            self.template_renderer.render_drawing_kernel(
                visual_expert_names=kernel_params['visual_expert_names']
            )
        
        # Generate utility kernel
        if kernel_params['utility_expert_names']:
            self.template_renderer.render_utility_kernel(
                utility_expert_names=kernel_params['utility_expert_names']
            )
    
    # ========== Sketch Generation Methods ==========
    
    def _collect_code_elements(self) -> Dict[str, Any]:
        """Collect all code elements for sketch generation."""
        # Helper functions
        helper_code = ""
        if self.synthesized_helpers:
            helper_parts = ["# Helper Functions"]
            for _, code in self.synthesized_helpers.items():
                helper_parts.append(f"\n{code}\n")
            helper_code = "\n".join(helper_parts)
        
        # Expert code
        force_experts = self.expert_registry.get_by_type('single') + self.expert_registry.get_by_type('interaction')
        expert_code = "\n\n".join(e.code for e in force_experts)
        
        # Combine helpers and experts
        if helper_code:
            combined_expert_code = f"{helper_code}\n\n{expert_code}"
        else:
            combined_expert_code = expert_code
        
        # Visual and utility experts
        visual_experts = self.expert_registry.get_by_type('drawing') + self.expert_registry.get_by_type('visual')
        drawing_code = "\n\n".join(e.code for e in visual_experts)
        
        utility_experts = self.expert_registry.get_by_type('utility')
        utility_code = "\n\n".join(e.code for e in utility_experts)
        
        return {
            'experts': [combined_expert_code] if combined_expert_code else [],
            'drawing_code': drawing_code,
            'utility_code': utility_code,
            'has_non_visual_experts': len(force_experts) > 0
        }
    
    def _prepare_sketch_metadata(self, description: str) -> Dict[str, Any]:  # noqa: ARG002
        """Prepare metadata for sketch generation."""
        # Delegate to state_manager for initialization components
        components = self.state_manager.generate_initialization_components(
            particle_count=getattr(self, 'detected_particle_count', self.tv.pn),
            species_config=self.current_species_config
        )
        
        return {
            'init_code': components['init_code'],
            'state_code': components['state_code'],
            'config_code': components['config_code'],
            'temporal_code': "",
            'respawn_code': "",
            'environmental_fields': ""
        }
    
    def _generate_kernels(self) -> Dict[str, str]:
        """Generate all kernel code."""
        kernel_params = self.expert_registry.get_kernel_params()
        
        # Integration kernel
        kernel = ""
        if kernel_params['single_expert_names'] or kernel_params['interaction_expert_names']:
            integration_params = {
                'single_expert_names': kernel_params['single_expert_names'],
                'interaction_expert_names': kernel_params['interaction_expert_names'],
                'visual_expert_names': kernel_params['visual_expert_names'],
                'expert_weights': kernel_params['expert_weights'],
                'species_conditions': kernel_params['species_conditions']
            }
            kernel = self.template_renderer.render_integration_kernel(
                **integration_params,
                species_config=self.current_species_config
            )
        
        # Drawing kernel
        drawing_kernel = ""
        if kernel_params['visual_expert_names']:
            drawing_kernel = self.template_renderer.render_drawing_kernel(
                visual_expert_names=kernel_params['visual_expert_names']
            )
        
        # Utility kernel
        utility_kernel = ""
        if kernel_params['utility_expert_names']:
            utility_kernel = self.template_renderer.render_utility_kernel(
                utility_expert_names=kernel_params['utility_expert_names']
            )
        
        return {
            'kernel': kernel,
            'drawing_kernel': drawing_kernel,
            'utility_kernel': utility_kernel
        }
    
    async def _apply_refinements(self, sketch: str, description: str) -> str:
        """Apply architectural refinements to sketch."""
        
        pattern_info = self.sketch_refiner.detect_architectural_pattern(description, sketch)
        
        if pattern_info['needs_refinement'] and pattern_info['primary_pattern']:
            pass
            
            result = await self.sketch_refiner.refine_to_architecture(
                sketch_code=sketch,
                description=description,
                pattern=pattern_info['primary_pattern']
            )
            
            if result['success']:
                return result['refined_code']
        
        return sketch
    
    def _save_to_file(
        self,
        sketch: str,
        filename: Optional[str],
        use_timestamp: bool
    ) -> Tuple[str, str]:
        """Save sketch to file."""
        if not filename and not use_timestamp:
            return sketch, ""
        
        # Create directory
        sketch_dir = Path('examples/generated_sketches')
        sketch_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file path
        if filename and not use_timestamp:
            file_path = Path(filename)
            if not file_path.parent.name or file_path.parent == Path('.'):
                file_path = sketch_dir / file_path.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = filename.replace('.py', '') if filename else 'generated_sketch'
            file_path = sketch_dir / f"{base_name}_{timestamp}.py"
        
        # Write file
        with open(file_path, 'w') as f:
            f.write(sketch)
        
        return sketch, str(file_path)
    
    
    # ========== Error Handling Helpers ==========
    
    def _aggregate_synthesis_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple component syntheses."""
        total_experts = sum(
            r['result'].get('experts_added', 0)
            for r in results
            if r.get('result') and isinstance(r['result'], dict)
        )
        
        expert_names = [r['expert_name'] for r in results]
        
        return {
            'success': True,
            'pattern_type': 'unknown',
            'experts_added': total_experts,
            'components': results,
            'states_created': 0,
            'helpers_synthesized': len(self.synthesized_helpers),
            'expert_names': expert_names
        }
    
    def _create_error_result(self) -> Dict[str, Any]:
        """Create a standard error result."""
        return {
            'success': False,
            'experts_added': 0,
            'expert_names': [],
            'states_created': 0,
            'pattern_type': 'unknown',
            'pattern_confidence': 0.0
        }
    
    def _format_synthesis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format synthesis result for consistent output."""
        expert_names = result.get('expert_names', [])
        
        # Extract names from components if needed
        if not expert_names and 'components' in result:
            for comp in result['components']:
                if isinstance(comp, dict):
                    if 'result' in comp and 'expert_names' in comp['result']:
                        expert_names.extend(comp['result']['expert_names'])
                    elif 'expert_name' in comp:
                        expert_names.append(comp['expert_name'])
        
        return {
            'success': result.get('success', True),
            'experts_added': result.get('experts_added', 0),
            'expert_names': expert_names,
            'states_created': result.get('states_created', 0),
            'species_count': len(self.current_species_config.species_ids) if self.current_species_config else 1,
            'pattern_type': result.get('pattern_type', 'unknown'),
            'pattern_confidence': result.get('pattern_confidence', 0.0)
        }