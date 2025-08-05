
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .synthesizer import Synthesizer
from .state_manager import StateManager
from .decomposer import BehaviorDecomposer
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
    expert_type: str  # 'single', 'interaction', 'drawing', 'drawing_interaction'
    code: str
    draw_order: Optional[str] = None  # 'pre' or 'post' for drawing experts


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
        
        self.synthesizer = Synthesizer(model_name, tolvera_instance, api_key)
        self.state_manager = StateManager(tolvera_instance)
        self.species_manager = SpeciesManager(tolvera_instance)
        
        from .prompts import ContextAwarePromptBuilder
        prompt_builder = ContextAwarePromptBuilder()
        self.decomposer = BehaviorDecomposer(model_name, prompt_builder, api_key)
        
        self.kernel_generator = IntegrationKernelGenerator()
        self.sketch_generator = SketchGenerator()
        
        self.experts: List[ExpertInfo] = []
        self.expert_weights: Dict[str, float] = {}
        
        self.current_species_config = None
        
        self.temporal_updates: List[Any] = []
        
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
            for sid, name in new.species_names.items():
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
                if new.species_names and sid in new.species_names:
                    name = new.species_names[sid]
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
        
        # Create a synthesis description that includes implementation
        description = f"{component.description}. {component.implementation}"
        
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
            
            expert_info = ExpertInfo(
                name=component.expert_name,
                description=component.description,
                weight=weight,
                expert_type='interaction' if expert.is_interaction else 'single',
                code=code
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
            
            # Always decompose first to assess complexity (unless explicitly skipped)
            if self.decomposer and not skip_decomposition:
                try:
                    decomposed = await self.decomposer.decompose(description)
                    
                    # Check if this is simple or complex
                    if decomposed.is_simple:
                        logger.info(f"Behavior is simple, synthesizing single expert: {decomposed.components[0].expert_name}")
                        # For simple behaviors, use the concrete implementation from decomposer
                        if decomposed.components:
                            component = decomposed.components[0]
                            # Update description with implementation guidance
                            enhanced_description = f"{description}. Implementation guidance: {component.implementation}"
                            description = enhanced_description
                            # Store the desired expert name for later use
                            self._desired_expert_name = component.expert_name
                    else:
                        logger.info(f"Behavior is complex with {len(decomposed.components)} experts, using complex behavior path")
                        # Pass the decomposed result to add_complex_behavior
                        complex_result = await self.add_complex_behavior(
                            description, weight, 
                            decompose=False,  # Already decomposed
                            _decomposed=decomposed  # Pass the decomposition
                        )
                        # Normalize the result format to match add_behavior's expected format
                        expert_names = []
                        total_states = 0
                        for comp in complex_result.get('components', []):
                            expert_names.append(comp['expert_name'])
                            comp_result = comp.get('result', {})
                            if comp_result and isinstance(comp_result, dict):
                                total_states += comp_result.get('states_created', 0)
                        
                        return {
                            'success': True,
                            'experts_added': complex_result.get('total_experts', 0),
                            'expert_names': expert_names,
                            'states_created': total_states,
                            'species_count': len(self.current_species_config.species_ids) if self.current_species_config else 1
                        }
                except Exception as e:
                    logger.warning(f"Decomposition failed, continuing with direct synthesis: {e}")
                    self._current_decomposition = None
            
            # Get current available states
            available_states = self.state_manager.get_available_states()
            
            # Synthesize the behavior
            response = await self.synthesizer.synthesize_behavior(
                description,
                available_states
            )
            
            # Update species configuration - merge with existing
            if response.species_config:
                self.current_species_config = self._merge_species_configs(
                    self.current_species_config, 
                    response.species_config
                )
                logger.info(f"Updated species configuration: {len(self.current_species_config.species_ids)} species")
            
            # Create states if needed
            if response.states_needed:
                # Convert list of StateDefinition objects to spec format
                states_spec = {'global': {}, 'particle': {}, 'species': {}}
                for state_def in response.states_needed:
                    states_spec[state_def.category][state_def.name] = state_def
                
                self.state_manager.create_states_from_spec(states_spec)
                logger.info(f"Created states: {[s.name for s in response.states_needed]}")
            
            # Store temporal updates if any
            if response.temporal_update:
                self.temporal_updates.append(response.temporal_update)
                logger.info(f"Added temporal updates for: {list(response.temporal_update.frame_updates.keys())}")
            
            # Register experts
            experts_added = []
            for expert in response.experts:
                # Extract the actual Taichi code
                if hasattr(expert, 'to_code'):
                    code = expert.to_code()
                elif hasattr(expert, '_code'):
                    code = expert._code
                else:
                    code = ""
                
                # Use desired expert name from decomposition if available
                expert_name = getattr(self, '_desired_expert_name', None) or expert.name
                
                code = self._fix_expert_function_name(code, expert_name)
                
                expert_info = ExpertInfo(
                    name=expert_name,
                    description=expert.description,
                    weight=weight * expert.weight,
                    expert_type='interaction' if expert.is_interaction else 'single',
                    code=code
                )
                self.experts.append(expert_info)
                self.expert_weights[expert_name] = expert_info.weight
                experts_added.append(expert_name)
                logger.info(f"Registered expert: {expert_name} (weight={expert_info.weight})")
                
                # Clear the desired expert name after use
                if hasattr(self, '_desired_expert_name'):
                    delattr(self, '_desired_expert_name')
            
            # Regenerate integration kernel if we have experts
            if self.experts:
                await self._regenerate_kernel()
            
            result = {
                'success': True,
                'experts_added': len(experts_added),
                'expert_names': experts_added,
                'states_created': len(response.states_needed) if response.states_needed else 0,
                'species_count': len(response.species_config.species_ids) if response.species_config else 1
            }
            
            # Update trace node with results
            if node:
                node.output_data = result
            
            return result
    
    async def _regenerate_kernel(self):
        # Separate experts by type
        single_experts = [e for e in self.experts if e.expert_type == 'single']
        interaction_experts = [e for e in self.experts if e.expert_type == 'interaction']
        
        # Generate kernel code
        kernel_code = self.kernel_generator.generate(
            single_expert_names=[e.name for e in single_experts],
            interaction_expert_names=[e.name for e in interaction_experts],
            expert_weights=self.expert_weights,
            tolvera_instance=self.tv
        )
        
        # Compile and register the kernel
        # This would integrate with Tölvera's kernel system
        logger.info("Regenerated integration kernel with all experts")
    
    async def _regenerate_drawing_kernel(self):
        from ..generation.drawing import DrawingKernelGenerator
        
        # Separate drawing experts by type and order
        pre_draw = [e for e in self.experts if e.expert_type == 'drawing' and hasattr(e, 'draw_order') and e.draw_order == 'pre']
        post_draw = [e for e in self.experts if e.expert_type == 'drawing' and (not hasattr(e, 'draw_order') or e.draw_order == 'post')]
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
        
        # Compile and register the drawing kernel
        logger.info("Regenerated drawing kernel with all drawing experts")
    
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
    
    def _generate_temporal_update_kernel(self) -> str:
        available_states = self.state_manager.get_available_states()
        
        has_grid_states = self._has_grid_states(available_states)
        has_oscillator_states = self._has_oscillator_states(available_states)
        has_growth_states = self._has_growth_states(available_states)
        
        kernel_parts = self._create_temporal_kernel_header()
        
        if has_grid_states:
            kernel_parts.extend(self._generate_grid_update_code(available_states))
        
        if has_oscillator_states:
            kernel_parts.extend(self._generate_oscillator_update_code(available_states))
        
        if has_growth_states:
            kernel_parts.extend(self._generate_growth_update_code(available_states))
        
        if 'energy' in available_states.get('particle', []):
            kernel_parts.extend(self._generate_energy_update_code())
        
        if self.temporal_updates:
            kernel_parts.extend(self._generate_synthesis_temporal_updates(available_states))
        
        if len(kernel_parts) <= 6:
            return ""
        
        return "\n".join(kernel_parts)
    
    def _has_grid_states(self, available_states: Dict[str, List[str]]) -> bool:
        return any(s in available_states.get('particle', []) 
                  for s in ['grid_x', 'grid_y', 'is_alive', 'neighbor_count'])
    
    def _has_oscillator_states(self, available_states: Dict[str, List[str]]) -> bool:
        return any(s in available_states.get('particle', []) 
                  for s in ['phase', 'frequency'])
    
    def _has_growth_states(self, available_states: Dict[str, List[str]]) -> bool:
        return any(s in available_states.get('particle', []) 
                  for s in ['age', 'cell_type'])
    
    def _create_temporal_kernel_header(self) -> List[str]:
        return [
            "@ti.kernel", 
            "def update_temporal_states():",
            "    frame = tv.ctx.i[None]",
            "    fps = 60.0",
            "    time = ti.cast(frame, ti.f32) / fps",
            "    dt = 1.0 / fps",
            ""
        ]
    
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
                    
                    # Create a synthesis description that includes implementation guidance
                    synthesis_description = f"{component.description}. Implementation: {component.implementation}"
                    
                    # Pass context requirements to synthesizer
                    synthesis_context.update({
                        "current_component": component.expert_name,
                        "constraints": component.context_requirements or [],
                        "previous_experts": synthesis_context["previous_experts"]
                    })
                    
                    # Check if it's a drawing behavior
                    if component.expert_type == 'visual':
                        result = await self.add_drawing_behavior(
                            synthesis_description,
                            component_weight
                        )
                    else:
                        # Synthesize with context - need to update synthesizer to accept context
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
    
    def generate_sketch(
        self, 
        description: str,
        filename: Optional[str] = None,
        use_timestamp: bool = True
    ) -> tuple[str, str]:
        # Collect all expert code
        expert_code = "\n\n".join(e.code for e in self.experts)
        
        # Generate initialization code
        init_code = self._generate_init_code()
        
        # Generate state code
        state_code = self._generate_state_code()
        
        # Generate kernel (simplified for now)
        kernel_code = self._generate_simple_kernel()
        
        # Generate temporal update kernel
        temporal_kernel_code = self._generate_temporal_update_kernel()
        
        # Generate complete sketch
        sketch = self.sketch_generator.generate(
            description=description,
            experts=[expert_code],
            kernel=kernel_code,
            init_code=init_code,
            state_code=state_code,
            temporal_code=temporal_kernel_code
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
                base_name = filename.replace('.py', '') if filename else 'generated_sketch'
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
            
            # Pass the species config for proper color assignment
            return self.species_manager.get_initialization_code(
                species_info,
                init_type=init_type,
                grid_size=grid_size,
                species_config=self.current_species_config
            )
        else:
            # Fallback to simple initialization
            init_code = """# Initialize particles
tv.p.randomise()

# Initialize default species colors
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
        # Separate experts by type
        single_experts = [e for e in self.experts if e.expert_type == 'single']
        interaction_experts = [e for e in self.experts if e.expert_type == 'interaction']
        drawing_experts = [e for e in self.experts if e.expert_type.startswith('drawing')]
        
        kernel_parts = ["@ti.kernel", "def apply_all_experts():"]
        
        if single_experts or interaction_experts:
            kernel_parts.append("    for i in range(tv.pn):")
            kernel_parts.append("        if tv.p.field[i].active > 0:")
            kernel_parts.append("            pos = tv.p.field[i].pos")
            kernel_parts.append("            vel = tv.p.field[i].vel")
            kernel_parts.append("            mass = tv.p.field[i].mass")
            kernel_parts.append("            species = tv.p.field[i].species")
            kernel_parts.append("            ")
            kernel_parts.append("            # Initialize total force")
            kernel_parts.append("            total_force = ti.math.vec2(0.0, 0.0)")
            kernel_parts.append("            ")
            
            # Add single-particle expert calls
            if single_experts:
                kernel_parts.append("            # Single-particle behaviors")
                for expert in single_experts:
                    weight = self.expert_weights.get(expert.name, 1.0)
                    kernel_parts.append(f"            total_force += {expert.name}(pos, vel, mass, species, i) * {weight}")
            
            # Add interaction expert calls
            if interaction_experts:
                kernel_parts.append("            ")
                kernel_parts.append("            # Interaction behaviors")
                kernel_parts.append("            for j in range(tv.pn):")
                kernel_parts.append("                if i != j and tv.p.field[j].active > 0:")
                for expert in interaction_experts:
                    weight = self.expert_weights.get(expert.name, 1.0)
                    kernel_parts.append(f"                    total_force += {expert.name}(tv.p.field[i], tv.p.field[j]) * {weight}")
            
            kernel_parts.append("            ")
            kernel_parts.append("            # Apply the combined force using F = ma")
            kernel_parts.append("            dt = 0.016  # ~60fps timestep")
            kernel_parts.append("            if mass > 0:")
            kernel_parts.append("                acceleration = total_force / mass")
            kernel_parts.append("                tv.p.field[i].vel += acceleration * dt")
            kernel_parts.append("            ")
            kernel_parts.append("            # Apply damping")
            kernel_parts.append("            tv.p.field[i].vel *= 0.98")
            kernel_parts.append("            ")
            kernel_parts.append("            # Update position based on new velocity")
            kernel_parts.append("            tv.p.field[i].pos += tv.p.field[i].vel * dt")
        
        # Add drawing expert calls if any
        if drawing_experts:
            if single_experts or interaction_experts:
                kernel_parts.append("")
            kernel_parts.append("    # Drawing behaviors")
            kernel_parts.append("    for i in range(tv.pn):")
            kernel_parts.append("        if tv.p.field[i].active > 0:")
            for expert in drawing_experts:
                if expert.expert_type == 'drawing':
                    kernel_parts.append(f"            {expert.name}(tv.px, tv.p.field[i], i)")
        
        return "\n".join(kernel_parts)