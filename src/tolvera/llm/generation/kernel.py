import logging
from typing import List, Dict, Optional, Any
from jinja2 import Environment, FileSystemLoader
import os
from ..core.behavior_requirements import BehaviorRequirements, TemporalRequirement, PixelFieldRequirement

logger = logging.getLogger(__name__)


class IntegrationKernelGenerator:
    """
    Generates integration kernels that combine multiple expert behaviors.
    Also generates temporal update kernels and pixel manipulation kernels.
    """
    
    def __init__(self):
        # Set up Jinja2 environment
        templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
        self.env = Environment(loader=FileSystemLoader(templates_dir))
    
    def generate(
        self,
        single_expert_names: List[str],
        interaction_expert_names: List[str],
        expert_weights: Dict[str, float],
        tolvera_instance: Any,
        species_conditions: Optional[Dict[str, List[int]]] = None,
        species_config: Optional[Any] = None,
        visual_expert_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate an integration kernel that applies all experts.
        
        Args:
            single_expert_names: Names of single-particle experts
            interaction_expert_names: Names of interaction experts
            expert_weights: Weight for each expert
            tolvera_instance: Tölvera instance for configuration
            species_conditions: Optional mapping of expert names to species IDs they apply to
            species_config: Optional species configuration with interaction pairs
            
        Returns:
            Generated kernel code as string
        """
        # Build species-specific expert mapping using explicit species_conditions
        species_expert_map = {}
        all_species_experts = []  # Experts that apply to all species
        
        if species_conditions:
            # Use the explicit species conditions passed from behavior agent
            for expert_name in single_expert_names:
                if expert_name in species_conditions:
                    # This expert has specific species it applies to
                    species_list = species_conditions[expert_name]
                    if species_list:  # Not None and not empty
                        for species_id in species_list:
                            if species_id not in species_expert_map:
                                species_expert_map[species_id] = []
                            species_expert_map[species_id].append(expert_name)
                    else:
                        # Empty list means apply to all
                        all_species_experts.append(expert_name)
                else:
                    # No species condition = apply to all
                    all_species_experts.append(expert_name)
        else:
            # No species conditions provided, all experts apply to all species
            all_species_experts = single_expert_names.copy()
        
        # Check if we have ANY species-specific experts
        has_species_specific = bool(species_expert_map)
        
        # Load and render template using Jinja2
        template = self.env.get_template('kernel/integration_kernel.j2')
        
        return template.render(
            single_expert_names=single_expert_names,
            interaction_expert_names=interaction_expert_names,
            expert_weights=expert_weights,
            species_conditions=species_conditions,
            species_config=species_config,
            species_expert_map=species_expert_map,
            all_species_experts=all_species_experts,
            has_species_specific=has_species_specific
        )
    
    def generate_from_response(
        self,
        integration_kernel_model: Any,
        expert_weights: Dict[str, float],
        tolvera_instance: Any
    ) -> str:
        """
        Generate kernel from an IntegrationKernel model object.
        
        Args:
            integration_kernel_model: IntegrationKernel model instance
            expert_weights: Weight for each expert
            tolvera_instance: Tölvera instance
            
        Returns:
            Generated kernel code
        """
        return self.generate(
            single_expert_names=integration_kernel_model.single_experts,
            interaction_expert_names=integration_kernel_model.interaction_experts,
            expert_weights=expert_weights,
            tolvera_instance=tolvera_instance
        )
    
    # Legacy generate_temporal_kernel method removed - utility experts now handle temporal updates
    
    def _generate_ca_temporal_update(self, available_states: Dict[str, List[str]]) -> List[str]:
        """Generate cellular automaton temporal update."""
        particle_states = available_states.get('particle', [])
        lines = []
        
        if all(s in particle_states for s in ['is_alive', 'neighbor_count', 'next_state']):
            lines.extend([
                "    # Step 1: Count neighbors",
                "    for i in range(tv.pn):",
                "        if tv.p.field[i].active > 0:",
                "            count = 0",
                "            grid_x = tv.s.llm_particle.field[i].grid_x",
                "            grid_y = tv.s.llm_particle.field[i].grid_y",
                "            ",
                "            # Check 8 neighbors",
                "            for dx in range(-1, 2):",
                "                for dy in range(-1, 2):",
                "                    if dx != 0 or dy != 0:",
                "                        nx = (grid_x + dx) % grid_size",
                "                        ny = (grid_y + dy) % grid_size",
                "                        neighbor_idx = ny * grid_size + nx",
                "                        if neighbor_idx < tv.pn:",
                "                            if tv.s.llm_particle.field[neighbor_idx].is_alive == 1:",
                "                                count += 1",
                "            tv.s.llm_particle.field[i].neighbor_count = count",
                "",
                "    # Step 2: Apply rules",
                "    for i in range(tv.pn):",
                "        if tv.p.field[i].active > 0:",
                "            alive = tv.s.llm_particle.field[i].is_alive",
                "            neighbors = tv.s.llm_particle.field[i].neighbor_count",
                "            new_state = apply_rules(alive, neighbors)",
                "            tv.s.llm_particle.field[i].next_state = new_state",
                "",
                "    # Step 3: Commit state",
                "    for i in range(tv.pn):",
                "        if tv.p.field[i].active > 0:",
                "            tv.s.llm_particle.field[i].is_alive = tv.s.llm_particle.field[i].next_state",
                "            # Update visual",
                "            if tv.s.llm_particle.field[i].is_alive == 1:",
                "                tv.p.field[i].size = 3.0",
                "            else:",
                "                tv.p.field[i].size = 0.5"
            ])
        
        return lines
    
    def _generate_energy_temporal_update(self, available_states: Dict[str, List[str]]) -> List[str]:
        """Generate energy decay temporal update."""
        particle_states = available_states.get('particle', [])
        lines = []
        
        if 'energy' in particle_states:
            lines.extend([
                "    # Update energy",
                "    for i in range(tv.pn):",
                "        if tv.p.field[i].active > 0:",
                "            # Decay energy over time",
                "            tv.s.llm_particle.field[i].energy *= 0.995",
                "            ",
                "            # Handle exhaustion",
                "            if tv.s.llm_particle.field[i].energy < 1.0:",
                "                tv.p.field[i].active = 0.0"
            ])
        
        return lines
    
    def _generate_phase_temporal_update(self, available_states: Dict[str, List[str]]) -> List[str]:
        """Generate phase/cycle temporal update."""
        global_states = available_states.get('global', [])
        lines = []
        
        if 'day_phase' in global_states or 'time_of_day' in global_states:
            lines.extend([
                "    # Update day/night cycle",
                "    frame = tv.ctx.i[None]",
                "    cycle_length = 3600  # frames per full cycle",
                "    phase = (frame % cycle_length) / float(cycle_length)",
                "    ",
                "    if 'day_phase' in tv.s.llm_global.field[0]:",
                "        tv.s.llm_global.field[0].day_phase = phase",
                "    if 'time_of_day' in tv.s.llm_global.field[0]:",
                "        tv.s.llm_global.field[0].time_of_day = phase"
            ])
        
        return lines
    
    def _generate_generic_temporal_update(
        self,
        temporal_req: TemporalRequirement,
        available_states: Dict[str, List[str]]
    ) -> List[str]:
        """Generate generic temporal update based on requirements."""
        lines = []
        
        if temporal_req.states_to_update:
            lines.append("    # Generic temporal updates")
            lines.append("    for i in range(tv.pn):")
            lines.append("        if tv.p.field[i].active > 0:")
            
            for state_name in temporal_req.states_to_update:
                if state_name in available_states.get('particle', []):
                    lines.append(f"            # Update {state_name}")
                    lines.append(f"            tv.s.llm_particle.field[i].{state_name} *= 0.99")
        
        return lines
    
    def generate_pure_drawing_kernel(
        self,
        drawing_code: str,
        function_name: str = "draw"
    ) -> str:
        """
        Generate a kernel for pure drawing behaviors without particles.
        
        Args:
            drawing_code: The drawing code to execute
            function_name: Name of the drawing function
            
        Returns:
            Generated drawing kernel code
        """
        # Split drawing code into lines for template processing
        drawing_code_lines = drawing_code.split('\n')
        
        # Load and render template using Jinja2
        template = self.env.get_template('kernel/pure_drawing_kernel.j2')
        
        return template.render(
            function_name=function_name,
            drawing_code_lines=drawing_code_lines
        )
    
    def generate_drawing_kernel_from_experts(
        self,
        visual_expert_names: List[str],
        function_name: str = "draw"
    ) -> str:
        """
        Generate a drawing kernel that calls visual expert functions.
        
        Args:
            visual_expert_names: Names of visual expert functions
            function_name: Name of the drawing kernel
            
        Returns:
            Generated drawing kernel code
        """
        if not visual_expert_names:
            return ""
        
        # Load and render template using Jinja2
        template = self.env.get_template('kernel/drawing_kernel.j2')
        
        return template.render(
            function_name=function_name,
            visual_expert_names=visual_expert_names
        )
    
    def generate_utility_kernel(
        self,
        utility_expert_names: List[str],
        function_name: str = "update_utilities"
    ) -> str:
        """
        Generate a kernel that calls utility expert functions.
        
        Args:
            utility_expert_names: Names of utility expert functions
            function_name: Name of the utility kernel
            
        Returns:
            Generated utility kernel code
        """
        if not utility_expert_names:
            return ""
        
        # Load and render template using Jinja2
        template = self.env.get_template('kernel/utility_kernel.j2')
        
        return template.render(
            function_name=function_name,
            utility_expert_names=utility_expert_names
        )
    
    def generate_pixel_kernel(
        self,
        pixel_req: Optional[PixelFieldRequirement],
        pattern_type: str = "particle_system"
    ) -> Dict[str, str]:
        """
        Generate pixel manipulation kernels.
        
        Args:
            pixel_req: Pixel field requirements
            pattern_type: Pattern type for specialized pixel ops
            
        Returns:
            Dictionary of kernel names to code
        """
        if not pixel_req:
            return {}
        
        kernels = {}
        
        if pixel_req.needs_diffusion:
            template = self.env.get_template('kernel/pixel_diffusion.j2')
            kernels["diffuse_pheromones"] = template.render()
        
        if pixel_req.needs_decay:
            template = self.env.get_template('kernel/pixel_decay.j2')
            kernels["decay_pheromones"] = template.render()
        
        if pixel_req.needs_deposition:
            template = self.env.get_template('kernel/pixel_deposition.j2')
            kernels["deposit_trails"] = template.render()
        
        return kernels