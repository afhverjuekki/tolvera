import logging
from typing import List, Dict, Optional, Any
from ..core.behavior_requirements import BehaviorRequirements, TemporalRequirement, PixelFieldRequirement

logger = logging.getLogger(__name__)


class IntegrationKernelGenerator:
    """
    Generates integration kernels that combine multiple expert behaviors.
    Also generates temporal update kernels and pixel manipulation kernels.
    """
    
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
        lines = ["@ti.kernel", "def apply_all_experts():"]
        
        # Main particle loop
        lines.append("    for i in range(tv.pn):")
        lines.append("        if tv.p.field[i].active > 0:")
        
        # Extract particle properties
        lines.extend([
            "            pos = tv.p.field[i].pos",
            "            vel = tv.p.field[i].vel", 
            "            mass = tv.p.field[i].mass",
            "            species = tv.p.field[i].species",
            "",
            "            # Initialize total force",
            "            total_force = ti.math.vec2(0.0, 0.0)"
        ])
        
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
        
        # Visual/drawing experts are handled in a separate drawing kernel
        # They are not called here in the integration kernel
        # (Drawing happens in a different phase of the render loop)
        
        # Apply single-particle experts with species conditions
        if single_expert_names:
            lines.append("")
            lines.append("            # Single-particle behaviors")
            
            # Check if we have ANY species-specific experts
            has_species_specific = bool(species_expert_map)
            
            if has_species_specific:
                # Generate species-specific application
                # First, apply species-specific experts
                for species_id in sorted(species_expert_map.keys()):
                    experts = species_expert_map[species_id]
                    if experts:
                        species_name = ""
                        if species_config and hasattr(species_config, 'species_names'):
                            species_name = species_config.species_names.get(species_id, f"species_{species_id}")
                        else:
                            species_name = f"species_{species_id}"
                        
                        # Special handling for food particles (typically species 2)
                        if species_id == 2 and "food" in species_name.lower():
                            lines.append(f"            if species == {species_id}:  # {species_name} (stationary)")
                            lines.append(f"                # Food particles don't move")
                            lines.append(f"                total_force = ti.math.vec2(0.0, 0.0)")
                        else:
                            lines.append(f"            if species == {species_id}:  # {species_name}")
                            for expert_name in experts:
                                weight = expert_weights.get(expert_name, 1.0)
                                lines.append(f"                total_force += {expert_name}(pos, vel, mass, species, i) * {weight}")
                
                # Then apply all-species experts (if any)
                if all_species_experts:
                    lines.append("            # Behaviors that apply to all species")
                    for expert_name in all_species_experts:
                        weight = expert_weights.get(expert_name, 1.0)
                        lines.append(f"            total_force += {expert_name}(pos, vel, mass, species, i) * {weight}")
            else:
                # No species-specific mapping, apply all experts to all particles
                for expert_name in single_expert_names:
                    weight = expert_weights.get(expert_name, 1.0)
                    lines.append(f"            total_force += {expert_name}(pos, vel, mass, species, i) * {weight}")
        
        # Apply interaction experts
        if interaction_expert_names:
            lines.append("")
            lines.append("            # Apply interaction experts")
            lines.append("            for j in range(tv.pn):")
            lines.append("                if i != j and tv.p.field[j].active > 0:")
            lines.append("                    species_j = tv.p.field[j].species")
            
            for expert_name in interaction_expert_names:
                weight = expert_weights.get(expert_name, 1.0)
                
                # Check if this expert should only apply to certain species
                if species_conditions and expert_name in species_conditions:
                    species_list = species_conditions[expert_name]
                    if species_list:  # Has specific species restrictions
                        # This interaction expert only applies when particle i is of certain species
                        lines.append(f"                    # {expert_name} applies to specific species")
                        condition_parts = [f"species == {sid}" for sid in species_list]
                        condition = " or ".join(condition_parts)
                        lines.append(f"                    if {condition}:")
                        lines.append(f"                        total_force += {expert_name}(tv.p.field[i], tv.p.field[j]) * {weight}")
                    else:
                        # Empty list or None means apply to all
                        lines.append(f"                    total_force += {expert_name}(tv.p.field[i], tv.p.field[j]) * {weight}")
                elif species_config and hasattr(species_config, 'interaction_pairs') and species_config.interaction_pairs:
                    # Use interaction pairs if available and no explicit species conditions
                    conditions = []
                    for s1, s2 in species_config.interaction_pairs:
                        conditions.append(f"(species == {s1} and species_j == {s2})")
                        if s1 != s2:  # Add reverse pair for symmetric interactions
                            conditions.append(f"(species == {s2} and species_j == {s1})")
                    
                    if conditions:
                        condition_str = " or ".join(conditions)
                        lines.append(f"                    if {condition_str}:")
                        lines.append(f"                        total_force += {expert_name}(tv.p.field[i], tv.p.field[j]) * {weight}")
                    else:
                        lines.append(f"                    total_force += {expert_name}(tv.p.field[i], tv.p.field[j]) * {weight}")
                else:
                    # No restrictions, apply to all
                    lines.append(f"                    total_force += {expert_name}(tv.p.field[i], tv.p.field[j]) * {weight}")
        
        # Apply forces to particles and integrate position
        lines.extend([
            "",
            "            # Apply force (F = ma, so a = F/m)",
            "            dt = 0.016  # 60 FPS timestep",
            "            acceleration = total_force / mass if mass > 0 else total_force",
            "            tv.p.field[i].vel += acceleration * dt",
            "            tv.p.field[i].vel *= 0.98  # Small damping for stability",
            "",
            "            # Update position from velocity (CRITICAL for movement)",
            "            tv.p.field[i].pos += tv.p.field[i].vel * tv.p.field[i].speed * dt"
        ])
        
        return "\n".join(lines)
    
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
        lines = [
            "@ti.kernel",
            f"def {function_name}():",
            '    """Pure drawing kernel."""'
        ]
        
        # Add the drawing code with proper indentation
        for line in drawing_code.split('\n'):
            if line.strip():
                lines.append(f"    {line}")
        
        return "\n".join(lines)
    
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
        
        lines = [
            "@ti.kernel",
            f"def {function_name}():",
            '    """Drawing kernel that executes visual behaviors."""'
        ]
        
        # Call each visual expert directly without any parameters
        # Pure drawing functions don't need particle data
        for expert_name in visual_expert_names:
            lines.append(f"    {expert_name}()  # Execute drawing function")
        
        return "\n".join(lines)
    
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
        
        lines = [
            "@ti.kernel",
            f"def {function_name}():",
            '    """Utility kernel that executes utility functions (temporal updates, state updates, etc.)."""'
        ]
        
        # Call each utility expert directly
        # Utility functions don't take particle parameters
        for expert_name in utility_expert_names:
            lines.append(f"    {expert_name}()  # Execute utility function")
        
        return "\n".join(lines)
    
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
            kernels["diffuse_pheromones"] = self._generate_diffusion_kernel()
        
        if pixel_req.needs_decay:
            kernels["decay_pheromones"] = self._generate_decay_kernel()
        
        if pixel_req.needs_deposition:
            kernels["deposit_trails"] = self._generate_deposition_kernel()
        
        return kernels
    
    def _generate_diffusion_kernel(self) -> str:
        """Generate pheromone diffusion kernel."""
        return """@ti.kernel
def diffuse_pheromones():
    \"\"\"Diffuse pheromone trails.\"\"\"
    # Create temp buffer for diffusion
    for x in range(tv.x):
        for y in range(tv.y):
            sum_r = 0.0
            sum_g = 0.0
            sum_b = 0.0
            count = 0
            
            # Sample 3x3 neighborhood
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx = (x + dx) % tv.x
                    ny = (y + dy) % tv.y
                    sum_r += tv.px.px.rgba[nx, ny][0]
                    sum_g += tv.px.px.rgba[nx, ny][1]
                    sum_b += tv.px.px.rgba[nx, ny][2]
                    count += 1
            
            # Apply diffusion with small coefficient
            if count > 0:
                diffusion_rate = 0.05
                tv.px.px.rgba[x, y][0] = tv.px.px.rgba[x, y][0] * (1 - diffusion_rate) + (sum_r / count) * diffusion_rate
                tv.px.px.rgba[x, y][1] = tv.px.px.rgba[x, y][1] * (1 - diffusion_rate) + (sum_g / count) * diffusion_rate
                tv.px.px.rgba[x, y][2] = tv.px.px.rgba[x, y][2] * (1 - diffusion_rate) + (sum_b / count) * diffusion_rate"""
    
    def _generate_decay_kernel(self) -> str:
        """Generate pheromone decay/evaporation kernel."""
        return """@ti.kernel
def decay_pheromones():
    \"\"\"Decay/evaporate pheromone trails.\"\"\"
    evaporation_rate = 0.99  # Could be from global state
    
    for x in range(tv.x):
        for y in range(tv.y):
            tv.px.px.rgba[x, y][0] *= evaporation_rate
            tv.px.px.rgba[x, y][1] *= evaporation_rate
            tv.px.px.rgba[x, y][2] *= evaporation_rate"""
    
    def _generate_deposition_kernel(self) -> str:
        """Generate trail deposition kernel."""
        return """@ti.kernel
def deposit_trails():
    \"\"\"Deposit pheromone trails at particle positions.\"\"\"
    deposit_amount = 0.5  # Could be from global state
    
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            x = ti.cast(tv.p.field[i].pos[0], ti.i32) % tv.x
            y = ti.cast(tv.p.field[i].pos[1], ti.i32) % tv.y
            
            if x >= 0 and x < tv.x and y >= 0 and y < tv.y:
                species = tv.p.field[i].species
                
                # Deposit based on species (different channels)
                if species == 0:
                    tv.px.px.rgba[x, y][0] += deposit_amount
                    tv.px.px.rgba[x, y][0] = ti.min(1.0, tv.px.px.rgba[x, y][0])
                elif species == 1:
                    tv.px.px.rgba[x, y][1] += deposit_amount
                    tv.px.px.rgba[x, y][1] = ti.min(1.0, tv.px.px.rgba[x, y][1])
                elif species == 2:
                    tv.px.px.rgba[x, y][2] += deposit_amount
                    tv.px.px.rgba[x, y][2] = ti.min(1.0, tv.px.px.rgba[x, y][2])"""