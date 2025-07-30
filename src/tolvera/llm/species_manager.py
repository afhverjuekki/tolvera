import logging
import random
from typing import List, Dict, Tuple, Optional
import taichi as ti
from .boundary_manager import BoundaryMode, BoundaryManager

logger = logging.getLogger(__name__)


class SpeciesManager:
    
    def __init__(self, tolvera_instance):
        self.tv = tolvera_instance
        self.current_species_count = self.tv.sn
        self.boundary_manager = BoundaryManager()
        logger.info(f"Initialized SpeciesManager with {self.current_species_count} species")
    
    def analyze_species_requirements(self, behaviors: List[Dict]) -> Tuple[List[int], Dict]:
        all_species_mentioned = set()
        interaction_pairs = []
        requires_multiple = False
        has_generic_behavior = False  # Track if any behavior applies to all species
        
        for behavior in behaviors:
            species_info = behavior.get('species_info', {})
            
            if species_info.get('species_mentioned'):
                all_species_mentioned.update(species_info['species_mentioned'])
            
            # Check if behavior applies to all species
            if species_info.get('requires_all_species', False):
                has_generic_behavior = True
            
            # Collect interaction pairs
            if species_info.get('interaction_pairs'):
                interaction_pairs.extend(species_info['interaction_pairs'])
                requires_multiple = True
            
            if behavior.get('is_interaction', False):
                requires_multiple = True
        
        if all_species_mentioned:
            species_ids = sorted(list(all_species_mentioned))
        elif requires_multiple and not has_generic_behavior:
            species_ids = [0, 1]
        else:
            species_ids = [0]
        
        analysis = {
            'species_ids': species_ids,
            'species_count': len(species_ids),
            'species_mentioned': sorted(list(all_species_mentioned)),
            'interaction_pairs': interaction_pairs,
            'requires_multiple': requires_multiple,
            'has_generic_behavior': has_generic_behavior,
            'species_mapping': {sid: idx for idx, sid in enumerate(species_ids)}
        }
        
        logger.info(f"Species analysis complete: {analysis}")
        return species_ids, analysis
    
    def get_species_initialization_code(self, species_ids: List[int], species_config: Optional[Dict] = None, grid_size: Optional[int] = None) -> str:
        default_colors = [
            [1.0, 0.3, 0.3, 1.0],  # Red
            [0.3, 0.3, 1.0, 1.0],  # Blue
            [0.3, 1.0, 0.3, 1.0],  # Green
            [1.0, 1.0, 0.3, 1.0],  # Yellow
            [1.0, 0.3, 1.0, 1.0],  # Magenta
        ]
        
        colors_for_species = {}
        for idx, species_id in enumerate(species_ids):
            if idx < len(default_colors):
                colors_for_species[species_id] = default_colors[idx]
            else:
                hue_offset = (idx - len(default_colors)) / max(1, (len(species_ids) - len(default_colors)))
                colors_for_species[species_id] = [
                    0.5 + 0.5 * random.random(),
                    0.5 + 0.5 * random.random(),
                    0.5 + 0.5 * random.random(),
                    1.0
                ]
        
        num_species = len(species_ids)
        
        if num_species == 1:
            species_assignment = f"                tv.p.field[particle_idx].species = {species_ids[0]}"
            species_assignment_regular = f"        tv.p.field[i].species = {species_ids[0]}"
        else:
            # Grid assignment (for particle_idx)
            species_assignment_parts = []
            for idx, species_id in enumerate(species_ids):
                if idx == 0:
                    species_assignment_parts.append(f"                if particle_idx % {num_species} == {idx}:")
                else:
                    species_assignment_parts.append(f"                elif particle_idx % {num_species} == {idx}:")
                species_assignment_parts.append(f"                    tv.p.field[particle_idx].species = {species_id}")
            species_assignment = "\n".join(species_assignment_parts)
            
            # Regular assignment (for i)
            species_assignment_regular_parts = []
            for idx, species_id in enumerate(species_ids):
                if idx == 0:
                    species_assignment_regular_parts.append(f"        if i % {num_species} == {idx}:")
                else:
                    species_assignment_regular_parts.append(f"        elif i % {num_species} == {idx}:")
                species_assignment_regular_parts.append(f"            tv.p.field[i].species = {species_id}")
            species_assignment_regular = "\n".join(species_assignment_regular_parts)
        
        if grid_size:
            # Grid initialization for cellular automata and similar patterns
            init_code = f'''
@ti.kernel
def init_particles_grid():
    # Initialize particles in a {grid_size}x{grid_size} grid
    # Using only species IDs: {species_ids}
    grid_spacing_x = tv.x / {grid_size}
    grid_spacing_y = tv.y / {grid_size}
    
    particle_idx = 0
    for row in range({grid_size}):
        for col in range({grid_size}):
            if particle_idx < tv.pn:
                # Position at grid center
                x = (col + 0.5) * grid_spacing_x
                y = (row + 0.5) * grid_spacing_y
                
                tv.p.field[particle_idx].active = 1.0
                tv.p.field[particle_idx].pos = ti.Vector([x, y])
                tv.p.field[particle_idx].vel = ti.Vector([0.0, 0.0])
                tv.p.field[particle_idx].size = min(grid_spacing_x, grid_spacing_y) * 0.8
                tv.p.field[particle_idx].mass = 1.0
                
                # Assign species
{species_assignment}
                
                # Store grid coordinates in custom states if available
                if hasattr(tv.s, 'llm_particle'):
                    if hasattr(tv.s.llm_particle.field[particle_idx], 'grid_x'):
                        tv.s.llm_particle.field[particle_idx].grid_x = col
                    if hasattr(tv.s.llm_particle.field[particle_idx], 'grid_y'):
                        tv.s.llm_particle.field[particle_idx].grid_y = row
                
                particle_idx += 1
    
    # Deactivate remaining particles if grid doesn't use all
    for i in range(particle_idx, tv.pn):
        tv.p.field[i].active = 0.0

init_particles_grid()

'''
        else:
            # Random initialization for non-grid patterns
            init_code = f'''
@ti.kernel
def init_particles():
    # Initialize particles with species IDs: {species_ids}
    for i in range(tv.pn):
        tv.p.field[i].active = 1.0
        tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
        tv.p.field[i].vel = ti.Vector([0.0, 0.0])
        tv.p.field[i].size = 5.0
        tv.p.field[i].mass = 1.0
        # Assign particles to only the specified species IDs
{species_assignment_regular}

init_particles()

'''
        
        for species_id, color in colors_for_species.items():
            init_code += f'tv.s.species.field[{species_id}].rgba = {color}\n'
        
        return init_code
    
    def detect_grid_requirements(self, behaviors: List[Dict], state_spec: Dict) -> Optional[int]:
        """Detect if behaviors require grid initialization and return suggested grid size."""
        # Check behaviors for grid patterns
        grid_indicators = [
            'cellular automaton', 'game of life', 'conway',
            'grid', 'cells live or die', 'cell state'
        ]
        
        for behavior in behaviors:
            desc = behavior.get('description', '').lower()
            if any(indicator in desc for indicator in grid_indicators):
                # Check state spec for grid size hints
                for category in ['particle_states', 'global_states']:
                    for state_name, state_info in state_spec.get(category, {}).items():
                        if state_info.get('is_grid_coordinate') or 'grid_size' in state_info:
                            return state_info.get('grid_size', 50)
                
                # Default grid size for cellular automata
                return 50
        
        # Check if state spec has grid coordinates
        for category in ['particle_states', 'global_states']:
            for state_name, state_info in state_spec.get(category, {}).items():
                if state_name in ['grid_x', 'grid_y'] or state_info.get('is_grid_coordinate'):
                    return state_info.get('grid_size', 50)
        
        return None
    
    def generate_species_aware_kernel(self, expert_info: List[Dict], species_ids: List[int], boundary_mode: BoundaryMode = BoundaryMode.NONE) -> str:
        single_experts = [e for e in expert_info if not e.get('is_interaction', False)]
        interaction_experts = [e for e in expert_info if e.get('is_interaction', False)]
        
        single_calls = []
        for expert in single_experts:
            species_info = expert.get('species_info', {})
            mentioned = species_info.get('species_mentioned', [])
            
            if mentioned:
                conditions = " or ".join([f"species == {s}" for s in mentioned])
                single_calls.append(
                    f"            if {conditions}:\n"
                    f"                total_force += expert_{expert['name']}(pos, vel, mass, species, i) * {expert['weight']:.2f}"
                )
            else:
                single_calls.append(
                    f"            total_force += expert_{expert['name']}(pos, vel, mass, species, i) * {expert['weight']:.2f}"
                )
        
        single_calls_str = "\n".join(single_calls) if single_calls else "            # No single-particle experts"
        
        interaction_calls = []
        for expert in interaction_experts:
            species_info = expert.get('species_info', {})
            pairs = species_info.get('interaction_pairs', [])
            
            if pairs:
                conditions = []
                for pair in pairs:
                    if pair[0] in species_ids and pair[1] in species_ids:
                        conditions.append(f"(p1.species == {pair[0]} and p2.species == {pair[1]})")
                        if pair[0] != pair[1]:
                            conditions.append(f"(p1.species == {pair[1]} and p2.species == {pair[0]})")
                
                if conditions: 
                    condition_str = " or ".join(conditions)
                    interaction_calls.append(
                        f"                    if {condition_str}:\n"
                        f"                        total_force += expert_{expert['name']}(p1, p2) * {expert['weight']:.2f}"
                    )
            else:
                species_check = " and ".join([f"(p1.species == {s} or p2.species == {s})" for s in species_ids[:2]]) if len(species_ids) <= 2 else ""
                if species_check:
                    interaction_calls.append(
                        f"                    # Apply to all pairs of valid species\n"
                        f"                    total_force += expert_{expert['name']}(p1, p2) * {expert['weight']:.2f}"
                    )
                else:
                    interaction_calls.append(
                        f"                    total_force += expert_{expert['name']}(p1, p2) * {expert['weight']:.2f}"
                    )
        
        interaction_calls_str = "\n".join(interaction_calls) if interaction_calls else "                    # No interaction experts"
        
        kernel_code = f'''@ti.kernel
def apply_all_experts():
    """Main kernel that integrates all expert forces including interactions."""
    dt = 0.016
    
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            # Get current particle state
            pos = tv.p.field[i].pos
            vel = tv.p.field[i].vel
            mass = tv.p.field[i].mass
            species = tv.p.field[i].species
            
            # Initialize total force
            total_force = ti.math.vec2(0.0, 0.0)
            
            # Apply single-particle experts
{single_calls_str}
            
'''
        
        # Add interaction loop only if we have interaction experts
        if interaction_experts:
            kernel_code += f'''            # Apply interaction experts
            for j in range(tv.pn):
                if i != j and tv.p.field[j].active > 0:
                    p1 = tv.p.field[i]
                    p2 = tv.p.field[j]
                    
{interaction_calls_str}
            
'''
        
        # Get boundary handling code based on mode
        boundary_code = self.boundary_manager.get_boundary_code(boundary_mode, use_new_pos=True)
        
        kernel_code += '''            # Update velocity with damping
            tv.p.field[i].vel = tv.p.field[i].vel * 0.99 + total_force * dt
            
            # Update position
            new_pos = tv.p.field[i].pos + tv.p.field[i].vel * dt
            
'''
        
        if boundary_code:
            kernel_code += boundary_code + '\n'
        else:
            # No boundary handling - just update position
            kernel_code += '            tv.p.field[i].pos = new_pos\n'
        
        return kernel_code
    
    def initialize_particles_grid(self, species_ids: List[int], grid_size: int = 50):
        """Initialize particles in a grid pattern for cellular automata."""
        default_colors = [
            [1.0, 0.3, 0.3, 1.0],  # Red
            [0.3, 0.3, 1.0, 1.0],  # Blue
            [0.3, 1.0, 0.3, 1.0],  # Green
            [1.0, 1.0, 0.3, 1.0],  # Yellow
            [1.0, 0.3, 1.0, 1.0],  # Magenta
        ]
        
        # Set species colors
        for idx, species_id in enumerate(species_ids):
            if idx < len(default_colors):
                color = default_colors[idx]
            else:
                color = [
                    0.5 + 0.5 * random.random(),
                    0.5 + 0.5 * random.random(),
                    0.5 + 0.5 * random.random(),
                    1.0
                ]
            self.tv.s.species.field[species_id].rgba = color
        
        # Define kernel inline to avoid exec issues
        @ti.kernel
        def init_grid_kernel(tv: ti.template(), grid_size: ti.i32, species_ids: ti.types.ndarray()):
            num_species = species_ids.shape[0]
            grid_spacing_x = tv.x / grid_size
            grid_spacing_y = tv.y / grid_size
            
            particle_idx = 0
            for row in range(grid_size):
                for col in range(grid_size):
                    if particle_idx < tv.pn:
                        # Position at grid center
                        x = (col + 0.5) * grid_spacing_x
                        y = (row + 0.5) * grid_spacing_y
                        
                        tv.p.field[particle_idx].active = 1.0
                        tv.p.field[particle_idx].pos = ti.Vector([x, y])
                        tv.p.field[particle_idx].vel = ti.Vector([0.0, 0.0])
                        tv.p.field[particle_idx].size = min(grid_spacing_x, grid_spacing_y) * 0.8
                        tv.p.field[particle_idx].mass = 1.0
                        
                        # Assign species cyclically
                        species_idx = particle_idx % num_species
                        tv.p.field[particle_idx].species = species_ids[species_idx]
                        
                        particle_idx += 1
            
            # Deactivate remaining particles
            for i in range(particle_idx, tv.pn):
                tv.p.field[i].active = 0.0
        
        # Convert species_ids to numpy array for Taichi
        import numpy as np
        species_array = np.array(species_ids, dtype=np.int32)
        
        # Call the kernel
        init_grid_kernel(self.tv, grid_size, species_array)
        logger.info(f"Initialized {grid_size}x{grid_size} grid with species {species_ids}")
    
    def initialize_particles_random(self, species_ids: List[int]):
        """Initialize particles randomly for non-grid patterns."""
        default_colors = [
            [1.0, 0.3, 0.3, 1.0],  # Red
            [0.3, 0.3, 1.0, 1.0],  # Blue
            [0.3, 1.0, 0.3, 1.0],  # Green
            [1.0, 1.0, 0.3, 1.0],  # Yellow
            [1.0, 0.3, 1.0, 1.0],  # Magenta
        ]
        
        # Set species colors
        for idx, species_id in enumerate(species_ids):
            if idx < len(default_colors):
                color = default_colors[idx]
            else:
                color = [
                    0.5 + 0.5 * random.random(),
                    0.5 + 0.5 * random.random(),
                    0.5 + 0.5 * random.random(),
                    1.0
                ]
            self.tv.s.species.field[species_id].rgba = color
        
        @ti.kernel
        def init_random_kernel(tv: ti.template(), species_ids: ti.types.ndarray()):
            num_species = species_ids.shape[0]
            for i in range(tv.pn):
                tv.p.field[i].active = 1.0
                tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
                tv.p.field[i].vel = ti.Vector([0.0, 0.0])
                tv.p.field[i].size = 5.0
                tv.p.field[i].mass = 1.0
                
                # Assign species cyclically
                species_idx = i % num_species
                tv.p.field[i].species = species_ids[species_idx]
        
        # Convert species_ids to numpy array for Taichi
        import numpy as np
        species_array = np.array(species_ids, dtype=np.int32)
        
        # Call the kernel
        init_random_kernel(self.tv, species_array)
        logger.info(f"Initialized particles randomly with species {species_ids}")