import logging
import random
from typing import List, Dict, Tuple, Optional
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
    
    def get_species_initialization_code(self, species_ids: List[int], species_config: Optional[Dict] = None) -> str:
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
            species_assignment = f"tv.p.field[i].species = {species_ids[0]}"
        else:
            species_assignment_parts = []
            for idx, species_id in enumerate(species_ids):
                if idx == 0:
                    species_assignment_parts.append(f"        if i % {num_species} == {idx}:")
                else:
                    species_assignment_parts.append(f"        elif i % {num_species} == {idx}:")
                species_assignment_parts.append(f"            tv.p.field[i].species = {species_id}")
            species_assignment = "\n".join(species_assignment_parts)
        
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
{species_assignment if num_species == 1 else species_assignment}

init_particles()

'''
        
        for species_id, color in colors_for_species.items():
            init_code += f'tv.s.species.field[{species_id}].rgba = {color}\n'
        
        return init_code
    
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
                    f"                total_force += expert_{expert['name']}(pos, vel, mass, species) * {expert['weight']:.2f}"
                )
            else:
                single_calls.append(
                    f"            total_force += expert_{expert['name']}(pos, vel, mass, species) * {expert['weight']:.2f}"
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