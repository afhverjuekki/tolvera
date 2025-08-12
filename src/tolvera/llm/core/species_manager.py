import re
import logging
from typing import Dict, List, Optional, Tuple
from .species_analyzer import SpeciesInfo, SpeciesAnalyzer

logger = logging.getLogger(__name__)


class SpeciesManager:
    
    def __init__(self, tolvera_instance):
        self.tv = tolvera_instance
        self.analyzer = SpeciesAnalyzer()
        self.current_species_info = None
        
    def analyze_and_configure(self, description: str) -> SpeciesInfo:
        self.current_species_info = self.analyzer.analyze(description)
        logger.info(f"Configured {self.current_species_info.total_count} species from description")
        return self.current_species_info
    
    def get_initialization_code(
        self, 
        species_info: SpeciesInfo,
        init_type: str = "random",
        grid_size: Optional[int] = None,
        species_config=None,
        speed_spec=None
    ) -> str:
        if init_type == "grid" and grid_size:
            return self._generate_grid_init(species_info, grid_size, species_config, speed_spec)
        elif init_type == "clustered":
            return self._generate_clustered_init(species_info, species_config, speed_spec)
        else:
            return self._generate_random_init(species_info, species_config, speed_spec)
    
    def get_species_colors(self, species_info: SpeciesInfo, species_config=None) -> Dict[int, List[float]]:
        colors = {}
        
        if species_config and species_config.colors:
            colors.update(species_config.colors)
        
        for i in range(species_info.total_count):
            if i not in colors:
                colors[i] = self.analyzer.get_color_for_species(i, species_info)
        
        return colors
    
    def _get_speed_value(self, speed_spec) -> float:
        """Get speed value from speed specification."""
        if not speed_spec:
            return 100.0  # Default medium speed
        
        if speed_spec.value is not None:
            return speed_spec.value
        
        # Map magnitude to values
        magnitude_map = {
            "slow": 50.0,
            "medium": 100.0, 
            "fast": 200.0,
            "very_fast": 300.0
        }
        
        return magnitude_map.get(speed_spec.magnitude, 100.0)
    
    def _generate_random_init(self, species_info: SpeciesInfo, species_config=None, speed_spec=None) -> str:
        colors = self.get_species_colors(species_info, species_config)
        
        if species_info.total_count > 1:
            species_map_code = f"""# Species mapping
species_map = ti.field(dtype=ti.i32, shape={species_info.total_count})
"""
            for i in range(species_info.total_count):
                species_map_code += f"species_map[{i}] = {i}\n"
            species_map_code += "\n"
        else:
            species_map_code = ""
        
        # Handle velocity initialization based on speed_spec
        if speed_spec and speed_spec.uniform:
            # All particles same speed, random directions
            speed_val = self._get_speed_value(speed_spec)
            velocity_init = f"""        # Uniform speed with random directions
        angle = ti.random() * 2 * 3.14159
        tv.p.field[i].vel = ti.Vector([
            ti.cos(angle) * {speed_val},
            ti.sin(angle) * {speed_val}
        ])"""
        else:
            # Default random velocities
            velocity_init = f"""        tv.p.field[i].vel = ti.Vector([
            (ti.random() - 0.5) * 100.0,
            (ti.random() - 0.5) * 100.0
        ])"""
        
        init_code = f"""{species_map_code}@ti.kernel
def init_particles():
    for i in range(tv.pn):
        tv.p.field[i].active = 1.0
        tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
{velocity_init}
        tv.p.field[i].size = 5.0
        tv.p.field[i].mass = 1.0
        tv.p.field[i].speed = 20.0  
        
        # Assign species"""
        
        if species_info.total_count == 1:
            init_code += f"\n        tv.p.field[i].species = 0"
        else:
            init_code += f"\n        tv.p.field[i].species = species_map[i % {species_info.total_count}]"
        
        init_code += "\n\ninit_particles()\n"
        
        if species_info.species_names:
            init_code += "\n# Species configuration:\n"
            for sid, name in species_info.species_names.items():
                init_code += f"# Species {sid}: {name}\n"
        
        init_code += "\n# Set species colors\n"
        for sid, color in colors.items():
            init_code += f"tv.s.species.field[{sid}].rgba = {color}\n"
        
        return init_code
    
    def _generate_grid_init(self, species_info: SpeciesInfo, grid_size: int, species_config=None, speed_spec=None) -> str:
        colors = self.get_species_colors(species_info, species_config)
        
        init_code = f"""@ti.kernel
def init_particles_grid():
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
                tv.p.field[particle_idx].speed = 20.0 
                
                # Assign species"""
        
        if species_info.total_count == 1:
            init_code += f"\n                tv.p.field[particle_idx].species = 0"
        else:
            # Pattern assignment for grid
            init_code += f"\n                # Assign species in pattern"
            init_code += f"\n                tv.p.field[particle_idx].species = particle_idx % {species_info.total_count}"
        
        init_code += """
                
                particle_idx += 1
    
    # Deactivate unused particles
    for i in range(particle_idx, tv.pn):
        tv.p.field[i].active = 0.0

init_particles_grid()
"""
        
        # Set colors
        init_code += "\n# Set species colors\n"
        for sid, color in colors.items():
            init_code += f"tv.s.species.field[{sid}].rgba = {color}\n"
        
        return init_code
    
    def _generate_clustered_init(self, species_info: SpeciesInfo, species_config=None, speed_spec=None) -> str:
        colors = self.get_species_colors(species_info, species_config)
        
        init_code = f"""@ti.kernel
def init_particles_clustered():
    particles_per_species = tv.pn // {species_info.total_count}
    remaining = tv.pn % {species_info.total_count}
    
    particle_idx = 0"""
        
        # Create clusters for each species
        for sid in range(species_info.total_count):
            cluster_angle = sid * 2.0 * 3.14159 / species_info.total_count
            init_code += f"""
    
    # Species {sid} cluster
    cluster_center_{sid} = ti.Vector([
        tv.x * 0.5 + ti.cos({cluster_angle}) * tv.x * 0.3,
        tv.y * 0.5 + ti.sin({cluster_angle}) * tv.y * 0.3
    ])
    
    for i in range(particles_per_species + (1 if {sid} < remaining else 0)):
        if particle_idx < tv.pn:
            # Random position within cluster
            angle = ti.random() * 2.0 * 3.14159
            radius = ti.random() * min(tv.x, tv.y) * 0.15
            
            tv.p.field[particle_idx].active = 1.0
            tv.p.field[particle_idx].pos = cluster_center_{sid} + ti.Vector([
                ti.cos(angle) * radius,
                ti.sin(angle) * radius
            ])"""
            
            # Add velocity initialization based on speed_spec
            if speed_spec and speed_spec.uniform:
                speed_val = self._get_speed_value(speed_spec)
                init_code += f"""
            # Uniform speed with random directions
            vel_angle = ti.random() * 2 * 3.14159
            tv.p.field[particle_idx].vel = ti.Vector([
                ti.cos(vel_angle) * {speed_val},
                ti.sin(vel_angle) * {speed_val}
            ])"""
            else:
                init_code += f"""
            tv.p.field[particle_idx].vel = ti.Vector([
                (ti.random() - 0.5) * 150.0,
                (ti.random() - 0.5) * 150.0
            ])"""
            
            init_code += f"""
            tv.p.field[particle_idx].size = 5.0
            tv.p.field[particle_idx].mass = 1.0
            tv.p.field[particle_idx].speed = 20.0 
            tv.p.field[particle_idx].species = {sid}
            
            particle_idx += 1"""
        
        init_code += "\n\ninit_particles_clustered()\n"
        
        # Add species names as comments
        if species_info.species_names:
            init_code += "\n# Species configuration:\n"
            for sid, name in species_info.species_names.items():
                init_code += f"# Species {sid}: {name} (clustered)\n"
        
        # Set colors
        init_code += "\n# Set species colors\n"
        for sid, color in colors.items():
            init_code += f"tv.s.species.field[{sid}].rgba = {color}\n"
        
        return init_code
    
    def get_species_context_for_prompts(self, species_info: SpeciesInfo) -> str:
        context = f"## Species Configuration\n\n"
        context += f"Total species: {species_info.total_count}\n"
        context += f"Species IDs in use: {list(range(species_info.total_count))}\n\n"
        
        if species_info.species_names:
            context += "### Species Names and Roles:\n"
            for sid, name in species_info.species_names.items():
                context += f"- Species {sid}: {name}\n"
            context += "\n"
        
        if species_info.species_behaviors:
            context += "### Species-Specific Behaviors:\n"
            for sid, behaviors in species_info.species_behaviors.items():
                name = species_info.species_names.get(sid, f"Species {sid}")
                context += f"- {name}:\n"
                for behavior in behaviors:
                    context += f"  - {behavior}\n"
            context += "\n"
        
        if species_info.interaction_pairs:
            context += "### Species Interactions:\n"
            for s1, s2 in species_info.interaction_pairs:
                n1 = species_info.species_names.get(s1, f"Species {s1}")
                n2 = species_info.species_names.get(s2, f"Species {s2}")
                context += f"- {n1} interacts with {n2}\n"
            context += "\n"
        
        # Add code generation hints
        context += "### Code Generation Hints:\n"
        context += "- Use `species` parameter to check particle species\n"
        context += "- Use conditional logic for species-specific behaviors\n"
        if species_info.total_count > 1:
            context += "- Consider different forces/behaviors for different species\n"
            context += f"- Species IDs range from 0 to {species_info.total_count - 1}\n"
        
        return context
    
    def should_use_grid_init(self, description: str) -> Tuple[bool, Optional[int]]:
        desc_lower = description.lower()
        
        # Grid indicators - ONLY for cellular automata, NOT ecosystems
        grid_patterns = [
            'cellular automaton', 'game of life', 'conway',
            'grid', 'cells', 'lattice'
        ]
        
        # Exclude ecosystem patterns from grid init
        ecosystem_patterns = [
            'ecosystem', 'predator', 'prey', 'fish', 'school',
            'hunt', 'chase', 'flee', 'scavenger', 'food chain'
        ]
        
        # Check if it's an ecosystem pattern - don't use grid
        if any(pattern in desc_lower for pattern in ecosystem_patterns):
            return False, None
        
        # Check for explicit grid patterns
        if any(pattern in desc_lower for pattern in grid_patterns):
            # Try to extract grid size
            size_match = re.search(r'(\d+)x\1', desc_lower)
            if size_match:
                return True, int(size_match.group(1))
            else:
                # Default grid size
                return True, 50
        
        return False, None