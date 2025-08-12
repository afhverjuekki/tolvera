import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.models import BehaviorSynthesisResponse


class SketchGenerator:
    """Generates complete Tölvera sketch files from synthesized components."""
    
    def __init__(self):
        self.template = '''"""
Auto-generated Tölvera sketch: {description}
Generated: {timestamp}
"""

import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    """Main function for Tölvera sketch."""
    # === Configuration ===
    # CRITICAL: Set ALL kwargs BEFORE creating Tölvera instance
    {config_code}
    
    # Create Tölvera instance with properly configured kwargs
    tv = Tolvera(**kwargs)
    
    # === Particle Initialization ===
{init_code}
    
    # === State Initialization ===
{state_code}
    
    # === Particle Force Experts ===
    # These functions calculate forces acting on individual particles
{expert_code}
    
    # === Integration Kernel ===
    # Applies all particle forces and updates physics
{kernel_code}
    
    # === Temporal Updates ===
    # Legacy temporal update kernels (if any)
{temporal_code}
    
    # === Utility Functions ===
    # State updates, temporal dynamics, and helper functions that don't act on particles
{utility_code}
    
    # === Utility Kernel ===
    # Executes utility functions for state management
{utility_kernel}
    
    # === Drawing Functions ===
{drawing_code}
    
    # === Drawing Kernel ===
{drawing_kernel}
    
    # === Respawn Functions ===
{respawn_code}
    
    @tv.render
    def _():
        tv.px.diffuse(0.99)
        
        # Pre-particle drawing
        {pre_draw_calls}
        
        # Apply behaviors and physics
        {update_calls}
        
        # Post-particle drawing
        {post_draw_calls}
        
        # Render particles
        {render_particles_call}
        
        return tv.px

if __name__ == "__main__":
    run(main)
'''
        
        self.pure_drawing_template = '''"""
Auto-generated Tölvera pure drawing sketch: {description}
Generated: {timestamp}
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """Main function for pure drawing sketch."""
    # === Configuration ===
    # Pure drawing mode - minimal setup
    kwargs['width'] = kwargs.get('width', 1920)
    kwargs['height'] = kwargs.get('height', 1080)
    kwargs['pn'] = 1  # Minimal particles (not used)
    kwargs['sn'] = 1  # Minimal species (not used)
    
    # Create Tölvera instance
    tv = Tolvera(**kwargs)
    
    # === Drawing Functions ===
{drawing_code}
    
    @tv.render
    def _():
        # Clear screen
        tv.px.clear()
        
        # Execute drawing
        draw()
        
        return tv.px

if __name__ == "__main__":
    run(main)
'''
    
    def generate(
        self,
        description: str,
        experts: List[str],
        kernel: str,
        init_code: str,
        state_code: str = "",
        temporal_code: str = "",
        config_code: str = "",
        utility_code: str = "",
        utility_kernel: str = "",
        drawing_code: str = "",
        drawing_kernel: str = "",
        respawn_code: str = "",
        pre_draw_calls: List[str] = None,
        post_draw_calls: List[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        has_non_visual_experts: bool = True
    ) -> str:
        """
        Generate complete sketch file.
        
        Args:
            description: Behavior description
            experts: List of expert function code strings
            kernel: Integration kernel code
            init_code: Particle initialization code
            state_code: State initialization code
            temporal_code: Temporal update code
            config_code: Additional configuration code
            utility_code: Utility expert functions (temporal updates, state updates, etc.)
            utility_kernel: Kernel that calls utility experts
            drawing_code: Drawing behavior functions
            drawing_kernel: Drawing kernel that calls visual experts
            respawn_code: Respawn functions for food/resources
            pre_draw_calls: Drawing calls before particles
            post_draw_calls: Drawing calls after particles
            metadata: Optional metadata to include
            has_non_visual_experts: Whether there are non-visual experts (single/interaction)
            
        Returns:
            Complete sketch code as string
        """
        # Build update calls - CORRECT ORDER IS CRITICAL
        update_calls = []
        
        # 1. Always apply utility experts first (handles temporal updates, state updates)
        if utility_kernel and "update_utilities" in utility_kernel:
            update_calls.append("update_utilities()  # Execute utility functions")
        
        # Only include particle physics if we have force experts
        if has_non_visual_experts:
            # 2. Apply expert behaviors to calculate forces
            update_calls.append("apply_all_experts()")
            
            # 3. Update physics (positions, velocities, boundaries)
            update_calls.append("tv.p()")
        
        # 4. Apply drawing behaviors if present
        if drawing_kernel and "draw" in drawing_kernel:
            update_calls.append("draw()  # Execute visual behaviors")
        
        # 5. Check for respawn (e.g., food particles)
        if respawn_code and "respawn_food" in respawn_code:
            update_calls.append("respawn_food()")
        
        # Format drawing calls
        if pre_draw_calls is None:
            pre_draw_calls = ["# No pre-draw effects"]
        if post_draw_calls is None:
            post_draw_calls = ["# No post-draw effects"]
        
        # Format components - indent code sections properly
        # Enhanced config code to always set kwargs properly
        if not config_code or config_code == "# No additional configuration":
            # Always generate proper kwargs configuration
            config_code = """# Configure Tölvera parameters
    # Default configuration - modify kwargs as needed
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080
    if 'pn' not in kwargs:
        kwargs['pn'] = 1000
    if 'sn' not in kwargs:
        kwargs['sn'] = 1  # Default to 1 species"""
        
        # Determine whether to render particles
        if has_non_visual_experts:
            render_particles_call = "tv.px.particles(tv.p, tv.s.species())"
        else:
            render_particles_call = "# No particle rendering (only visual behaviors)"
        
        return self.template.format(
            description=description,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_code=config_code.strip(),
            init_code=self._indent(init_code.strip(), 4),
            state_code=self._indent(state_code.strip() if state_code else "# No custom states needed", 4),
            expert_code=self._indent("\n\n".join(experts), 4),
            kernel_code=self._indent(kernel, 4),
            temporal_code=self._indent(temporal_code if temporal_code else "# No temporal updates needed", 4),
            utility_code=self._indent(utility_code if utility_code else "# No utility functions", 4),
            utility_kernel=self._indent(utility_kernel if utility_kernel else "# No utility kernel", 4),
            drawing_code=self._indent(drawing_code if drawing_code else "# No drawing functions", 4),
            drawing_kernel=self._indent(drawing_kernel if drawing_kernel else "# No drawing kernel", 4),
            respawn_code=self._indent(respawn_code if respawn_code else "# No respawn functions", 4),
            pre_draw_calls="\n        ".join(pre_draw_calls),
            post_draw_calls="\n        ".join(post_draw_calls),
            update_calls="\n        ".join(update_calls),
            render_particles_call=render_particles_call
        )
    
    def _indent(self, code: str, spaces: int = 4) -> str:
        """Indent code block by specified spaces."""
        if not code:
            return code
        
        indent = " " * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)
    
    def generate_from_response(
        self,
        response: 'BehaviorSynthesisResponse',
        description: str,
        tv_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate sketch from a BehaviorSynthesisResponse object.
        
        Args:
            response: The synthesis response object
            description: Original behavior description
            tv_config: Tölvera configuration parameters
            
        Returns:
            Complete sketch code
        """
        # Get expert code
        expert_code = [expert.to_code() for expert in response.experts]
        
        # Get kernel code
        kernel_code = response.integration_kernel.to_code()
        
        # Get initialization code
        init_code = response.get_init_code()
        
        # Get state code
        if hasattr(response, 'get_state_code'):
            state_code = response.get_state_code()
        else:
            state_code = self._generate_state_code(response.states_needed)
        
        # Get temporal code
        temporal_code = ""
        if response.temporal_update:
            temporal_code = response.temporal_update.to_code()
        
        # Build config code - ALWAYS set kwargs properly
        config_lines = []
        
        # Get species count from response
        species_count = 1
        if hasattr(response, 'species_config') and response.species_config:
            if hasattr(response.species_config, 'species_ids'):
                species_count = len(response.species_config.species_ids)
            elif hasattr(response.species_config, 'total_count'):
                species_count = response.species_config.total_count
        
        # Generate proper kwargs configuration - using correct Tolvera parameter names
        config_lines.append("# === CRITICAL: Set ALL kwargs BEFORE creating Tölvera instance ===")
        config_lines.append(f"kwargs['species'] = {species_count}  # Use detected species count")
        
        # Set particle count - use 'particles' which is the correct parameter name
        default_particles = tv_config.get('particles', 1000) if tv_config else 1000
        config_lines.append(f"kwargs['particles'] = kwargs.get('particles', {default_particles})  # Particle count")
        
        config_lines.append("kwargs['width'] = kwargs.get('width', 1920)")
        config_lines.append("kwargs['height'] = kwargs.get('height', 1080)")
        
        config_code = "\n    ".join(config_lines)
        
        return self.generate(
            description=description,
            experts=expert_code,
            kernel=kernel_code,
            init_code=init_code,
            state_code=state_code,
            temporal_code=temporal_code,
            config_code=config_code
        )
    
    def _generate_state_code(self, states_needed) -> str:
        if not states_needed:
            return "# No custom states defined"
        
        # Handle both list and dict formats
        if isinstance(states_needed, list):
            # Convert list of StateDefinition objects to dict format
            states_dict = {'global': {}, 'particle': {}, 'species': {}}
            for state_def in states_needed:
                category = state_def.category if hasattr(state_def, 'category') else 'particle'
                name = state_def.name if hasattr(state_def, 'name') else 'unknown'
                # Skip built-in particle properties
                if category == 'particle' and name.lower() in {'pos', 'vel', 'mass', 'size', 'speed', 'species', 'active', 'ppos', 'pvel'}:
                    continue
                states_dict[category][name] = state_def
            states_needed = states_dict
        
        code_lines = []
        
        # Global states
        if 'global' in states_needed and states_needed['global']:
            code_lines.append("# Global states")
            code_lines.append("if 'llm_global' not in tv.s:")
            code_lines.append("    global_states = {")
            
            for name, state_def in states_needed['global'].items():
                type_str = state_def.type if hasattr(state_def, 'type') else state_def.get('type', 'ti.f32')
                min_val = state_def.min if hasattr(state_def, 'min') else state_def.get('min', 0.0)
                max_val = state_def.max if hasattr(state_def, 'max') else state_def.get('max', 1.0)
                code_lines.append(f"        '{name}': ({type_str}, {min_val}, {max_val}),")
            
            code_lines.append("    }")
            code_lines.append("    tv.s.set('llm_global', {")
            code_lines.append("        'state': global_states,")
            code_lines.append("        'shape': 1,")
            code_lines.append("        'osc': ('get', 'set'),")
            code_lines.append("        'randomise': False")
            code_lines.append("    })")
            code_lines.append("")
        
        # Temporal states are now part of global states (removed llm_temporal category)
        
        # Particle states
        if 'particle' in states_needed and states_needed['particle']:
            # Filter out built-in properties
            BUILTIN_PARTICLE_PROPS = {'pos', 'vel', 'mass', 'size', 'speed', 'species', 'active', 'ppos', 'pvel'}
            filtered_particle_states = {
                name: state_def for name, state_def in states_needed['particle'].items()
                if name.lower() not in BUILTIN_PARTICLE_PROPS
            }
            
            if filtered_particle_states:
                code_lines.append("# Particle states")
                code_lines.append("if 'llm_particle' not in tv.s:")
                code_lines.append("    particle_states = {")
                
                for name, state_def in filtered_particle_states.items():
                    type_str = state_def.type if hasattr(state_def, 'type') else state_def.get('type', 'ti.f32')
                    min_val = state_def.min if hasattr(state_def, 'min') else state_def.get('min', 0.0)
                    max_val = state_def.max if hasattr(state_def, 'max') else state_def.get('max', 1.0)
                    code_lines.append(f"        '{name}': ({type_str}, {min_val}, {max_val}),")
                
                code_lines.append("    }")
                code_lines.append("    tv.s.set('llm_particle', {")
                code_lines.append("        'state': particle_states,")
                code_lines.append("        'shape': tv.pn,")
                code_lines.append("        'osc': ('get',),")
                code_lines.append("        'randomise': True")
                code_lines.append("    })")
            
            # Add initialization for special states
            init_lines = []
            for name, state_def in states_needed['particle'].items():
                if 'home' in name and 'vec2' in str(state_def.type if hasattr(state_def, 'type') else state_def.get('type', '')):
                    init_lines.append("    # Initialize home positions")
                    init_lines.append("    @ti.kernel")
                    init_lines.append("    def init_home_positions():")
                    init_lines.append("        for i in range(tv.pn):")
                    init_lines.append(f"            tv.s.llm_particle.field[i].{name} = tv.p.field[i].pos")
                    init_lines.append("    init_home_positions()")
                    break
            
            if init_lines:
                code_lines.extend(init_lines)
            code_lines.append("")
        
        # Species states
        if 'species' in states_needed and states_needed['species']:
            code_lines.append("# Species states")
            code_lines.append("if 'llm_species' not in tv.s:")
            code_lines.append("    species_states = {")
            
            for name, state_def in states_needed['species'].items():
                type_str = state_def.type if hasattr(state_def, 'type') else state_def.get('type', 'ti.f32')
                min_val = state_def.min if hasattr(state_def, 'min') else state_def.get('min', 0.0)
                max_val = state_def.max if hasattr(state_def, 'max') else state_def.get('max', 1.0)
                code_lines.append(f"        '{name}': ({type_str}, {min_val}, {max_val}),")
            
            code_lines.append("    }")
            code_lines.append("    tv.s.set('llm_species', {")
            code_lines.append("        'state': species_states,")
            code_lines.append("        'shape': tv.sn,")
            code_lines.append("        'osc': ('get', 'set'),")
            code_lines.append("        'randomise': True")
            code_lines.append("    })")
        
        return "\n".join(code_lines) if code_lines else "# No custom states defined"
    
    def generate_pure_drawing(
        self,
        description: str,
        drawing_code: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a pure drawing sketch without particle systems.
        
        Args:
            description: Description of what to draw
            drawing_code: The drawing kernel/function code
            metadata: Optional metadata
            
        Returns:
            Complete pure drawing sketch code
        """
        return self.pure_drawing_template.format(
            description=description,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            drawing_code=self._indent(drawing_code.strip(), 4)
        )