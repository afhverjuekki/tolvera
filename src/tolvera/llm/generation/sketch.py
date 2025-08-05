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
    tv = Tolvera(**kwargs)
    
    # === Configuration ===
    {config_code}
    
    # === Particle Initialization ===
{init_code}
    
    # === State Initialization ===
{state_code}
    
    # === Expert Functions ===
{expert_code}
    
    # === Integration Kernel ===
{kernel_code}
    
    # === Temporal Updates ===
{temporal_code}
    
    @tv.render
    def _():
        tv.px.diffuse(0.99)
        
        # Update states and apply behaviors
        {update_calls}
        
        # Render particles
        tv.px.particles(tv.p, tv.s.species())
        
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
        metadata: Optional[Dict[str, Any]] = None
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
            metadata: Optional metadata to include
            
        Returns:
            Complete sketch code as string
        """
        # Build update calls
        update_calls = []
        
        # Add temporal updates if present
        if temporal_code and "update_temporal_states" in temporal_code:
            update_calls.append("update_temporal_states()")
        
        # Always call particle update
        update_calls.append("tv.p()")
        
        # Always apply experts
        update_calls.append("apply_all_experts()")
        
        # Format components - indent code sections properly
        return self.template.format(
            description=description,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_code=config_code.strip() if config_code else "# No additional configuration",
            init_code=self._indent(init_code.strip(), 4),
            state_code=self._indent(state_code.strip() if state_code else "# No custom states needed", 4),
            expert_code=self._indent("\n\n".join(experts), 4),
            kernel_code=self._indent(kernel, 4),
            temporal_code=self._indent(temporal_code if temporal_code else "# No temporal updates needed", 4),
            update_calls="\n        ".join(update_calls)
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
        
        # Build config code if needed
        config_code = ""
        if tv_config:
            config_lines = []
            if 'species' in tv_config:
                config_lines.append(f"# Configured for {tv_config['species']} species")
            if 'particles' in tv_config:
                config_lines.append(f"# Particle count: {tv_config['particles']}")
            config_code = "\n".join(config_lines)
        
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