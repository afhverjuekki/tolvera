import logging
from typing import Dict, List, Any, Optional, Tuple
import taichi as ti
from .models import StateDefinition

logger = logging.getLogger(__name__)


class StateManager:
    
    def __init__(self, tv):
        self.tv = tv
        self.state_registry = {
            'global': {},
            'particle': {},
            'species': {}
        }
        
        self.container_names = {
            'global': 'llm_global',
            'particle': 'llm_particle',
            'species': 'llm_species'
        }
        
        # Track temporal updates for states
        self.temporal_updates = {
            'global': {},
            'particle': {},
            'species': {}
        }
        
        logger.info(f"Initialized StateManager for Tölvera with {self.tv.pn} particles")
    
    def create_states_from_spec(self, spec: Dict[str, Any]) -> None:
        for category in ['global', 'particle', 'species']:
            if category in spec and spec[category]:
                self._create_category_states(category, spec[category])
    
    def collect_and_create_states(self, all_states_specs: List[Dict[str, Any]]) -> None:
        """
        Collect all state requirements from multiple specs and create them once.
        
        Args:
            all_states_specs: List of state specification dictionaries
        """
        # Merge all state specifications
        merged_spec = {'global': {}, 'particle': {}, 'species': {}}
        
        for spec in all_states_specs:
            for category in ['global', 'particle', 'species']:
                if category in spec and spec[category]:
                    for state_name, state_def in spec[category].items():
                        if state_name not in merged_spec[category]:
                            merged_spec[category][state_name] = state_def
                            logger.info(f"Collected {category} state: {state_name}")
                        else:
                            logger.debug(f"State {state_name} already collected for {category}")
        
        # Create all states at once
        logger.info(f"Creating all collected states: {sum(len(states) for states in merged_spec.values())} total")
        self.create_states_from_spec(merged_spec)
    
    def _create_category_states(self, category: str, states: Dict[str, StateDefinition]):
        BUILTIN_PARTICLE_PROPS = {'pos', 'vel', 'mass', 'size', 'speed', 'species', 'active', 'ppos', 'pvel'}
        
        shape_map = {
            'global': 1,
            'particle': self.tv.pn,
            'species': self.tv.sn
        }
        
        container_name = self.container_names[category]
        
        # Check if container already exists
        if container_name in self.tv.s and self.tv.s[container_name] is not None:
            logger.info(f"State container '{container_name}' already exists, skipping creation")
            # Update registry with any new states
            for state_name, state_def in states.items():
                if state_name not in self.state_registry[category]:
                    logger.info(f"Registering new {category} state: {state_name} (container already exists)")
                    self.state_registry[category][state_name] = state_def
                    # Register temporal update if present
                    if hasattr(state_def, 'temporal_update') and state_def.temporal_update:
                        self.register_temporal_update(category, state_name, state_def.temporal_update)
            return  # Don't recreate the container
        
        if category == 'particle':
            filtered_states = {}
            for prop_name, prop_def in states.items():
                if prop_name.lower() in BUILTIN_PARTICLE_PROPS:
                    logger.warning(f"Skipping duplicate particle property '{prop_name}' - already exists in core particle struct")
                else:
                    filtered_states[prop_name] = prop_def
            states = filtered_states
            
            if not states:
                logger.info(f"No custom particle states needed after filtering built-in properties")
                return
        
        state_spec = {}
        for prop_name, prop_def in states.items():
            # Handle both StateDefinition/StateRequirement Pydantic models and dicts
            if hasattr(prop_def, 'type'):
                # It's a Pydantic model (StateDefinition or StateRequirement)
                type_str = prop_def.type
                taichi_type = self._get_taichi_type(type_str)
                min_val = prop_def.min
                max_val = prop_def.max
                
                # Convert to int if this is an integer type but we have float min/max
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    if isinstance(min_val, float):
                        min_val = int(min_val)
                    if isinstance(max_val, float):
                        max_val = int(max_val)
                
                state_spec[prop_name] = (taichi_type, min_val, max_val)
            elif isinstance(prop_def, dict):
                type_str = prop_def.get('type', 'ti.f32')
                taichi_type = self._get_taichi_type(type_str)
                # Set appropriate defaults based on type
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    min_val = prop_def.get('min', 0)
                    max_val = prop_def.get('max', 100)
                else:
                    min_val = prop_def.get('min', 0.0)
                    max_val = prop_def.get('max', 1.0)
                state_spec[prop_name] = (taichi_type, min_val, max_val)
        
        # Check if we have any integer states - if so, don't randomise
        # Integer states should be initialized explicitly
        has_integer_states = any(
            spec[0] in (ti.i32, ti.i64, ti.u32, ti.u64)
            for spec in state_spec.values()
        )
        
        self.tv.s.set(container_name, {
            'state': state_spec,
            'shape': shape_map[category],
            'osc': ('get', 'set') if category != 'particle' else ('get',),
            'randomise': False if has_integer_states or category == 'global' else True
        })
        
        self.state_registry[category] = states
        
        self._initialize_states(category, states)
        
        logger.info(f"Created {category} state container '{container_name}' with properties: {list(states.keys())}")
    
    def _get_taichi_type(self, type_str: str):
        type_map = {
            'ti.f32': ti.f32,
            'ti.f64': ti.f64,
            'ti.i32': ti.i32,
            'ti.i64': ti.i64,
            'ti.u32': ti.u32,
            'ti.u64': ti.u64,
            'ti.math.vec2': ti.math.vec2,
            'ti.math.vec3': ti.math.vec3,
            'ti.math.vec4': ti.math.vec4,
        }
        
        if type_str not in type_map:
            logger.warning(f"Unknown type '{type_str}', defaulting to ti.f32")
            return ti.f32
        
        return type_map[type_str]
    
    def _initialize_states(self, category: str, states: Dict[str, Any]):
        container_name = self.container_names[category]
        
        if container_name not in self.tv.s:
            return
        
        state_obj = self.tv.s[container_name]
        if not hasattr(state_obj, 'field'):
            return
        
        if category == 'global':
            self._init_global_states(state_obj, states)
        elif category == 'particle':
            self._init_particle_states(state_obj, states)
        elif category == 'species':
            self._init_species_states(state_obj, states)
    
    def _init_global_states(self, state_obj, states: Dict[str, Any]):
        for prop_name, prop_def in states.items():
            # Handle both StateDefinition objects and StateRequirement objects from behavior_requirements
            if hasattr(prop_def, 'initial'):
                # It's a Pydantic model (StateDefinition or StateRequirement)
                initial = prop_def.initial
                min_val = prop_def.min
                max_val = prop_def.max
                type_str = prop_def.type
            elif isinstance(prop_def, dict):
                # It's a dictionary
                initial = prop_def.get('initial')
                type_str = prop_def.get('type', 'ti.f32')
                # Set appropriate defaults based on type
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    min_val = prop_def.get('min', 0)
                    max_val = prop_def.get('max', 100)
                else:
                    min_val = prop_def.get('min', 0.0)
                    max_val = prop_def.get('max', 1.0)
            else:
                # Fallback for other types
                initial = None
                type_str = 'ti.f32'
                # Check if it's an integer type and set appropriate defaults
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    min_val = 0
                    max_val = 100
                else:
                    min_val = 0.0
                    max_val = 1.0
            
            if initial is not None:
                setattr(state_obj.field[0], prop_name, initial)
            elif 'day_phase' in prop_name:
                setattr(state_obj.field[0], prop_name, 0.25)  # Dawn
            elif 'frame_count' in prop_name:
                setattr(state_obj.field[0], prop_name, 0)
            else:
                prop_lower = prop_name.lower()
                
                if 'gravity' in prop_lower:
                    initial_val = 300.0 if max_val >= 300.0 else max_val * 0.3
                elif any(word in prop_lower for word in ['force', 'strength', 'power']):
                    initial_val = max_val * 0.3
                elif 'temperature' in prop_lower:
                    initial_val = min_val + (max_val - min_val) * 0.25
                elif any(word in prop_lower for word in ['rate', 'speed']):
                    initial_val = (min_val + max_val) / 2
                elif 'time' in prop_lower or 'phase' in prop_lower:
                    initial_val = min_val
                else:
                    if isinstance(min_val, (list, tuple)):
                        initial_val = [(min_val[i] + max_val[i]) / 2 for i in range(len(min_val))]
                    else:
                        initial_val = (min_val + max_val) / 2
                
                if not isinstance(initial_val, list) and ('i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str):
                    initial_val = int(initial_val)
                    
                setattr(state_obj.field[0], prop_name, initial_val)
    
    def _init_particle_states(self, state_obj, states: Dict[str, Any]):
        import random
        
        for prop_name, prop_def in states.items():
            # Extract definition - handle both Pydantic models and dicts
            if hasattr(prop_def, 'initial'):
                # It's a Pydantic model (StateDefinition or StateRequirement)
                initial = prop_def.initial
                min_val = prop_def.min
                max_val = prop_def.max
                type_str = prop_def.type
            elif isinstance(prop_def, dict):
                # It's a dictionary
                initial = prop_def.get('initial')
                type_str = prop_def.get('type', 'ti.f32')
                # Set appropriate defaults based on type
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    min_val = prop_def.get('min', 0)
                    max_val = prop_def.get('max', 100)
                else:
                    min_val = prop_def.get('min', 0.0)
                    max_val = prop_def.get('max', 1.0)
            else:
                # Fallback
                initial = None
                type_str = 'ti.f32'
                # Check if it's an integer type and set appropriate defaults
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    min_val = 0
                    max_val = 100
                else:
                    min_val = 0.0
                    max_val = 1.0
            
            for i in range(self.tv.pn):
                if initial is not None:
                    setattr(state_obj.field[i], prop_name, initial)
                elif 'home' in prop_name.lower() and 'vec2' in type_str:
                    if hasattr(self.tv.p, 'field') and i < self.tv.pn:
                        pos = self.tv.p.field[i].pos
                        setattr(state_obj.field[i], prop_name, ti.math.vec2(pos[0], pos[1]))
                    else:
                        x = random.random() * self.tv.x
                        y = random.random() * self.tv.y
                        setattr(state_obj.field[i], prop_name, ti.math.vec2(x, y))
                elif 'energy' in prop_name.lower():
                    if isinstance(min_val, (list, tuple)):
                        raise ValueError("Energy should be scalar")
                    initial_val = min_val + (max_val - min_val) * 0.8
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        initial_val = int(initial_val)
                    setattr(state_obj.field[i], prop_name, initial_val)
                else:
                    # Random within range
                    if isinstance(min_val, (list, tuple)):
                        initial_val = [min_val[j] + random.random() * (max_val[j] - min_val[j]) 
                                      for j in range(len(min_val))]
                    else:
                        initial_val = min_val + random.random() * (max_val - min_val)
                        if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                            initial_val = int(initial_val)
                    setattr(state_obj.field[i], prop_name, initial_val)
    
    def _init_species_states(self, state_obj, states: Dict[str, Any]):
        for prop_name, prop_def in states.items():
            # Extract definition - handle both Pydantic models and dicts
            if hasattr(prop_def, 'initial'):
                # It's a Pydantic model (StateDefinition or StateRequirement)
                initial = prop_def.initial
                min_val = prop_def.min
                max_val = prop_def.max
                type_str = prop_def.type
            elif isinstance(prop_def, dict):
                # It's a dictionary
                initial = prop_def.get('initial')
                type_str = prop_def.get('type', 'ti.f32')
                # Set appropriate defaults based on type
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    min_val = prop_def.get('min', 0)
                    max_val = prop_def.get('max', 100)
                else:
                    min_val = prop_def.get('min', 0.0)
                    max_val = prop_def.get('max', 1.0)
            else:
                # Fallback
                initial = None
                type_str = 'ti.f32'
                # Check if it's an integer type and set appropriate defaults
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    min_val = 0
                    max_val = 100
                else:
                    min_val = 0.0
                    max_val = 1.0
            
            for s in range(self.tv.sn):
                if initial is not None:
                    setattr(state_obj.field[s], prop_name, initial)
                else:
                    # Default to middle of range with some variation
                    if isinstance(min_val, (list, tuple)):
                        initial_val = [(min_val[i] + max_val[i]) / 2 for i in range(len(min_val))]
                    else:
                        variation = (s / max(1, self.tv.sn - 1)) * 0.4 - 0.2
                        initial_val = (min_val + max_val) / 2 + (max_val - min_val) * variation
                        initial_val = max(min_val, min(max_val, initial_val))
                        if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                            initial_val = int(initial_val)
                    setattr(state_obj.field[s], prop_name, initial_val)
    
    def get_available_states(self) -> Dict[str, List[str]]:
        return {
            cat: list(states.keys())
            for cat, states in self.state_registry.items()
            if states
        }
    
    def generate_state_initialization_code(self) -> str:
        if not any(self.state_registry.values()):
            return "# No custom states defined"
        
        code_lines = []
        
        if self.state_registry['global']:
            code_lines.append("\n# Global states")
            code_lines.append("if 'llm_global' not in tv.s:")
            code_lines.append("    tv.s.set('llm_global', {")
            code_lines.append("        'state': {")
            
            for name, state_def in self.state_registry['global'].items():
                # Handle both Pydantic models and dicts
                if hasattr(state_def, 'type'):
                    # It's a Pydantic model (StateDefinition or StateRequirement)
                    type_str = state_def.type
                    min_val = state_def.min
                    max_val = state_def.max
                    # Convert to int if this is an integer type but we have float min/max
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        if isinstance(min_val, float):
                            min_val = int(min_val)
                        if isinstance(max_val, float):
                            max_val = int(max_val)
                elif isinstance(state_def, dict):
                    type_str = state_def.get('type', 'ti.f32')
                    # Ensure integer types get integer defaults
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        min_val = state_def.get('min', 0)
                        max_val = state_def.get('max', 100)
                    else:
                        min_val = state_def.get('min', 0.0)
                        max_val = state_def.get('max', 1.0)
                else:
                    type_str = 'ti.f32'
                    # Ensure integer types get integer defaults
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        min_val = 0
                        max_val = 100
                    else:
                        min_val = 0.0
                        max_val = 1.0
                
                code_lines.append(f"            '{name}': ({type_str}, {min_val}, {max_val}),")
            
            code_lines.append("        },")
            code_lines.append("        'shape': 1,")
            code_lines.append("        'osc': ('get', 'set'),")
            code_lines.append("        'randomise': False")
            code_lines.append("    })")
        
        if self.state_registry['particle']:
            code_lines.append("\n# Particle states")
            code_lines.append("if 'llm_particle' not in tv.s:")
            code_lines.append("    tv.s.set('llm_particle', {")
            code_lines.append("        'state': {")
            
            for name, state_def in self.state_registry['particle'].items():
                # Handle both Pydantic models and dicts
                if hasattr(state_def, 'type'):
                    # It's a Pydantic model (StateDefinition or StateRequirement)
                    type_str = state_def.type
                    min_val = state_def.min
                    max_val = state_def.max
                    # Convert to int if this is an integer type but we have float min/max
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        if isinstance(min_val, float):
                            min_val = int(min_val)
                        if isinstance(max_val, float):
                            max_val = int(max_val)
                elif isinstance(state_def, dict):
                    type_str = state_def.get('type', 'ti.f32')
                    # Ensure integer types get integer defaults
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        min_val = state_def.get('min', 0)
                        max_val = state_def.get('max', 100)
                    else:
                        min_val = state_def.get('min', 0.0)
                        max_val = state_def.get('max', 1.0)
                else:
                    type_str = 'ti.f32'
                    # Ensure integer types get integer defaults
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        min_val = 0
                        max_val = 100
                    else:
                        min_val = 0.0
                        max_val = 1.0
                
                code_lines.append(f"            '{name}': ({type_str}, {min_val}, {max_val}),")
            
            code_lines.append("        },")
            code_lines.append("        'shape': tv.pn,")
            code_lines.append("        'osc': ('get',),")
            
            # Check if any particle states are integer types - if so, disable randomization
            has_integer_states = False
            for name, state_def in self.state_registry['particle'].items():
                type_str = state_def.type if hasattr(state_def, 'type') else 'ti.f32'
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    has_integer_states = True
                    break
            
            randomise_value = "False" if has_integer_states else "True"
            code_lines.append(f"        'randomise': {randomise_value}")
            code_lines.append("    })")
        
        if self.state_registry['species']:
            code_lines.append("\n# Species states")
            code_lines.append("if 'llm_species' not in tv.s:")
            code_lines.append("    tv.s.set('llm_species', {")
            code_lines.append("        'state': {")
            
            for name, state_def in self.state_registry['species'].items():
                # Handle both Pydantic models and dicts
                if hasattr(state_def, 'type'):
                    # It's a Pydantic model (StateDefinition or StateRequirement)
                    type_str = state_def.type
                    min_val = state_def.min
                    max_val = state_def.max
                    # Convert to int if this is an integer type but we have float min/max
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        if isinstance(min_val, float):
                            min_val = int(min_val)
                        if isinstance(max_val, float):
                            max_val = int(max_val)
                elif isinstance(state_def, dict):
                    type_str = state_def.get('type', 'ti.f32')
                    # Ensure integer types get integer defaults
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        min_val = state_def.get('min', 0)
                        max_val = state_def.get('max', 100)
                    else:
                        min_val = state_def.get('min', 0.0)
                        max_val = state_def.get('max', 1.0)
                else:
                    type_str = 'ti.f32'
                    # Ensure integer types get integer defaults
                    if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                        min_val = 0
                        max_val = 100
                    else:
                        min_val = 0.0
                        max_val = 1.0
                
                code_lines.append(f"            '{name}': ({type_str}, {min_val}, {max_val}),")
            
            code_lines.append("        },")
            code_lines.append("        'shape': tv.sn,")
            code_lines.append("        'osc': ('get', 'set'),")
            
            # Check if any species states are integer types - if so, disable randomization
            has_integer_states = False
            for name, state_def in self.state_registry['species'].items():
                type_str = state_def.type if hasattr(state_def, 'type') else 'ti.f32'
                if 'i32' in type_str or 'i64' in type_str or 'u32' in type_str or 'u64' in type_str or 'int' in type_str:
                    has_integer_states = True
                    break
            
            randomise_value = "False" if has_integer_states else "True"
            code_lines.append(f"        'randomise': {randomise_value}")
            code_lines.append("    })")
        
        return "\n".join(code_lines)
    
    def generate_state_value_initialization_code(self) -> str:
        if not any(self.state_registry.values()):
            return "# No state values to initialize"
        
        code_lines = []
        
        # Global states - these need explicit initialization
        if self.state_registry['global']:
            code_lines.append("\n# Initialize global state values")
            for name, state_def in self.state_registry['global'].items():
                initial_val = self._get_initial_value_for_codegen(name, state_def, 'global')
                if initial_val is not None:
                    # Ensure integer types get integer values
                    type_str = state_def.type if hasattr(state_def, 'type') else state_def.get('type', 'ti.f32')
                    if 'i32' in str(type_str) or 'i64' in str(type_str):
                        # Remove decimal point from float string representation
                        if '.' in str(initial_val):
                            initial_val = str(int(float(initial_val)))
                    code_lines.append(f"tv.s.llm_global.field[0].{name} = {initial_val}")
        
        # Particle states - only if they need specific initialization
        if self.state_registry['particle']:
            # Check if any particle states need explicit initialization
            needs_init = False
            for name, state_def in self.state_registry['particle'].items():
                if self._needs_explicit_initialization(name, state_def, 'particle'):
                    needs_init = True
                    break
            
            if needs_init:
                code_lines.append("\n# Initialize particle state values")
                code_lines.append("@ti.kernel")
                code_lines.append("def init_particle_states():")
                code_lines.append("    for i in range(tv.pn):")
                
                for name, state_def in self.state_registry['particle'].items():
                    initial_val = self._get_initial_value_for_codegen(name, state_def, 'particle')
                    if initial_val is not None:
                        # Ensure integer types get integer values
                        type_str = state_def.type if hasattr(state_def, 'type') else state_def.get('type', 'ti.f32')
                        if 'i32' in str(type_str) or 'i64' in str(type_str):
                            # Remove decimal point from float string representation
                            if '.' in str(initial_val):
                                initial_val = str(int(float(initial_val)))
                        code_lines.append(f"        tv.s.llm_particle.field[i].{name} = {initial_val}")
                
                code_lines.append("\ninit_particle_states()")
        
        # Species states - only if they need specific initialization
        if self.state_registry['species']:
            needs_init = False
            for name, state_def in self.state_registry['species'].items():
                if self._needs_explicit_initialization(name, state_def, 'species'):
                    needs_init = True
                    break
            
            if needs_init:
                code_lines.append("\n# Initialize species state values")
                for s in range(self.tv.sn):
                    for name, state_def in self.state_registry['species'].items():
                        initial_val = self._get_initial_value_for_codegen(name, state_def, 'species', species_id=s)
                        if initial_val is not None:
                            # Ensure integer types get integer values
                            type_str = state_def.type if hasattr(state_def, 'type') else state_def.get('type', 'ti.f32')
                            if 'i32' in str(type_str) or 'i64' in str(type_str):
                                # Remove decimal point from float string representation
                                if '.' in str(initial_val):
                                    initial_val = str(int(float(initial_val)))
                            code_lines.append(f"tv.s.llm_species.field[{s}].{name} = {initial_val}")
        
        return "\n".join(code_lines) if code_lines else "# State values use defaults"
    
    def _get_initial_value_for_codegen(self, name: str, state_def: Any, category: str, species_id: int = 0):
        # Import here to avoid circular import
        from ..core.behavior_requirements import StateRequirement
        
        # Extract definition based on object type
        if isinstance(state_def, StateDefinition):
            initial = state_def.initial
            min_val = state_def.min
            max_val = state_def.max
            type_str = state_def.type
        elif isinstance(state_def, StateRequirement):
            initial = state_def.initial
            min_val = state_def.min
            max_val = state_def.max
            type_str = state_def.type
        else:
            # Assume it's a dict
            initial = state_def.get('initial')
            min_val = state_def.get('min', 0.0)
            max_val = state_def.get('max', 1.0)
            type_str = state_def.get('type', 'ti.f32')
        
        # If explicit initial value is provided, use it
        if initial is not None:
            if isinstance(initial, list):
                return f"ti.math.vec{len(initial)}({', '.join(map(str, initial))})"
            return str(initial)
        
        # For global states, we always want to initialize them explicitly
        if category == 'global':
            prop_lower = name.lower()
            
            # Physics forces should have meaningful defaults
            if 'gravity' in prop_lower:
                # Gravity should be a reasonable physics value
                return "300.0" if max_val >= 300.0 else str(max_val * 0.3)
            elif any(word in prop_lower for word in ['force', 'strength', 'power']):
                # Other forces start at moderate values
                return str(max_val * 0.3)
            elif 'temperature' in prop_lower:
                # Temperature often starts at room temp
                return str(min_val + (max_val - min_val) * 0.25)
            elif any(word in prop_lower for word in ['rate', 'speed']):
                # Rates often start moderate
                return str((min_val + max_val) / 2)
            elif 'time' in prop_lower or 'phase' in prop_lower:
                # Time/phase usually starts at 0
                return str(min_val)
            else:
                # Default to middle of range for other global states
                if isinstance(min_val, list):
                    mid_vals = [(min_val[i] + max_val[i]) / 2 for i in range(len(min_val))]
                    return f"ti.math.vec{len(mid_vals)}({', '.join(map(str, mid_vals))})"
                else:
                    return str((min_val + max_val) / 2)
        
        # For particle/species states, only return explicit values if needed
        return None
    
    def _needs_explicit_initialization(self, name: str, state_def: Any, category: str) -> bool:
        # Global states always need explicit initialization
        if category == 'global':
            return True
        
        # Check if there's an explicit initial value
        if isinstance(state_def, StateDefinition) and state_def.initial is not None:
            return True
        elif isinstance(state_def, dict) and state_def.get('initial') is not None:
            return True
        
        # Some particle states need special initialization
        prop_lower = name.lower() if isinstance(name, str) else ""
        if 'home' in prop_lower or 'target' in prop_lower:
            return True
        
        return False
    
    def register_temporal_update(self, category: str, state_name: str, temporal_update) -> None:
        """Register a temporal update rule for a state.
        
        Args:
            category: State category (global, particle, species)
            state_name: Name of the state
            temporal_update: TemporalUpdate object or dict with update info
        """
        if category not in self.temporal_updates:
            logger.warning(f"Unknown category '{category}' for temporal update")
            return
            
        self.temporal_updates[category][state_name] = temporal_update
        logger.info(f"Registered temporal update for {category}.{state_name}")
    
    def generate_temporal_update_kernel(self) -> str:
        """Generate a complete temporal update kernel from all registered temporal updates."""
        # Check if we have any temporal updates
        has_updates = any(
            updates for updates in self.temporal_updates.values()
        )
        
        if not has_updates:
            return ""
        
        kernel_lines = [
            "@ti.kernel",
            "def update_temporal_states():",
            "    '''Update all temporal states based on registered rules.'''",
            "    frame = tv.ctx.i[None]",
            "    dt = 1.0 / 60.0  # Assuming 60 FPS",
            ""
        ]
        
        # Generate global state updates
        if self.temporal_updates['global']:
            kernel_lines.append("    # Global state updates")
            for state_name, update in self.temporal_updates['global'].items():
                update_expr = self._format_temporal_update(
                    update, state_name, 'global', '0'
                )
                if update_expr:
                    kernel_lines.extend(update_expr)
            kernel_lines.append("")
        
        # Generate particle state updates
        if self.temporal_updates['particle']:
            kernel_lines.append("    # Particle state updates")
            kernel_lines.append("    for i in range(tv.pn):")
            kernel_lines.append("        if tv.p.field[i].active > 0:")
            
            for state_name, update in self.temporal_updates['particle'].items():
                update_expr = self._format_temporal_update(
                    update, state_name, 'particle', 'i', indent=3
                )
                if update_expr:
                    kernel_lines.extend(update_expr)
            kernel_lines.append("")
        
        # Generate species state updates
        if self.temporal_updates['species']:
            kernel_lines.append("    # Species state updates")
            kernel_lines.append("    for s in range(tv.sn):")
            
            for state_name, update in self.temporal_updates['species'].items():
                update_expr = self._format_temporal_update(
                    update, state_name, 'species', 's', indent=2
                )
                if update_expr:
                    kernel_lines.extend(update_expr)
        
        return "\n".join(kernel_lines)
    
    def _format_temporal_update(self, update, state_name: str, category: str, 
                                index: str, indent: int = 1) -> List[str]:
        """Format a single temporal update into kernel code."""
        lines = []
        ind = "    " * indent
        container = self.container_names[category]
        
        # Handle both TemporalUpdate objects and dicts
        if hasattr(update, 'update_expression'):
            # It's a TemporalUpdate pydantic model
            update_expr = update.update_expression
            update_cond = update.update_condition
            update_freq = update.update_frequency
            affects = update.affects_behavior
        elif isinstance(update, dict):
            # It's a dictionary
            update_expr = update.get('update_expression', '')
            update_cond = update.get('update_condition')
            update_freq = update.get('update_frequency', 1)
            affects = update.get('affects_behavior')
        else:
            return lines
        
        # Check update frequency
        if update_freq > 1:
            lines.append(f"{ind}if frame % {update_freq} == 0:")
            ind += "    "
        
        # Add condition if specified
        if update_cond:
            # Parse condition to replace state references
            condition = update_cond
            if category == 'particle':
                condition = condition.replace('species', 'tv.p.field[i].species')
                condition = condition.replace('vel', 'tv.p.field[i].vel')
            lines.append(f"{ind}if {condition}:")
            ind += "    "
        
        # Get current value
        lines.append(f"{ind}value = tv.s.{container}.field[{index}].{state_name}")
        
        # Apply update expression
        # Parse the expression to replace 'value' with actual value
        expr = update_expr.replace(state_name, 'value')
        
        # Handle special variables in expression
        if category == 'particle':
            expr = expr.replace('vel.norm()', 'tv.p.field[i].vel.norm()')
            expr = expr.replace('speed', 'tv.p.field[i].vel.norm()')
        
        lines.append(f"{ind}{expr}")
        
        # Get state bounds from registry
        state_def = self.state_registry.get(category, {}).get(state_name)
        if state_def:
            min_val = state_def.min if hasattr(state_def, 'min') else state_def.get('min', 0.0)
            max_val = state_def.max if hasattr(state_def, 'max') else state_def.get('max', 1.0)
            
            # Clamp value
            lines.append(f"{ind}value = max({min_val}, min({max_val}, value))")
        
        # Store updated value
        lines.append(f"{ind}tv.s.{container}.field[{index}].{state_name} = value")
        
        # Apply behavioral effects if specified
        if affects and category == 'particle':
            lines.append(f"{ind}# Behavioral coupling")
            # Parse affects to apply behavioral changes
            affects_parsed = affects.replace(':', ':\n' + ind + '    ')
            lines.append(f"{ind}{affects_parsed}")
        
        return lines
    
    def clear_states(self):
        self.state_registry = {
            'global': {},
            'particle': {},
            'species': {}
        }
        self.temporal_updates = {
            'global': {},
            'particle': {},
            'species': {}
        }
        logger.info("Cleared state registry and temporal updates")