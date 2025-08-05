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
        
        logger.info(f"Initialized StateManager for Tölvera with {self.tv.pn} particles")
    
    def create_states_from_spec(self, spec: Dict[str, Any]) -> None:
        for category in ['global', 'particle', 'species']:
            if category in spec and spec[category]:
                self._create_category_states(category, spec[category])
    
    def _create_category_states(self, category: str, states: Dict[str, StateDefinition]):
        BUILTIN_PARTICLE_PROPS = {'pos', 'vel', 'mass', 'size', 'speed', 'species', 'active', 'ppos', 'pvel'}
        
        shape_map = {
            'global': 1,
            'particle': self.tv.pn,
            'species': self.tv.sn
        }
        
        container_name = self.container_names[category]
        
        if container_name in self.tv.s and self.tv.s[container_name] is not None:
            logger.info(f"State container '{container_name}' already exists, updating registry")
            self.state_registry[category].update(states)
            return
        
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
            if isinstance(prop_def, StateDefinition):
                state_spec[prop_name] = (
                    self._get_taichi_type(prop_def.type),
                    prop_def.min,
                    prop_def.max
                )
            elif isinstance(prop_def, dict):
                state_spec[prop_name] = (
                    self._get_taichi_type(prop_def['type']),
                    prop_def['min'],
                    prop_def['max']
                )
        
        self.tv.s.set(container_name, {
            'state': state_spec,
            'shape': shape_map[category],
            'osc': ('get', 'set') if category != 'particle' else ('get',),
            'randomise': category != 'global'
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
            if isinstance(prop_def, StateDefinition):
                initial = prop_def.initial
                min_val = prop_def.min
                max_val = prop_def.max
                type_str = prop_def.type
            else:
                initial = prop_def.get('initial')
                min_val = prop_def.get('min', 0.0)
                max_val = prop_def.get('max', 1.0)
                type_str = prop_def.get('type', 'ti.f32')
            
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
                
                if not isinstance(initial_val, list) and ('i32' in type_str or 'i64' in type_str or 'int' in type_str):
                    initial_val = int(initial_val)
                    
                setattr(state_obj.field[0], prop_name, initial_val)
    
    def _init_particle_states(self, state_obj, states: Dict[str, Any]):
        import random
        
        for prop_name, prop_def in states.items():
            # Extract definition
            if isinstance(prop_def, StateDefinition):
                initial = prop_def.initial
                min_val = prop_def.min
                max_val = prop_def.max
                type_str = prop_def.type
            else:
                initial = prop_def.get('initial')
                min_val = prop_def.get('min', 0.0)
                max_val = prop_def.get('max', 1.0)
                type_str = prop_def.get('type', 'ti.f32')
            
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
                    setattr(state_obj.field[i], prop_name, initial_val)
                else:
                    # Random within range
                    if isinstance(min_val, (list, tuple)):
                        initial_val = [min_val[j] + random.random() * (max_val[j] - min_val[j]) 
                                      for j in range(len(min_val))]
                    else:
                        initial_val = min_val + random.random() * (max_val - min_val)
                        if 'i32' in type_str or 'i64' in type_str or 'int' in type_str:
                            initial_val = int(initial_val)
                    setattr(state_obj.field[i], prop_name, initial_val)
    
    def _init_species_states(self, state_obj, states: Dict[str, Any]):
        for prop_name, prop_def in states.items():
            # Extract definition
            if isinstance(prop_def, StateDefinition):
                initial = prop_def.initial
                min_val = prop_def.min
                max_val = prop_def.max
                type_str = prop_def.type
            else:
                initial = prop_def.get('initial')
                min_val = prop_def.get('min', 0.0)
                max_val = prop_def.get('max', 1.0)
                type_str = prop_def.get('type', 'ti.f32')
            
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
                        if 'i32' in type_str or 'i64' in type_str or 'int' in type_str:
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
                if isinstance(state_def, StateDefinition):
                    type_str = state_def.type
                    min_val = state_def.min
                    max_val = state_def.max
                else:
                    type_str = state_def.get('type', 'ti.f32')
                    min_val = state_def.get('min', 0.0)
                    max_val = state_def.get('max', 1.0)
                
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
                if isinstance(state_def, StateDefinition):
                    type_str = state_def.type
                    min_val = state_def.min
                    max_val = state_def.max
                else:
                    type_str = state_def.get('type', 'ti.f32')
                    min_val = state_def.get('min', 0.0)
                    max_val = state_def.get('max', 1.0)
                
                code_lines.append(f"            '{name}': ({type_str}, {min_val}, {max_val}),")
            
            code_lines.append("        },")
            code_lines.append("        'shape': tv.pn,")
            code_lines.append("        'osc': ('get',),")
            code_lines.append("        'randomise': True")
            code_lines.append("    })")
        
        if self.state_registry['species']:
            code_lines.append("\n# Species states")
            code_lines.append("if 'llm_species' not in tv.s:")
            code_lines.append("    tv.s.set('llm_species', {")
            code_lines.append("        'state': {")
            
            for name, state_def in self.state_registry['species'].items():
                if isinstance(state_def, StateDefinition):
                    type_str = state_def.type
                    min_val = state_def.min
                    max_val = state_def.max
                else:
                    type_str = state_def.get('type', 'ti.f32')
                    min_val = state_def.get('min', 0.0)
                    max_val = state_def.get('max', 1.0)
                
                code_lines.append(f"            '{name}': ({type_str}, {min_val}, {max_val}),")
            
            code_lines.append("        },")
            code_lines.append("        'shape': tv.sn,")
            code_lines.append("        'osc': ('get', 'set'),")
            code_lines.append("        'randomise': True")
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
        # Extract definition
        if isinstance(state_def, StateDefinition):
            initial = state_def.initial
            min_val = state_def.min
            max_val = state_def.max
            type_str = state_def.type
        else:
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
    
    def clear_states(self):
        self.state_registry = {
            'global': {},
            'particle': {},
            'species': {}
        }
        logger.info("Cleared state registry")