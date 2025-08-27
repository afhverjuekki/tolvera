from typing import Dict, List, Any, Optional, Tuple, Union
from jinja2 import Environment, FileSystemLoader
import os
import taichi as ti
from .data_models import StateDefinition


class StateManager:
    """Manages custom states for particle simulations.
    
    This class provides a single source of truth for all state management,
    including creation, initialization, and code generation for global,
    particle, and species states.
    """
    
    # Built-in particle properties that should not be duplicated
    BUILTIN_PARTICLE_PROPS = frozenset({
        'pos', 'vel', 'mass', 'size', 'speed', 'species', 
        'active', 'ppos', 'pvel'
    })
    
    # Taichi type mapping
    TYPE_MAP = {
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
    
    # Default ranges for different types
    DEFAULT_RANGES = {
        'float': (0.0, 1.0),
        'int': (0, 100),
    }
    
    def __init__(self, tv):
        """Initialize the StateManager.
        
        Args:
            tv: Tölvera instance for accessing particle system
        """
        self.tv = tv
        
        # State registry tracks all created states
        self.state_registry = {
            'global': {},
            'particle': {},
            'species': {}
        }
        
        # Container names for state storage
        self.container_names = {
            'global': 'llm_global',
            'particle': 'llm_particle',
            'species': 'llm_species'
        }
        
        # Initialize Jinja2 environment for template rendering
        templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
        self.env = Environment(loader=FileSystemLoader(templates_dir))
    
    def create_states_from_spec(self, spec: Dict[str, Any]) -> None:
        """Create states from a specification dictionary.
        
        Args:
            spec: Dictionary with 'global', 'particle', and/or 'species' keys
                  containing state definitions
        """
        for category in ['global', 'particle', 'species']:
            if category in spec and spec[category]:
                self._create_category_states(category, spec[category])
    
    def collect_and_create_states(self, all_states_specs: List[Dict[str, Any]]) -> None:
        """Collect and merge state specifications from multiple sources.
        
        This method consolidates state requirements from multiple specifications
        and creates them all at once, avoiding duplicate state creation.
        
        Args:
            all_states_specs: List of state specification dictionaries
        """
        # Merge all state specifications
        merged_spec = {'global': {}, 'particle': {}, 'species': {}}
        
        for spec in all_states_specs:
            for category in ['global', 'particle', 'species']:
                if category not in spec:
                    continue
                    
                for state_name, state_def in spec[category].items():
                    if state_name not in merged_spec[category]:
                        merged_spec[category][state_name] = state_def
        
        # Create all states at once
        total_states = sum(len(states) for states in merged_spec.values())
        if total_states > 0:
            self.create_states_from_spec(merged_spec)
    
    def _extract_state_params(self, state_def: Any) -> Tuple[str, Any, Any, Optional[Any]]:
        """Extract parameters from a state definition (Pydantic model or dict).
        
        This centralizes the logic for handling both StateDefinition objects
        and plain dictionaries, eliminating code duplication.
        
        Args:
            state_def: StateDefinition object or dictionary
            
        Returns:
            Tuple of (type_str, min_val, max_val, initial_val)
        """
        if isinstance(state_def, StateDefinition):
            # Pydantic model
            type_str = state_def.type
            min_val = state_def.min
            max_val = state_def.max
            initial = state_def.initial
        elif isinstance(state_def, dict):
            # Dictionary
            type_str = state_def.get('type', 'ti.f32')
            initial = state_def.get('initial')
            
            # Determine appropriate defaults based on type
            is_integer = any(t in type_str for t in ['i32', 'i64', 'u32', 'u64', 'int'])
            default_min, default_max = self.DEFAULT_RANGES['int' if is_integer else 'float']
            
            min_val = state_def.get('min', default_min)
            max_val = state_def.get('max', default_max)
        else:
            # Fallback for unexpected types
            type_str = 'ti.f32'
            min_val, max_val = self.DEFAULT_RANGES['float']
            initial = None
        
        # Ensure integer types have integer bounds
        if self._is_integer_type(type_str) and isinstance(min_val, float):
            min_val = int(min_val)
            max_val = int(max_val)
            if initial is not None and isinstance(initial, float):
                initial = int(initial)
        
        return type_str, min_val, max_val, initial
    
    def _is_integer_type(self, type_str: str) -> bool:
        """Check if a type string represents an integer type."""
        return any(t in type_str for t in ['i32', 'i64', 'u32', 'u64', 'int'])
    
    def _get_taichi_type(self, type_str: str):
        """Convert type string to Taichi type object.
        
        Args:
            type_str: String representation of type
            
        Returns:
            Taichi type object
        """
        if type_str not in self.TYPE_MAP:
            return ti.f32
        return self.TYPE_MAP[type_str]
    
    def _create_category_states(self, category: str, states: Dict[str, Any]) -> None:
        """Create states for a specific category.
        
        Args:
            category: 'global', 'particle', or 'species'
            states: Dictionary of state definitions
        """
        container_name = self.container_names[category]
        
        # Check if container already exists
        if container_name in self.tv.s and self.tv.s[container_name] is not None:
            # Update registry with new states
            for state_name, state_def in states.items():
                if state_name not in self.state_registry[category]:
                    self.state_registry[category][state_name] = state_def
            return
        
        # Filter out built-in particle properties
        if category == 'particle':
            states = self._filter_builtin_properties(states)
            if not states:
                return
        
        # Build state specification
        state_spec = self._build_state_spec(states)
        
        # Determine shape based on category
        shape_map = {'global': 1, 'particle': self.tv.pn, 'species': self.tv.sn}
        shape = shape_map[category]
        
        # Determine randomization policy
        has_integer_states = any(
            self._is_integer_type(self._extract_state_params(state_def)[0])
            for state_def in states.values()
        )
        randomise = not (has_integer_states or category == 'global')
        
        # Create state container
        self.tv.s.set(container_name, {
            'state': state_spec,
            'shape': shape,
            'osc': ('get', 'set') if category != 'particle' else ('get',),
            'randomise': randomise
        })
        
        # Update registry and initialize
        self.state_registry[category] = states
        self._initialize_states(category, states)
    
    def _filter_builtin_properties(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out built-in particle properties from state definitions.
        
        Args:
            states: Dictionary of state definitions
            
        Returns:
            Filtered dictionary without built-in properties
        """
        filtered = {}
        for prop_name, prop_def in states.items():
            if prop_name.lower() not in self.BUILTIN_PARTICLE_PROPS:
                filtered[prop_name] = prop_def
        return filtered
    
    def _build_state_spec(self, states: Dict[str, Any]) -> Dict[str, Tuple]:
        """Build Taichi state specification from state definitions.
        
        Args:
            states: Dictionary of state definitions
            
        Returns:
            Dictionary of state specifications for Taichi
        """
        state_spec = {}
        for prop_name, prop_def in states.items():
            type_str, min_val, max_val, _ = self._extract_state_params(prop_def)
            taichi_type = self._get_taichi_type(type_str)
            state_spec[prop_name] = (taichi_type, min_val, max_val)
        return state_spec
    
    def _initialize_states(self, category: str, states: Dict[str, Any]) -> None:
        """Initialize state values based on category.
        
        Args:
            category: State category
            states: Dictionary of state definitions
        """
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
    
    def _init_global_states(self, state_obj, states: Dict[str, Any]) -> None:
        """Initialize global state values."""
        for prop_name, prop_def in states.items():
            type_str, min_val, max_val, initial = self._extract_state_params(prop_def)
            
            if initial is not None:
                value = initial
            else:
                value = self._get_default_initial_value(prop_name, min_val, max_val)
            
            # Ensure integer types get integer values
            if self._is_integer_type(type_str) and isinstance(value, float):
                value = int(value)
            
            setattr(state_obj.field[0], prop_name, value)
    
    def _init_particle_states(self, state_obj, states: Dict[str, Any]) -> None:
        """Initialize particle state values."""
        import random
        
        for prop_name, prop_def in states.items():
            type_str, min_val, max_val, initial = self._extract_state_params(prop_def)
            
            for i in range(self.tv.pn):
                if initial is not None:
                    value = initial
                else:
                    # Random within range for all particle states
                    if isinstance(min_val, list):
                        value = [min_val[j] + random.random() * (max_val[j] - min_val[j]) 
                                for j in range(len(min_val))]
                    else:
                        value = min_val + random.random() * (max_val - min_val)
                
                # Ensure integer types get integer values
                if self._is_integer_type(type_str) and isinstance(value, (float, int)):
                    value = int(value)
                
                setattr(state_obj.field[i], prop_name, value)
    
    def _init_species_states(self, state_obj, states: Dict[str, Any]) -> None:
        """Initialize species state values."""
        for prop_name, prop_def in states.items():
            type_str, min_val, max_val, initial = self._extract_state_params(prop_def)
            
            for s in range(self.tv.sn):
                if initial is not None:
                    value = initial
                else:
                    # Default with variation across species
                    if isinstance(min_val, list):
                        value = [(min_val[i] + max_val[i]) / 2 for i in range(len(min_val))]
                    else:
                        variation = (s / max(1, self.tv.sn - 1)) * 0.4 - 0.2
                        value = (min_val + max_val) / 2 + (max_val - min_val) * variation
                        value = max(min_val, min(max_val, value))
                
                # Ensure integer types get integer values
                if self._is_integer_type(type_str) and isinstance(value, float):
                    value = int(value)
                
                setattr(state_obj.field[s], prop_name, value)
    
    def _prepare_template_context(self) -> Dict[str, Any]:
        """Prepare context data for the state initialization template.
        
        Returns:
            Dictionary with all necessary data for template rendering
        """
        categories = []
        
        for cat_name in ['global', 'particle', 'species']:
            if not self.state_registry[cat_name]:
                continue
                
            cat_data = {
                'name': cat_name,
                'container_name': self.container_names[cat_name],
                'has_states': bool(self.state_registry[cat_name]),
                'states': {},
                'init_values': [],
                'needs_explicit_init': False,
                'particle_kernel_statements': []
            }
            
            # Determine shape
            shape_map = {'global': '1', 'particle': 'tv.pn', 'species': 'tv.sn'}
            cat_data['shape'] = shape_map[cat_name]
            
            # Determine OSC settings
            cat_data['osc'] = ['get'] if cat_name == 'particle' else ['get', 'set']
            
            # Process states
            has_integer = False
            for name, state_def in self.state_registry[cat_name].items():
                type_str, min_val, max_val, initial = self._extract_state_params(state_def)
                
                # Check for integer types
                if self._is_integer_type(type_str):
                    has_integer = True
                
                cat_data['states'][name] = {
                    'type': type_str,
                    'min': min_val,
                    'max': max_val,
                    'initial': initial
                }
            
            # Determine randomization policy
            cat_data['randomise'] = False if has_integer or cat_name == 'global' else True
            cat_data['has_integer'] = has_integer
            
            # Prepare initialization values
            if cat_name == 'global':
                # Global states always need initialization
                for name, state_def in self.state_registry[cat_name].items():
                    initial_val = self._get_initial_value_for_codegen(name, state_def, 'global')
                    if initial_val is not None:
                        cat_data['init_values'].append(f"tv.s.llm_global.field[0].{name} = {initial_val}")
            
            elif cat_name == 'particle':
                # Check if particle states need explicit init
                cat_data['needs_explicit_init'] = self._category_needs_explicit_init('particle')
                if cat_data['needs_explicit_init']:
                    for name, state_def in self.state_registry['particle'].items():
                        initial_val = self._get_initial_value_for_codegen(name, state_def, 'particle')
                        if initial_val is not None:
                            cat_data['particle_kernel_statements'].append(
                                f"tv.s.llm_particle.field[i].{name} = {initial_val}"
                            )
            
            elif cat_name == 'species':
                # Species states initialization
                if self._category_needs_explicit_init('species'):
                    for s in range(self.tv.sn):
                        for name, state_def in self.state_registry['species'].items():
                            initial_val = self._get_initial_value_for_codegen(
                                name, state_def, 'species', species_id=s
                            )
                            if initial_val is not None:
                                cat_data['init_values'].append(
                                    f"tv.s.llm_species.field[{s}].{name} = {initial_val}"
                                )
            
            categories.append(cat_data)
        
        return {'categories': categories}
    
    def render_state_initialization(self) -> str:
        """Render state initialization code using Jinja2 template.
        
        Returns:
            Complete state initialization code
        """
        if not any(self.state_registry.values()):
            return "# No custom states defined"
        
        # Prepare template context
        context = self._prepare_template_context()
        
        # Load and render template
        template = self.env.get_template('state/state_initialization.j2')
        return template.render(**context)
    
    def _get_default_initial_value(self, prop_name: str, min_val: Any, max_val: Any) -> Any:
        """Get default initial value for a state property.
        
        When no explicit initial value is provided by the LLM, this method
        returns a sensible default (midpoint of the valid range).
        
        Args:
            prop_name: Name of the property (unused, kept for compatibility)
            min_val: Minimum value or list of minimum values
            max_val: Maximum value or list of maximum values
            
        Returns:
            Default initial value (midpoint of range)
        """
        # Return midpoint of range as a sensible default
        if isinstance(min_val, list):
            return [(min_val[i] + max_val[i]) / 2 for i in range(len(min_val))]
        else:
            return (min_val + max_val) / 2
    
    def get_available_states(self) -> Dict[str, List[str]]:
        """Get all available states by category.
        
        Returns:
            Dictionary mapping categories to lists of state names
        """
        return {
            cat: list(states.keys())
            for cat, states in self.state_registry.items()
            if states
        }
    
    def generate_state_initialization_code(self, behavior_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate state container initialization code using template.
        
        Args:
            behavior_context: Optional context for intelligent defaults (unused but kept for compatibility)
            
        Returns:
            Python code string for state initialization
        """
        return self.render_state_initialization()
    
    def generate_state_value_initialization_code(self, behavior_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate state value initialization code using template.
        
        Args:
            behavior_context: Optional context for intelligent initialization (unused but kept for compatibility)
            
        Returns:
            Python code string for value initialization
        """
        # This is now handled by the unified render_state_initialization method
        return ""  # The template handles both container and value initialization
    
    def _category_needs_explicit_init(self, category: str) -> bool:
        """Check if a category needs explicit initialization code.
        
        Args:
            category: State category
            
        Returns:
            True if explicit initialization is needed
        """
        if category == 'global':
            return True
        
        for name, state_def in self.state_registry[category].items():
            if self._needs_explicit_initialization(name, state_def):
                return True
        
        return False
    
    def _needs_explicit_initialization(self, name: str, state_def: Any) -> bool:
        """Check if a specific state needs explicit initialization.
        
        Args:
            name: State name (unused, kept for compatibility)
            state_def: State definition
            
        Returns:
            True if explicit initialization is needed
        """
        # Only need explicit initialization if an initial value was provided
        _, _, _, initial = self._extract_state_params(state_def)
        return initial is not None
    
    def _get_initial_value_for_codegen(self, name: str, state_def: Any, category: str, 
                                       species_id: int = 0) -> Optional[str]:
        """Get initial value for code generation.
        
        Args:
            name: State name
            state_def: State definition
            category: State category
            species_id: Species ID for species-specific states
            
        Returns:
            String representation of initial value or None
        """
        type_str, min_val, max_val, initial = self._extract_state_params(state_def)
        
        # Use explicit initial if provided
        if initial is not None:
            if isinstance(initial, list):
                return f"ti.math.vec{len(initial)}({', '.join(map(str, initial))})"
            return str(int(initial) if self._is_integer_type(type_str) else initial)
        
        # Global states always need initialization
        if category == 'global':
            value = self._get_default_initial_value(name, min_val, max_val)
            
            if isinstance(value, list):
                return f"ti.math.vec{len(value)}({', '.join(map(str, value))})"
            else:
                if self._is_integer_type(type_str):
                    return str(int(value))
                return str(value)
        
        # Other categories only if special initialization needed
        return None
    
    def clear_states(self) -> None:
        """Clear all state registrations."""
        self.state_registry = {
            'global': {},
            'particle': {},
            'species': {}
        }
    
    def generate_initialization_components(self, particle_count=None, species_config=None):
        """Generate initialization code components for sketch metadata.
        
        Args:
            particle_count: Optional particle count override
            species_config: Optional species configuration
            
        Returns:
            Dictionary with init_code, state_code, and config_code
        """
        # Generate initialization code
        init_template = self.env.get_template('init/particle_initialization.j2')
        
        init_context = {
            'init_type': 'random',
            'species_count': len(species_config.species_ids) if species_config else 1,
            'uniform_speed': False,
            'speed_magnitude': 100.0,
            'grid_size': None,
            'species_colors': {}
        }
        
        if species_config and species_config.colors:
            for color_mapping in species_config.colors:
                init_context['species_colors'][color_mapping.species_id] = color_mapping.rgba
        
        init_code = init_template.render(**init_context)
        
        # Generate state code using the new template
        state_code = self.render_state_initialization()
        
        # Generate configuration
        config_code = ""
        if species_config or particle_count:
            species_count = len(species_config.species_ids) if species_config else 1
            particle_count = particle_count if particle_count else 1000
            
            config_code = f"""# Override default Tölvera parameters
    # Detected {species_count} species from description
    import sys
    if 'species' not in kwargs:
        kwargs['species'] = {species_count}
    if 'particles' not in kwargs:
        kwargs['particles'] = {particle_count}
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080"""
        
        return {
            'init_code': init_code,
            'state_code': state_code,
            'config_code': config_code
        }