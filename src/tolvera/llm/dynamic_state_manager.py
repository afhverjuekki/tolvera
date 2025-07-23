import logging
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import taichi as ti

from ..state import State

logger = logging.getLogger(__name__)


@dataclass
class StateMetadata:
    name: str
    category: str  # "global", "particle", "species"
    properties: Dict[str, Any]
    created_at_frame: int = 0


class DynamicStateManager:
    
    def __init__(self, tolvera_instance):
        """
        Initialize the state manager.
        
        Args:
            tolvera_instance: The Tölvera instance to create states in
        """
        self.tv = tolvera_instance
        self.state_registry = {}  # Track created states
        self.temporal_config = None
        self.temporal_update_kernel = None
        
        self.state_prefixes = {
            'global': 'llm_global',
            'particle': 'llm_particle', 
            'species': 'llm_species'
        }
        
        logger.info(f"Initialized DynamicStateManager for Tölvera with {self.tv.pn} particles")
    
    def create_states_from_spec(self, state_spec: Dict[str, Any]) -> Dict[str, str]:
        """
        Create Tölvera states based on the specification from StateSynthesizer.
        
        Args:
            state_spec: State specification dictionary
            
        Returns:
            Dictionary mapping state categories to their Tölvera names
        """
        created_states = {}
        
        if 'global_states' in state_spec and state_spec['global_states']:
            state_name = self._create_global_states(state_spec['global_states'])
            created_states['global'] = state_name
        
        if 'particle_states' in state_spec and state_spec['particle_states']:
            state_name = self._create_particle_states(state_spec['particle_states'])
            created_states['particle'] = state_name
        
        if 'species_states' in state_spec and state_spec['species_states']:
            state_name = self._create_species_states(state_spec['species_states'])
            created_states['species'] = state_name
        
        if 'temporal_config' in state_spec and state_spec['temporal_config']:
            tc = state_spec['temporal_config']
            if hasattr(tc, 'suggested_day_duration'):
                self.temporal_config = tc
            else:
                from .state_synthesizer import TemporalConfig
                self.temporal_config = TemporalConfig(
                    requires_time=True,
                    time_units=tc.get('time_units', []),
                    suggested_day_duration=tc.get('day_duration', tc.get('suggested_day_duration', 10.0)),
                    frame_rate=tc.get('frame_rate', 60.0),
                    time_scale=tc.get('time_scale', 1.0)
                )
            logger.info(f"Set temporal config: day_duration={self.temporal_config.suggested_day_duration}s")
        elif state_spec.get('requires_temporal') and 'day_duration' in state_spec:
            from .state_synthesizer import TemporalConfig
            self.temporal_config = TemporalConfig(
                suggested_day_duration=state_spec.get('day_duration', 10.0),
                time_units="seconds"
            )
            logger.info(f"Created temporal config from state spec: day_duration={self.temporal_config.suggested_day_duration}s")
        
        logger.info(f"Created states: {created_states}")
        logger.info(f"Tölvera state dict now contains: {list(self.tv.s.keys())}")
        return created_states
    
    def _create_global_states(self, global_states: Dict[str, Any]) -> str:
        """Create or update global state properties."""
        state_name = self.state_prefixes['global']
        
        if state_name in self.tv.s and self.tv.s[state_name] is not None:
            logger.info(f"Global state '{state_name}' already exists, merging properties")
            if state_name in self.state_registry:
                existing_props = self.state_registry[state_name].properties
                merged_props = {**existing_props, **global_states}
                global_states = merged_props
                logger.info(f"Merged global state properties: {list(global_states.keys())}")
            else:
                self.state_registry[state_name] = StateMetadata(
                    name=state_name,
                    category="global",
                    properties=global_states
                )
                logger.info(f"Created registry entry for existing global state '{state_name}'")
            
            self.state_registry[state_name].properties = global_states
            return state_name
        
        state_dict = self._convert_state_spec(global_states)
        
        self.tv.s.set(state_name, {
            "state": state_dict,
            "shape": 1,
            "osc": ("get", "set"),
            "randomise": False
        })
        
        self._initialize_global_state_values(state_name, global_states)
        
        self.state_registry[state_name] = StateMetadata(
            name=state_name,
            category="global",
            properties=global_states
        )
        
        logger.info(f"Created global state '{state_name}' with properties: {list(global_states.keys())}")
        return state_name
    
    def _create_particle_states(self, particle_states: Dict[str, Any]) -> str:
        """Create or update per-particle state properties."""
        state_name = self.state_prefixes['particle']
        
        if state_name in self.tv.s and self.tv.s[state_name] is not None:
            logger.info(f"Particle state '{state_name}' already exists, merging properties")
            if state_name in self.state_registry:
                existing_props = self.state_registry[state_name].properties
                merged_props = {**existing_props, **particle_states}
                particle_states = merged_props
                logger.info(f"Merged particle state properties: {list(particle_states.keys())}")
            else:
                self.state_registry[state_name] = StateMetadata(
                    name=state_name,
                    category="particle",
                    properties=particle_states
                )
                logger.info(f"Created registry entry for existing particle state '{state_name}'")
            
            self.state_registry[state_name].properties = particle_states
            return state_name
        
        state_dict = self._convert_state_spec(particle_states)
        
        self.tv.s.set(state_name, {
            "state": state_dict,
            "shape": self.tv.pn,
            "osc": ("get"),
            "randomise": True
        })
        
        self._initialize_particle_state_values(state_name, particle_states)
        
        self.state_registry[state_name] = StateMetadata(
            name=state_name,
            category="particle",
            properties=particle_states
        )
        
        logger.info(f"Created particle state '{state_name}' with properties: {list(particle_states.keys())}")
        return state_name
    
    def _create_species_states(self, species_states: Dict[str, Any]) -> str:
        """Create or update per-species state properties."""
        state_name = self.state_prefixes['species']
        
        if state_name in self.tv.s and self.tv.s[state_name] is not None:
            logger.info(f"Species state '{state_name}' already exists, merging properties")
            if state_name in self.state_registry:
                existing_props = self.state_registry[state_name].properties
                merged_props = {**existing_props, **species_states}
                species_states = merged_props
                logger.info(f"Merged species state properties: {list(species_states.keys())}")
            else:
                self.state_registry[state_name] = StateMetadata(
                    name=state_name,
                    category="species",
                    properties=species_states
                )
                logger.info(f"Created registry entry for existing species state '{state_name}'")
            
            self.state_registry[state_name].properties = species_states
            return state_name
        
        state_dict = self._convert_state_spec(species_states)
        
        self.tv.s.set(state_name, {
            "state": state_dict,
            "shape": self.tv.sn,
            "osc": ("get", "set"),
            "randomise": True
        })
        
        self.state_registry[state_name] = StateMetadata(
            name=state_name,
            category="species",
            properties=species_states
        )
        
        logger.info(f"Created species state '{state_name}' with properties: {list(species_states.keys())}")
        return state_name
    
    def _convert_state_spec(self, states: Dict[str, Any]) -> Dict[str, tuple]:
        """
        Convert state specification to Tölvera format.
        
        Converts from:
            {"energy": {"type": "ti.f32", "min": 0.0, "max": 1.0}}
        To:
            {"energy": (ti.f32, 0.0, 1.0)}
        """
        tolvera_states = {}
        
        for prop_name, prop_spec in states.items():
            # Get the type
            type_str = prop_spec['type']
            type_obj = self._get_taichi_type(type_str)
            
            # Get min/max values
            min_val = prop_spec['min']
            max_val = prop_spec['max']
            
            # Handle vector types which may have array min/max
            if 'vec' in type_str:
                # Convert single values to appropriate vector bounds
                if isinstance(min_val, (int, float)):
                    if 'vec2' in type_str:
                        min_val = (min_val, min_val)
                    elif 'vec3' in type_str:
                        min_val = (min_val, min_val, min_val)
                    elif 'vec4' in type_str:
                        min_val = (min_val, min_val, min_val, min_val)
                
                if isinstance(max_val, (int, float)):
                    if 'vec2' in type_str:
                        max_val = (max_val, max_val)
                    elif 'vec3' in type_str:
                        max_val = (max_val, max_val, max_val)
                    elif 'vec4' in type_str:
                        max_val = (max_val, max_val, max_val, max_val)
                
                # Convert lists to tuples for consistency
                if isinstance(min_val, list):
                    min_val = tuple(min_val)
                if isinstance(max_val, list):
                    max_val = tuple(max_val)
            
            tolvera_states[prop_name] = (type_obj, min_val, max_val)
        
        return tolvera_states
    
    def _get_taichi_type(self, type_str: str):
        """Convert string type to actual Taichi type."""
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
    
    def _initialize_global_state_values(self, state_name: str, global_states: Dict[str, Any]):
        """Initialize specific global state values that shouldn't be random."""
        # After state creation, set specific initial values
        if state_name in self.tv.s and self.tv.s[state_name] is not None:
            state_obj = self.tv.s[state_name]
            # Check if it's a State object with a field
            if hasattr(state_obj, 'field'):
                # Initialize each state with appropriate value
                for prop_name, prop_spec in global_states.items():
                    # Use 'initial' value if provided, otherwise use smart defaults
                    if 'initial' in prop_spec:
                        initial_val = prop_spec['initial']
                        setattr(state_obj.field[0], prop_name, initial_val)
                        logger.info(f"Initialized {prop_name} to specified initial value: {initial_val}")
                    elif 'frame_count' in prop_name:
                        setattr(state_obj.field[0], prop_name, 0)
                    elif 'day_phase' in prop_name:
                        setattr(state_obj.field[0], prop_name, 0.25)  # Start at dawn
                    elif any(keyword in prop_name.lower() for keyword in ['strength', 'force', 'power', 'energy', 'activity']):
                        # For force/energy-related states, use high initial value for visible effects
                        min_val = prop_spec['min']
                        max_val = prop_spec['max']
                        # Use 75% of max for good initial activity
                        initial_val = min_val + (max_val - min_val) * 0.75
                        setattr(state_obj.field[0], prop_name, initial_val)
                        logger.info(f"Initialized {prop_name} to 75% of range: {initial_val}")
    
    def _initialize_particle_state_values(self, state_name: str, particle_states: Dict[str, Any]):
        """Initialize specific particle state values, especially position states."""
        if state_name in self.tv.s and self.tv.s[state_name] is not None:
            state_obj = self.tv.s[state_name]
            # Check if it's a State object with a field
            if hasattr(state_obj, 'field'):
                # Initialize each state type appropriately
                for prop_name, prop_spec in particle_states.items():
                    # Check if initial value is specified
                    if 'initial' in prop_spec:
                        initial_val = prop_spec['initial']
                        for i in range(self.tv.pn):
                            setattr(state_obj.field[i], prop_name, initial_val)
                        logger.info(f"Initialized all particles' {prop_name} to specified value: {initial_val}")
                    
                    # Home positions: initialize to current particle positions
                    elif 'home' in prop_name.lower() and 'vec2' in prop_spec['type']:
                        logger.info(f"Initializing {prop_name} to particle positions")
                        for i in range(self.tv.pn):
                            if hasattr(self.tv.p, 'field'):
                                # Get current particle position
                                pos = self.tv.p.field[i].pos
                                # Set as home position
                                setattr(state_obj.field[i], prop_name, ti.math.vec2(pos[0], pos[1]))
                            else:
                                # Fallback to random positions in screen bounds
                                x = random.random() * self.tv.x
                                y = random.random() * self.tv.y
                                setattr(state_obj.field[i], prop_name, ti.math.vec2(x, y))
                    
                    elif any(keyword in prop_name.lower() for keyword in ['energy', 'activity', 'strength', 'power']):
                        min_val = prop_spec['min']
                        max_val = prop_spec['max']
                        initial_val = min_val + (max_val - min_val) * 0.75
                        for i in range(self.tv.pn):
                            setattr(state_obj.field[i], prop_name, initial_val)
                        logger.info(f"Initialized all particles' {prop_name} to 75% of range: {initial_val}")
    
    def get_synthesis_context(self) -> Dict[str, Any]:
        """
        Get comprehensive context information about available states for synthesis prompts.
        
        Returns:
            Dictionary with state information formatted for LLM prompts
        """
        context = {
            'available_states': {},
            'temporal_config': None,
            'state_summary': self._generate_state_summary(),
            'access_examples': self.generate_state_access_examples()
        }
        
        state_spec = {
            'global_states': {},
            'particle_states': {},
            'species_states': {},
            'temporal_config': None
        }
        
        for state_name, metadata in self.state_registry.items():
            state_info = {
                'name': f"tv.s.{state_name}",
                'category': metadata.category,
                'properties': {},
                'purpose': self._infer_state_purpose(metadata)
            }
            
            for prop_name, prop_spec in metadata.properties.items():
                prop_info = {
                    'type': prop_spec['type'],
                    'range': f"[{prop_spec['min']}, {prop_spec['max']}]",
                    'description': prop_spec.get('description', ''),
                    'access_pattern': self._get_access_pattern(state_name, prop_name, metadata.category)
                }
                state_info['properties'][prop_name] = prop_info
                
                if metadata.category == 'global' and state_name == self.state_prefixes['global']:
                    state_spec['global_states'][prop_name] = prop_spec
                elif metadata.category == 'particle' and state_name == self.state_prefixes['particle']:
                    state_spec['particle_states'][prop_name] = prop_spec
                elif metadata.category == 'species' and state_name == self.state_prefixes['species']:
                    state_spec['species_states'][prop_name] = prop_spec
            
            context['available_states'][state_name] = state_info
        
        if self.temporal_config:
            context['temporal_config'] = {
                'suggested_day_duration': self.temporal_config.suggested_day_duration,
                'frames_per_day': self.temporal_config.frames_per_day,
                'time_units': self.temporal_config.time_units,
                'time_scale': self.temporal_config.time_scale
            }
            state_spec['temporal_config'] = self.temporal_config
        
        context['state_spec'] = state_spec
        
        return context
    
    def _generate_state_summary(self) -> str:
        """Generate a human-readable summary of all available states."""
        if not self.state_registry:
            return "No custom states have been created."
        
        summary_parts = []
        
        categories = {'global': 0, 'particle': 0, 'species': 0}
        for metadata in self.state_registry.values():
            categories[metadata.category] += len(metadata.properties)
        
        if categories['global'] > 0:
            summary_parts.append(f"{categories['global']} global state(s)")
        if categories['particle'] > 0:
            summary_parts.append(f"{categories['particle']} particle state(s)")
        if categories['species'] > 0:
            summary_parts.append(f"{categories['species']} species state(s)")
        
        return "This behavior has " + ", ".join(summary_parts) + " available for use."
    
    def _infer_state_purpose(self, metadata: StateMetadata) -> str:
        purposes = []
        
        for prop_name, prop_spec in metadata.properties.items():
            desc = prop_spec.get('description', '').lower()
            
            if 'time' in prop_name or 'phase' in prop_name or 'cycle' in desc:
                purposes.append("temporal dynamics")
            elif 'energy' in prop_name or 'health' in prop_name:
                purposes.append("resource management")
            elif 'memory' in prop_name or 'history' in prop_name or 'last' in prop_name:
                purposes.append("behavioral memory")
            elif 'target' in prop_name or 'goal' in prop_name:
                purposes.append("goal-oriented behavior")
            elif 'social' in desc or 'interaction' in desc:
                purposes.append("social dynamics")
        
        return ", ".join(set(purposes)) if purposes else "behavior modulation"
    
    def _get_access_pattern(self, state_name: str, prop_name: str, category: str) -> str:
        if category == 'global':
            return f"tv.s.{state_name}.field[0].{prop_name}"
        elif category == 'particle':
            return f"tv.s.{state_name}.field[particle_idx].{prop_name}"
        elif category == 'species':
            return f"tv.s.{state_name}.field[species].{prop_name}"
        else:
            return f"tv.s.{state_name}.field[index].{prop_name}"
    
    def generate_state_access_examples(self) -> str:
        from .prompt_loader import load_prompt
        
        # Load the documentation template
        template = load_prompt("state_access_documentation")
        
        # Generate state sections
        state_sections = []
        
        # Global state access examples
        if self.state_prefixes['global'] in self.state_registry:
            section = ["# === GLOBAL STATES (shared across entire system) ==="]
            metadata = self.state_registry[self.state_prefixes['global']]
            for prop_name, prop_spec in metadata.properties.items():
                type_str = prop_spec['type']
                desc = prop_spec.get('description', '')
                range_str = f"[{prop_spec['min']}, {prop_spec['max']}]"
                section.append("#")
                section.append(f"# STATE NAME: {prop_name}")
                section.append(f"# Type: {type_str}, Range: {range_str}")
                if desc:
                    section.append(f"# Description: {desc}")
                section.append("# ACCESS CODE:")
                section.append(f"{prop_name} = tv.s.{self.state_prefixes['global']}.field[0].{prop_name}")
            state_sections.append("\n".join(section))
        
        # Particle state access examples
        if self.state_prefixes['particle'] in self.state_registry:
            section = ["# === PARTICLE STATES (individual per particle) ==="]
            metadata = self.state_registry[self.state_prefixes['particle']]
            for prop_name, prop_spec in metadata.properties.items():
                type_str = prop_spec['type']
                desc = prop_spec.get('description', '')
                range_str = f"[{prop_spec['min']}, {prop_spec['max']}]"
                section.append("#")
                section.append(f"# STATE NAME: {prop_name}")
                section.append(f"# Type: {type_str}, Range: {range_str}")
                if desc:
                    section.append(f"# Description: {desc}")
                section.append("# ACCESS CODE:")
                section.append(f"{prop_name} = tv.s.{self.state_prefixes['particle']}.field[particle_idx].{prop_name}")
            state_sections.append("\n".join(section))
        
        # Species state access examples
        if self.state_prefixes['species'] in self.state_registry:
            section = ["# === SPECIES STATES (shared by all particles of same species) ==="]
            metadata = self.state_registry[self.state_prefixes['species']]
            for prop_name, prop_spec in metadata.properties.items():
                type_str = prop_spec['type']
                desc = prop_spec.get('description', '')
                range_str = f"[{prop_spec['min']}, {prop_spec['max']}]"
                section.append("#")
                section.append(f"# STATE NAME: {prop_name}")
                section.append(f"# Type: {type_str}, Range: {range_str}")
                if desc:
                    section.append(f"# Description: {desc}")
                section.append("# ACCESS CODE:")
                section.append(f"{prop_name} = tv.s.{self.state_prefixes['species']}.field[species].{prop_name}")
            state_sections.append("\n".join(section))
        
        # Generate temporal section
        temporal_section = ""
        if self.temporal_config:
            temporal_lines = [
                "# Temporal Configuration:",
                f"# - Day duration: {self.temporal_config.suggested_day_duration} seconds",
                f"# - Frames per day: {self.temporal_config.frames_per_day}"
            ]
            if self.temporal_config.time_units:
                temporal_lines.append(f"# - Time units: {', '.join(self.temporal_config.time_units)}")
            temporal_section = "\n".join(temporal_lines)
        
        # Format the template
        return template.format(
            state_sections="\n\n".join(state_sections) if state_sections else "# No custom states defined",
            temporal_section=temporal_section
        )
    
    def clear_states(self):
        """Clear all dynamically created states."""
        # Note: We can't actually remove states from Tölvera once created,
        # but we can clear our registry and reset values
        self.state_registry.clear()
        self.temporal_config = None
        self.temporal_update_kernel = None
        logger.info("Cleared dynamic state registry")