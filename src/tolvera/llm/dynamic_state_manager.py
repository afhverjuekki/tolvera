"""
Dynamic State Manager for creating and managing LLM-generated states in Tölvera.

This module creates Tölvera states based on specifications from the StateSynthesizer.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import taichi as ti

from ..state import State

logger = logging.getLogger(__name__)


@dataclass
class StateMetadata:
    """Metadata about a created state."""
    name: str
    category: str  # "global", "particle", "species"
    properties: Dict[str, Any]
    created_at_frame: int = 0


class DynamicStateManager:
    """Creates and manages dynamically generated states in Tölvera."""
    
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
        
        # State name prefixes for LLM-generated states
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
        
        # Create global states
        if 'global_states' in state_spec and state_spec['global_states']:
            state_name = self._create_global_states(state_spec['global_states'])
            created_states['global'] = state_name
        
        # Create particle states
        if 'particle_states' in state_spec and state_spec['particle_states']:
            state_name = self._create_particle_states(state_spec['particle_states'])
            created_states['particle'] = state_name
        
        # Create species states
        if 'species_states' in state_spec and state_spec['species_states']:
            state_name = self._create_species_states(state_spec['species_states'])
            created_states['species'] = state_name
        
        # Store temporal config if present or create from day_duration
        if 'temporal_config' in state_spec:
            self.temporal_config = state_spec['temporal_config']
            logger.info(f"Set temporal config: day_duration={self.temporal_config.suggested_day_duration}s")
        elif state_spec.get('requires_temporal') and 'day_duration' in state_spec:
            # Create a temporal config from the state spec
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
        
        # Check if state already exists
        if hasattr(self.tv.s, state_name):
            logger.info(f"Global state '{state_name}' already exists, merging properties")
            # Merge with existing properties
            if state_name in self.state_registry:
                existing_props = self.state_registry[state_name].properties
                merged_props = {**existing_props, **global_states}
                global_states = merged_props
                logger.info(f"Merged global state properties: {list(global_states.keys())}")
            else:
                # Create registry entry even if state exists
                self.state_registry[state_name] = StateMetadata(
                    name=state_name,
                    category="global",
                    properties=global_states
                )
                logger.info(f"Created registry entry for existing global state '{state_name}'")
            
            # Update registry with merged properties
            self.state_registry[state_name].properties = global_states
            return state_name
        
        # Convert state spec to Tölvera format
        state_dict = self._convert_state_spec(global_states)
        
        # Create the state with shape=1 for global
        self.tv.s[state_name] = {
            "state": state_dict,
            "shape": 1,
            "osc": ("get", "set"),
            "randomise": False  # Global states usually shouldn't be randomized
        }
        
        # Initialize specific values if needed
        self._initialize_global_state_values(state_name, global_states)
        
        # Register the state
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
        
        # Check if state already exists
        if hasattr(self.tv.s, state_name):
            logger.info(f"Particle state '{state_name}' already exists, merging properties")
            # Merge with existing properties
            if state_name in self.state_registry:
                existing_props = self.state_registry[state_name].properties
                merged_props = {**existing_props, **particle_states}
                particle_states = merged_props
                logger.info(f"Merged particle state properties: {list(particle_states.keys())}")
            else:
                # Create registry entry even if state exists
                self.state_registry[state_name] = StateMetadata(
                    name=state_name,
                    category="particle",
                    properties=particle_states
                )
                logger.info(f"Created registry entry for existing particle state '{state_name}'")
            
            # Update registry with merged properties
            self.state_registry[state_name].properties = particle_states
            return state_name
        
        # Convert state spec to Tölvera format
        state_dict = self._convert_state_spec(particle_states)
        
        # Create the state with shape=particle_count
        self.tv.s[state_name] = {
            "state": state_dict,
            "shape": self.tv.pn,
            "osc": ("get"),
            "randomise": True  # Particle states can be randomized
        }
        
        # Initialize position states to particle positions if needed
        self._initialize_particle_state_values(state_name, particle_states)
        
        # Register the state
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
        
        # Check if state already exists
        if hasattr(self.tv.s, state_name):
            logger.info(f"Species state '{state_name}' already exists, merging properties")
            # Merge with existing properties
            if state_name in self.state_registry:
                existing_props = self.state_registry[state_name].properties
                merged_props = {**existing_props, **species_states}
                species_states = merged_props
                logger.info(f"Merged species state properties: {list(species_states.keys())}")
            else:
                # Create registry entry even if state exists
                self.state_registry[state_name] = StateMetadata(
                    name=state_name,
                    category="species",
                    properties=species_states
                )
                logger.info(f"Created registry entry for existing species state '{state_name}'")
            
            # Update registry with merged properties
            self.state_registry[state_name].properties = species_states
            return state_name
        
        # Convert state spec to Tölvera format
        state_dict = self._convert_state_spec(species_states)
        
        # Create the state with shape=species_count
        self.tv.s[state_name] = {
            "state": state_dict,
            "shape": self.tv.sn,
            "osc": ("get", "set"),
            "randomise": True
        }
        
        # Register the state
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
        if hasattr(self.tv.s, state_name):
            state_obj = getattr(self.tv.s, state_name)
            # Check if it's a State object with a field
            if hasattr(state_obj, 'field'):
                # Frame count should start at 0
                if 'frame_count' in global_states:
                    state_obj.field[0].frame_count = 0
                
                # Day phase might start at a specific time (e.g., dawn)
                if 'day_phase' in global_states:
                    state_obj.field[0].day_phase = 0.25  # Start at dawn
    
    def _initialize_particle_state_values(self, state_name: str, particle_states: Dict[str, Any]):
        """Initialize specific particle state values, especially position states."""
        if hasattr(self.tv.s, state_name):
            state_obj = getattr(self.tv.s, state_name)
            # Check if it's a State object with a field
            if hasattr(state_obj, 'field'):
                # Initialize position-related states to current particle positions
                for prop_name, prop_spec in particle_states.items():
                    if 'home' in prop_name.lower() and 'vec2' in prop_spec['type']:
                        # Initialize home positions to current particle positions
                        logger.info(f"Initializing {prop_name} to particle positions")
                        for i in range(self.tv.pn):
                            if hasattr(self.tv.p, 'field') and self.tv.p.field.shape[1] >= 2:
                                # Get current particle position
                                x = self.tv.p.field[i, 0]
                                y = self.tv.p.field[i, 1]
                                # Set as home position
                                setattr(state_obj.field[i], prop_name, ti.math.vec2(x, y))
                            else:
                                # Fallback to random positions in screen bounds
                                x = ti.random() * self.tv.x
                                y = ti.random() * self.tv.y
                                setattr(state_obj.field[i], prop_name, ti.math.vec2(x, y))
    
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
        
        # Add detailed information about each created state
        for state_name, metadata in self.state_registry.items():
            state_info = {
                'name': f"tv.s.{state_name}",
                'category': metadata.category,
                'properties': {},
                'purpose': self._infer_state_purpose(metadata)
            }
            
            # Format property information with full details
            for prop_name, prop_spec in metadata.properties.items():
                prop_info = {
                    'type': prop_spec['type'],
                    'range': f"[{prop_spec['min']}, {prop_spec['max']}]",
                    'description': prop_spec.get('description', ''),
                    'access_pattern': self._get_access_pattern(state_name, prop_name, metadata.category)
                }
                state_info['properties'][prop_name] = prop_info
            
            context['available_states'][state_name] = state_info
        
        # Add temporal config if present
        if self.temporal_config:
            context['temporal_config'] = {
                'suggested_day_duration': self.temporal_config.suggested_day_duration,
                'frames_per_day': self.temporal_config.frames_per_day,
                'time_units': self.temporal_config.time_units,
                'time_scale': self.temporal_config.time_scale
            }
        
        return context
    
    def _generate_state_summary(self) -> str:
        """Generate a human-readable summary of all available states."""
        if not self.state_registry:
            return "No custom states have been created."
        
        summary_parts = []
        
        # Count states by category
        categories = {'global': 0, 'particle': 0, 'species': 0}
        for metadata in self.state_registry.values():
            categories[metadata.category] += len(metadata.properties)
        
        # Build summary
        if categories['global'] > 0:
            summary_parts.append(f"{categories['global']} global state(s)")
        if categories['particle'] > 0:
            summary_parts.append(f"{categories['particle']} particle state(s)")
        if categories['species'] > 0:
            summary_parts.append(f"{categories['species']} species state(s)")
        
        return "This behavior has " + ", ".join(summary_parts) + " available for use."
    
    def _infer_state_purpose(self, metadata: StateMetadata) -> str:
        """Infer the purpose of a state based on its properties."""
        purposes = []
        
        for prop_name, prop_spec in metadata.properties.items():
            desc = prop_spec.get('description', '').lower()
            
            # Infer based on property name and description
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
        """Get the access pattern for a specific property."""
        if category == 'global':
            return f"tv.s.{state_name}.field[0].{prop_name}"
        elif category == 'particle':
            return f"tv.s.{state_name}.field[particle_idx].{prop_name}"
        elif category == 'species':
            return f"tv.s.{state_name}.field[species].{prop_name}"
        else:
            return f"tv.s.{state_name}.field[index].{prop_name}"
    
    def generate_state_access_examples(self) -> str:
        """Generate comprehensive documentation for accessing the created states."""
        examples = []
        
        # Add critical warning
        examples.append("# ⚠️⚠️⚠️ CRITICAL: STATE CATEGORY AND NAME REQUIREMENTS ⚠️⚠️⚠️")
        examples.append("# 1. YOU MUST USE THESE EXACT STATE NAMES - NO EXCEPTIONS!")
        examples.append("# 2. YOU MUST ACCESS STATES FROM THE CORRECT CATEGORY!")
        examples.append("# Using wrong state names or wrong category will cause compilation errors.")
        examples.append("#")
        examples.append("# Example of WRONG usage (compilation errors):")
        examples.append("#   activity = tv.s.llm_species.field[s].activity_level  # ❌ ERROR if 'activity_level' is a particle state")
        examples.append("#   day_cycle = tv.s.llm_particle.field[i].day_cycle     # ❌ ERROR if 'day_cycle' is a species state")
        examples.append("#   invalid_state = tv.s.llm_global.field[0].nonexistent # ❌ ERROR state doesn't exist")
        examples.append("#   position = tv.s.llm_particle.field[i].position       # ❌ ERROR 'position' not a custom state!")
        examples.append("#")
        examples.append("# Example of CORRECT usage:")
        examples.append("#   # Check the category below, then use the exact access pattern shown")
        examples.append("#   time_of_day = tv.s.llm_global.field[0].time_of_day      # ✓ If 'time_of_day' is in GLOBAL STATES")
        examples.append("#   activity = tv.s.llm_particle.field[i].activity_level    # ✓ If 'activity_level' is in PARTICLE STATES")
        examples.append("#   day_cycle = tv.s.llm_species.field[s].day_cycle         # ✓ If 'day_cycle' is in SPECIES STATES")
        examples.append("#   pos = tv.p.field[i].pos                                 # ✓ Core particle position")
        examples.append("#   vel = tv.p.field[i].vel                                 # ✓ Core particle velocity")
        examples.append("")
        examples.append("# ⚠️⚠️⚠️ CORE PARTICLE PROPERTIES (NOT CUSTOM STATES!) ⚠️⚠️⚠️")
        examples.append("# These are built-in properties accessed via tv.p.field[i]:")
        examples.append("#   pos: ti.math.vec2 - Current position (NOT 'position'!)")
        examples.append("#   vel: ti.math.vec2 - Current velocity (NOT 'velocity'!)")
        examples.append("#   species: ti.i32 - Species ID")
        examples.append("#   active: ti.f32 - Activity level (0.0-1.0)")
        examples.append("#   ppos: ti.math.vec2 - Previous position")
        examples.append("#   pvel: ti.math.vec2 - Previous velocity")
        examples.append("#   mass: ti.f32 - Particle mass")
        examples.append("#   size: ti.f32 - Particle size")
        examples.append("#   speed: ti.f32 - Particle speed limit")
        examples.append("#")
        examples.append("# NEVER access core properties as custom states:")
        examples.append("#   ❌ WRONG: tv.s.llm_particle.field[i].position")
        examples.append("#   ✓ CORRECT: tv.p.field[i].pos")
        examples.append("")
        examples.append("# AVAILABLE CUSTOM STATES FOR THIS BEHAVIOR:")
        examples.append("# These states were designed specifically for your behavior description.")
        examples.append("# PAY ATTENTION TO WHICH CATEGORY EACH STATE BELONGS TO!")
        examples.append("")
        
        # Global state access examples
        if self.state_prefixes['global'] in self.state_registry:
            examples.append("# === GLOBAL STATES (shared across entire system) ===")
            metadata = self.state_registry[self.state_prefixes['global']]
            for prop_name, prop_spec in metadata.properties.items():
                type_str = prop_spec['type']
                desc = prop_spec.get('description', '')
                range_str = f"[{prop_spec['min']}, {prop_spec['max']}]"
                examples.append(f"#")
                examples.append(f"# STATE NAME: {prop_name}")
                examples.append(f"# Type: {type_str}, Range: {range_str}")
                if desc:
                    examples.append(f"# Description: {desc}")
                examples.append(f"# ACCESS CODE:")
                examples.append(f"{prop_name} = tv.s.{self.state_prefixes['global']}.field[0].{prop_name}")
            examples.append("")
        
        # Particle state access examples
        if self.state_prefixes['particle'] in self.state_registry:
            examples.append("# === PARTICLE STATES (individual per particle) ===")
            metadata = self.state_registry[self.state_prefixes['particle']]
            for prop_name, prop_spec in metadata.properties.items():
                type_str = prop_spec['type']
                desc = prop_spec.get('description', '')
                range_str = f"[{prop_spec['min']}, {prop_spec['max']}]"
                examples.append(f"#")
                examples.append(f"# STATE NAME: {prop_name}")
                examples.append(f"# Type: {type_str}, Range: {range_str}")
                if desc:
                    examples.append(f"# Description: {desc}")
                examples.append(f"# ACCESS CODE:")
                examples.append(f"{prop_name} = tv.s.{self.state_prefixes['particle']}.field[particle_idx].{prop_name}")
            examples.append("")
        
        # Species state access examples
        if self.state_prefixes['species'] in self.state_registry:
            examples.append("# Species States (shared by all particles of same species):")
            metadata = self.state_registry[self.state_prefixes['species']]
            for prop_name, prop_spec in metadata.properties.items():
                type_str = prop_spec['type']
                desc = prop_spec.get('description', '')
                range_str = f"[{prop_spec['min']}, {prop_spec['max']}]"
                examples.append(f"# - {prop_name} ({type_str}): {desc} Range: {range_str}")
                examples.append(f"{prop_name} = tv.s.{self.state_prefixes['species']}.field[species].{prop_name}")
            examples.append("")
        
        # Add temporal information if available
        if self.temporal_config:
            examples.append("# Temporal Configuration:")
            examples.append(f"# - Day duration: {self.temporal_config.suggested_day_duration} seconds")
            examples.append(f"# - Frames per day: {self.temporal_config.frames_per_day}")
            if self.temporal_config.time_units:
                examples.append(f"# - Time units: {', '.join(self.temporal_config.time_units)}")
            examples.append("")
        
        # Add usage hints
        examples.append("# Usage hints:")
        examples.append("# - Access states early in your function to use them in calculations")
        examples.append("# - States persist across frames, enabling memory and evolution")
        examples.append("# - Combine states creatively to produce emergent behaviors")
        
        return "\n".join(examples)
    
    def clear_states(self):
        """Clear all dynamically created states."""
        # Note: We can't actually remove states from Tölvera once created,
        # but we can clear our registry and reset values
        self.state_registry.clear()
        self.temporal_config = None
        self.temporal_update_kernel = None
        logger.info("Cleared dynamic state registry")