"""
Behavior Requirements Analyzer

Analyzes complete behavior descriptions to determine all requirements upfront,
including states, kernel types, helper functions, and render sequences.
This enables holistic synthesis with shared context.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StateRequirement(BaseModel):
    """Requirement for a state field."""
    name: str = Field(description="State field name")
    category: str = Field(description="State category: global, particle, or species")
    type: str = Field(description="Taichi type (ti.f32, ti.i32, ti.math.vec2, etc.)")
    min: float = Field(default=0.0, description="Minimum value")
    max: float = Field(default=1.0, description="Maximum value")
    initial: Optional[float] = Field(default=None, description="Initial value")
    description: str = Field(default="", description="What this state represents")
    temporal_update: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Temporal update rule if this state changes over time"
    )


class PixelFieldRequirement(BaseModel):
    """Requirements for pixel field operations."""
    needs_diffusion: bool = Field(default=False, description="Requires diffusion kernel")
    needs_decay: bool = Field(default=False, description="Requires decay/evaporation")
    needs_deposition: bool = Field(default=False, description="Particles deposit to pixels")
    needs_sensing: bool = Field(default=False, description="Particles sense from pixels")
    operations: List[str] = Field(default_factory=list, description="List of pixel operations needed")


class TemporalRequirement(BaseModel):
    """Requirements for temporal/discrete updates."""
    update_type: str = Field(description="Type of temporal update (synchronous, continuous, periodic)")
    update_frequency: str = Field(default="every_frame", description="When to update (every_frame, every_n_frames)")
    states_to_update: List[str] = Field(default_factory=list, description="States that need temporal updates")
    update_logic: str = Field(default="", description="Description of update logic")


class BehaviorRequirements(BaseModel):
    """Complete requirements for a behavior."""
    # Pattern detection
    pattern_type: str = Field(default="particle_system", description="Detected pattern type")
    pattern_confidence: float = Field(default=1.0, description="Confidence in pattern detection")
    
    # States needed
    state_requirements: List[StateRequirement] = Field(default_factory=list)
    
    # System requirements
    pixel_field: Optional[PixelFieldRequirement] = None
    temporal: Optional[TemporalRequirement] = None
    
    # Required experts (not helpers - experts will generate their own helpers)
    required_experts: List[str] = Field(default_factory=list, description="Names of experts needed for this pattern")
    
    # Additional kernels needed
    additional_kernels: List[str] = Field(default_factory=list, description="Additional kernel types needed")
    
    # Render loop structure
    render_sequence: List[str] = Field(default_factory=list, description="Sequence of operations in render loop")
    
    # Implementation notes
    implementation_notes: List[str] = Field(default_factory=list, description="Notes for implementation")
    
    # Shared context
    shared_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters shared across experts")
    constraints: List[str] = Field(default_factory=list, description="Constraints to maintain")


class BehaviorRequirementsAnalyzer:
    """Analyzes behavior descriptions to determine all requirements upfront."""
    
    # Pattern indicators for different a-life patterns
    PATTERN_INDICATORS = {
        "cellular_automaton": [
            "cellular", "automaton", "grid", "cells", "neighbors", 
            "alive", "dead", "conway", "game of life", "birth", "survival"
        ],
        "physarum": [
            "slime", "trail", "pheromone", "sense", "deposit", 
            "follow", "mold", "physarum", "chemical", "scent"
        ],
        "ecosystem": [
            "predator", "prey", "food", "energy", "hunt", 
            "chase", "escape", "ecosystem", "food chain", "species interact"
        ],
        "swarm": [
            "flock", "swarm", "school", "herd", "cohesion", 
            "alignment", "separation", "boids", "murmuration"
        ],
        "growth": [
            "grow", "branch", "root", "tree", "plant", 
            "fractal", "diffusion limited", "aggregation"
        ],
        "reaction_diffusion": [
            "reaction", "diffusion", "turing", "pattern", 
            "spots", "stripes", "morphogenesis"
        ],
        "ant_colony": [
            "ant", "colony", "foraging", "nest", "food source",
            "recruitment", "trail following"
        ],
        "neural": [
            "neural", "neuron", "synapse", "spike", "activation",
            "network", "brain", "signal propagation"
        ]
    }
    
    # State requirements for common patterns
    PATTERN_STATE_REQUIREMENTS = {
        "cellular_automaton": [
            StateRequirement(name="grid_x", category="particle", type="ti.i32", min=0, max=100),
            StateRequirement(name="grid_y", category="particle", type="ti.i32", min=0, max=100),
            StateRequirement(name="is_alive", category="particle", type="ti.i32", min=0, max=1),
            StateRequirement(name="neighbor_count", category="particle", type="ti.i32", min=0, max=8),
            StateRequirement(name="next_state", category="particle", type="ti.i32", min=0, max=1),
            StateRequirement(name="generation_count", category="global", type="ti.i32", min=0, max=10000)
        ],
        "physarum": [
            StateRequirement(name="heading", category="particle", type="ti.f32", min=0, max=6.28319),
            StateRequirement(name="sensor_angle", category="particle", type="ti.f32", min=0, max=1.57),
            StateRequirement(name="sensor_distance", category="particle", type="ti.f32", min=5, max=50),
            StateRequirement(name="turn_speed", category="particle", type="ti.f32", min=0, max=3),
            StateRequirement(name="move_speed", category="particle", type="ti.f32", min=10, max=100),
            StateRequirement(name="deposit_amount", category="global", type="ti.f32", min=0.1, max=1.0),
            StateRequirement(name="evaporation_rate", category="global", type="ti.f32", min=0.95, max=0.999),
            StateRequirement(name="diffusion_rate", category="global", type="ti.f32", min=0, max=0.1)
        ],
        "ecosystem": [
            StateRequirement(name="energy", category="particle", type="ti.f32", min=0, max=100, initial=80),
            StateRequirement(name="max_energy", category="species", type="ti.f32", min=50, max=200),
            StateRequirement(name="hunt_radius", category="species", type="ti.f32", min=50, max=200),
            StateRequirement(name="escape_radius", category="species", type="ti.f32", min=50, max=200),
            StateRequirement(name="energy_decay_rate", category="species", type="ti.f32", min=0.1, max=2.0)
        ]
    }
    
    # Required experts for patterns (experts will generate their own helpers)
    PATTERN_EXPERTS = {
        "cellular_automaton": ["grid_state_update", "neighbor_counter", "rule_applier"],
        "physarum": ["trail_sensor", "trail_depositor", "movement_controller"],
        "ecosystem": ["predator_hunt", "prey_escape", "energy_transfer"],
        "swarm": ["separation_force", "alignment_force", "cohesion_force"],
        "ant_colony": ["pheromone_sensor", "pheromone_depositor", "path_follower"]
    }
    
    # Render sequences for patterns
    PATTERN_RENDER_SEQUENCES = {
        "cellular_automaton": ["clear", "update_temporal_states", "apply_experts", "render_particles"],
        "physarum": ["decay_pheromones", "diffuse_pheromones", "apply_experts", "deposit_trails", "render_particles"],
        "ecosystem": ["update_resources", "apply_experts", "handle_interactions", "update_energy", "render_particles"],
        "swarm": ["apply_experts", "render_particles"],
        "reaction_diffusion": ["diffuse_chemicals", "react_chemicals", "apply_experts", "render_particles"]
    }
    
    def __init__(self):
        """Initialize the behavior requirements analyzer."""
        logger.info("Initialized BehaviorRequirementsAnalyzer")
    
    def analyze(self, description: str, decomposed_behavior: Optional[Any] = None) -> BehaviorRequirements:
        """
        Analyze a behavior description to determine all requirements.
        
        Args:
            description: Natural language behavior description
            decomposed_behavior: Optional decomposed behavior from decomposer
            
        Returns:
            BehaviorRequirements with all detected requirements
        """
        logger.info(f"Analyzing requirements for: {description[:100]}...")
        
        # Detect pattern type
        pattern_type, confidence = self._detect_pattern_type(description)
        logger.info(f"Detected pattern: {pattern_type} (confidence: {confidence:.2f})")
        
        # Initialize requirements
        requirements = BehaviorRequirements(
            pattern_type=pattern_type,
            pattern_confidence=confidence
        )
        
        # Store pattern-specific requirements as hints, not automatic additions
        # The LLM will determine which are actually needed based on the specific behavior
        if pattern_type in self.PATTERN_STATE_REQUIREMENTS:
            # Store as hints for the synthesizer to consider
            requirements.implementation_notes.append(
                f"Pattern '{pattern_type}' typically uses states: " + 
                ", ".join([s.name for s in self.PATTERN_STATE_REQUIREMENTS[pattern_type]])
            )
            logger.info(f"Added state hints for {pattern_type} pattern")
        
        # Add required experts
        if pattern_type in self.PATTERN_EXPERTS:
            requirements.required_experts.extend(self.PATTERN_EXPERTS[pattern_type])
            logger.info(f"Added {len(self.PATTERN_EXPERTS[pattern_type])} required experts for {pattern_type}")
        
        # Add render sequence
        if pattern_type in self.PATTERN_RENDER_SEQUENCES:
            requirements.render_sequence.extend(self.PATTERN_RENDER_SEQUENCES[pattern_type])
            logger.info(f"Added render sequence with {len(self.PATTERN_RENDER_SEQUENCES[pattern_type])} steps")
        
        # Analyze for pixel field requirements
        pixel_req = self._analyze_pixel_requirements(description, pattern_type)
        if pixel_req:
            requirements.pixel_field = pixel_req
            logger.info("Added pixel field requirements")
        
        # Analyze for temporal requirements
        temporal_req = self._analyze_temporal_requirements(description, pattern_type)
        if temporal_req:
            requirements.temporal = temporal_req
            requirements.additional_kernels.append("update_temporal_states")
            logger.info("Added temporal requirements")
        
        # Extract requirements from decomposed behavior if provided
        if decomposed_behavior:
            self._extract_decomposed_requirements(decomposed_behavior, requirements)
        
        # Analyze for potential state requirements from description
        # These are hints for the LLM, not automatic requirements
        state_hints = self._analyze_state_hints(description)
        if state_hints:
            requirements.implementation_notes.append(
                "Based on description keywords, consider these states: " + 
                ", ".join([s.name for s in state_hints])
            )
        
        # Add implementation notes based on pattern
        requirements.implementation_notes = self._get_implementation_notes(pattern_type, description)
        
        return requirements
    
    def _detect_pattern_type(self, description: str) -> Tuple[str, float]:
        """Detect the pattern type from description."""
        desc_lower = description.lower()
        
        # Score each pattern
        pattern_scores = {}
        for pattern, indicators in self.PATTERN_INDICATORS.items():
            score = sum(1 for indicator in indicators if indicator in desc_lower)
            if score > 0:
                pattern_scores[pattern] = score / len(indicators)
        
        # Get highest scoring pattern
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            confidence = pattern_scores[best_pattern]
            return best_pattern, confidence
        
        return "particle_system", 0.5
    
    def _analyze_pixel_requirements(self, description: str, pattern_type: str) -> Optional[PixelFieldRequirement]:
        """Analyze if pixel field operations are needed."""
        desc_lower = description.lower()
        
        # Check for pixel-related keywords
        pixel_keywords = {
            "diffusion": ["diffuse", "spread", "propagate"],
            "decay": ["decay", "evaporate", "fade", "dissipate"],
            "deposition": ["deposit", "leave", "drop", "place", "trail"],
            "sensing": ["sense", "detect", "follow", "smell", "see"]
        }
        
        pixel_req = PixelFieldRequirement()
        needs_pixel = False
        
        for op_type, keywords in pixel_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                setattr(pixel_req, f"needs_{op_type}", True)
                pixel_req.operations.append(op_type)
                needs_pixel = True
        
        # Pattern-specific pixel requirements
        if pattern_type == "physarum":
            pixel_req.needs_diffusion = True
            pixel_req.needs_decay = True
            pixel_req.needs_deposition = True
            pixel_req.needs_sensing = True
            pixel_req.operations = ["diffusion", "decay", "deposition", "sensing"]
            needs_pixel = True
        elif pattern_type == "ant_colony":
            pixel_req.needs_decay = True
            pixel_req.needs_deposition = True
            pixel_req.needs_sensing = True
            pixel_req.operations = ["decay", "deposition", "sensing"]
            needs_pixel = True
        
        return pixel_req if needs_pixel else None
    
    def _analyze_temporal_requirements(self, description: str, pattern_type: str) -> Optional[TemporalRequirement]:
        """Analyze if temporal/discrete updates are needed."""
        desc_lower = description.lower()
        
        # Extended temporal keywords
        temporal_keywords = [
            "over time", "gradually", "slowly", "cycle", "generation",
            "step", "update", "synchronous", "discrete", "turn",
            "day", "night", "season", "periodic", "rhythm",
            "depletes", "regenerates", "ages", "grows", "decays",
            "tired", "exhausted", "hungry", "loses energy", "gains energy"
        ]
        
        if any(kw in desc_lower for kw in temporal_keywords):
            temporal_req = TemporalRequirement(
                update_type="continuous" if "gradually" in desc_lower else "synchronous"
            )
            
            # Pattern-specific temporal requirements
            if pattern_type == "cellular_automaton":
                temporal_req.update_type = "synchronous"
                temporal_req.states_to_update = ["is_alive", "neighbor_count", "next_state"]
                temporal_req.update_logic = "Count neighbors, apply rules, commit state"
            elif any(phrase in desc_lower for phrase in ["energy", "tired", "exhausted", "hungry"]):
                temporal_req.states_to_update = ["energy"]
                temporal_req.update_logic = "Decay energy over time based on activity"
            elif "day" in desc_lower or "night" in desc_lower:
                temporal_req.states_to_update = ["day_phase", "time_of_day"]
                temporal_req.update_logic = "Cycle through day/night phases"
            elif "ages" in desc_lower or "grows" in desc_lower:
                temporal_req.states_to_update = ["age", "size"]
                temporal_req.update_logic = "Age progression and growth"
            elif "phase" in desc_lower or "oscillate" in desc_lower:
                temporal_req.states_to_update = ["phase", "frequency"]
                temporal_req.update_logic = "Oscillator phase updates"
            
            return temporal_req
        
        # Pattern-specific temporal needs
        if pattern_type in ["cellular_automaton", "reaction_diffusion", "neural"]:
            return TemporalRequirement(
                update_type="synchronous",
                states_to_update=["grid_state", "next_state"],
                update_logic="Synchronous state updates"
            )
        elif pattern_type == "ecosystem":
            # Ecosystems often have implicit energy dynamics
            return TemporalRequirement(
                update_type="continuous",
                states_to_update=["energy"],
                update_logic="Energy depletion and regeneration"
            )
        
        return None
    
    def _analyze_state_hints(self, description: str) -> List[StateRequirement]:
        """Analyze for additional state requirements from description."""
        desc_lower = description.lower()
        additional_states = []
        
        # Check for common state patterns with temporal updates
        state_patterns = {
            "energy": {
                "name": "energy",
                "type": "ti.f32",
                "min": 0.0,
                "max": 100.0,
                "initial": 80.0,
                "temporal": self._get_energy_temporal(desc_lower)
            },
            "health": {
                "name": "health",
                "type": "ti.f32",
                "min": 0.0,
                "max": 100.0,
                "initial": 100.0,
                "temporal": None
            },
            "age": {
                "name": "age",
                "type": "ti.f32",
                "min": 0.0,
                "max": 1000.0,
                "initial": 0.0,
                "temporal": {
                    "update_expression": "age += 1",
                    "update_frequency": 1,
                    "affects_behavior": "if age > 500: active = 0.0"
                } if "ages" in desc_lower or "life cycle" in desc_lower else None
            },
            "consumed": {
                "name": "consumed",
                "type": "ti.i32",
                "min": 0,
                "max": 1,
                "initial": 0,
                "temporal": None  # Consumption handled by behavior experts
            },
            "charge": {
                "name": "charge",
                "type": "ti.f32",
                "min": -1.0,
                "max": 1.0,
                "initial": 0.0,
                "temporal": None
            },
            "temperature": {
                "name": "temperature",
                "type": "ti.f32",
                "min": 0.0,
                "max": 100.0,
                "initial": 20.0,
                "temporal": {
                    "update_expression": "temperature = temperature * 0.99 + ambient_temp * 0.01",
                    "update_frequency": 1
                }
            },
            "phase": {
                "name": "phase",
                "type": "ti.f32",
                "min": 0.0,
                "max": 6.28319,
                "initial": 0.0,
                "temporal": {
                    "update_expression": "phase = (phase + frequency * 0.1) % 6.28319",
                    "update_frequency": 1
                }
            },
            "home": {
                "name": "home_pos",
                "type": "ti.math.vec2",
                "min": 0.0,
                "max": 1920.0,
                "initial": None,
                "temporal": None
            },
            "target": {
                "name": "target_pos",
                "type": "ti.math.vec2",
                "min": 0.0,
                "max": 1920.0,
                "initial": None,
                "temporal": None
            },
            "memory": {
                "name": "memory_pos",
                "type": "ti.math.vec2",
                "min": 0.0,
                "max": 1920.0,
                "initial": None,
                "temporal": None
            }
        }
        
        for keyword, config in state_patterns.items():
            if keyword in desc_lower and not any(s.name == config["name"] for s in additional_states):
                category = "particle"
                additional_states.append(StateRequirement(
                    name=config["name"],
                    category=category,
                    type=config["type"],
                    min=config["min"],
                    max=config["max"],
                    initial=config["initial"],
                    description=f"State for {keyword}",
                    temporal_update=config["temporal"]
                ))
        
        # Check for food/consumption patterns
        food_keywords = ["food", "eat", "consume", "feed", "resource", "prey"]
        if any(kw in desc_lower for kw in food_keywords):
            if not any(s.name == "consumed" for s in additional_states):
                additional_states.append(StateRequirement(
                    name="consumed",
                    category="particle",
                    type="ti.i32",
                    min=0,
                    max=1,
                    initial=0,
                    description="Whether particle has been consumed",
                    temporal_update=None
                ))
        
        # Check for temporal-specific patterns
        if "loses energy over time" in desc_lower or "energy depletes" in desc_lower:
            if not any(s.name == "energy" for s in additional_states):
                additional_states.append(StateRequirement(
                    name="energy",
                    category="particle",
                    type="ti.f32",
                    min=0.0,
                    max=100.0,
                    initial=80.0,
                    description="Energy that depletes over time",
                    temporal_update={
                        "update_expression": "energy *= 0.995",
                        "update_frequency": 1,
                        "affects_behavior": "if energy < 20: vel *= 0.8"
                    }
                ))
        
        return additional_states
    
    def _get_energy_temporal(self, desc_lower: str) -> Optional[Dict[str, Any]]:
        """Determine energy temporal update based on description."""
        # Only add energy decay if explicitly mentioned
        if "loses energy" in desc_lower or "energy depletes" in desc_lower or "energy decreases" in desc_lower:
            return {
                "update_expression": "energy -= vel.norm() * 0.01",  # Only deplete when moving
                "update_frequency": 1,
                "affects_behavior": "if energy < 20: vel *= 0.8",
                "update_condition": "if vel.norm() > 0.1"  # Only deplete if actually moving
            }
        elif "gets tired" in desc_lower or "becomes exhausted" in desc_lower or "fatigue" in desc_lower:
            return {
                "update_expression": "energy -= vel.norm() * 0.02",  # Activity-based fatigue
                "update_frequency": 1,
                "affects_behavior": "if energy < 30: vel *= 0.7",
                "update_condition": "if vel.norm() > 0.5"  # Only if moving significantly
            }
        # DO NOT add automatic energy decay for general mentions
        return None
    
    def _extract_decomposed_requirements(self, decomposed_behavior: Any, requirements: BehaviorRequirements) -> None:
        """Extract requirements from decomposed behavior."""
        if hasattr(decomposed_behavior, 'suggested_states'):
            for state_tuple in decomposed_behavior.suggested_states:
                # Handle new 5-tuple format: (name, category, type, min, max)
                if len(state_tuple) >= 5:
                    state_name, category, type_str, min_val, max_val = state_tuple[:5]
                elif len(state_tuple) == 2:
                    # Backward compatibility with old 2-tuple format
                    state_name, category = state_tuple
                    type_str = "ti.f32"
                    min_val = 0.0
                    max_val = 1.0
                else:
                    continue  # Skip invalid tuples
                    
                if not any(s.name == state_name for s in requirements.state_requirements):
                    requirements.state_requirements.append(StateRequirement(
                        name=state_name,
                        category=category,
                        type=type_str,
                        min=min_val,
                        max=max_val,
                        description=f"State from decomposition: {state_name}"
                    ))
        
        if hasattr(decomposed_behavior, 'context') and decomposed_behavior.context:
            if hasattr(decomposed_behavior.context, 'constraints'):
                requirements.constraints.extend(decomposed_behavior.context.constraints)
            if hasattr(decomposed_behavior.context, 'shared_parameters'):
                for param in decomposed_behavior.context.shared_parameters:
                    # Parse parameter strings like "detection_distance: 100.0"
                    if ':' in param:
                        key, value = param.split(':', 1)
                        requirements.shared_parameters[key.strip()] = value.strip()
    
    def _get_implementation_notes(self, pattern_type: str, description: str) -> List[str]:
        """Get implementation notes based on pattern and description."""
        notes = []
        
        if pattern_type == "cellular_automaton":
            notes.append("Use fixed grid positions, particles don't move")
            notes.append("Implement synchronous updates with double buffering")
            notes.append("Color particles based on alive/dead state")
        elif pattern_type == "physarum":
            notes.append("Use pixel field for pheromone trails")
            notes.append("Implement sensor-based turning behavior")
            notes.append("Apply diffusion and evaporation to trails")
        elif pattern_type == "ecosystem":
            notes.append("Track energy transfer between species")
            notes.append("Implement species-specific behaviors")
            notes.append("Handle birth/death based on energy levels")
        elif pattern_type == "swarm":
            notes.append("Balance cohesion, separation, and alignment forces")
            notes.append("Use neighbor detection within radius")
            notes.append("Apply smooth steering behaviors")
        
        return notes