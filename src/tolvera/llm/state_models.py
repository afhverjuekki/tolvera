from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum


class TaichiType(str, Enum):
    F32 = "ti.f32"
    F64 = "ti.f64"
    I32 = "ti.i32"
    I64 = "ti.i64"
    U32 = "ti.u32"
    U64 = "ti.u64"
    VEC2 = "ti.math.vec2"
    VEC3 = "ti.math.vec3"
    VEC4 = "ti.math.vec4"


class StateProperty(BaseModel):
    type: TaichiType = Field(description="Taichi type for this state property")
    min: float | List[float] = Field(description="Minimum value(s) for this property")
    max: float | List[float] = Field(description="Maximum value(s) for this property")
    initial: Optional[float | List[float]] = Field(None, description="Optional initial value (defaults to min)")
    description: str = Field(description="Human-readable description of what this state represents")


class TemporalConfig(BaseModel):
    day_duration: float = Field(default=10.0, description="Duration of one day/night cycle in seconds")
    time_units: List[str] = Field(default_factory=list, description="Time units used (e.g., ['day', 'night', 'dawn', 'dusk'])")
    frame_rate: float = Field(default=60.0, description="Expected frame rate")
    time_scale: float = Field(default=1.0, description="Time scaling factor")


class StateAnalysisResult(BaseModel):
    global_states: Dict[str, StateProperty] = Field(
        default_factory=dict, 
        description="Global states shared across the entire system"
    )
    particle_states: Dict[str, StateProperty] = Field(
        default_factory=dict, 
        description="Per-particle states (individual properties for each particle)"
    )
    species_states: Dict[str, StateProperty] = Field(
        default_factory=dict, 
        description="Per-species states (shared by particles of the same species)"
    )
    pixel_states: Dict[str, StateProperty] = Field(
        default_factory=dict, 
        description="Per-pixel states for pixel-based simulations"
    )
    temporal_config: Optional[TemporalConfig] = Field(
        None, 
        description="Temporal configuration if behavior involves time-based changes"
    )

    def has_any_states(self) -> bool:
        """Check if any states are defined."""
        return bool(self.global_states or self.particle_states or 
                   self.species_states or self.pixel_states)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with existing code."""
        result = {
            'global_states': {},
            'particle_states': {},
            'species_states': {},
            'pixel_states': {},
            'temporal_config': None
        }
        
        for state_name, prop in self.global_states.items():
            result['global_states'][state_name] = {
                'type': prop.type.value,
                'min': prop.min,
                'max': prop.max,
                'description': prop.description
            }
            if prop.initial is not None:
                result['global_states'][state_name]['initial'] = prop.initial
        
        for state_name, prop in self.particle_states.items():
            result['particle_states'][state_name] = {
                'type': prop.type.value,
                'min': prop.min,
                'max': prop.max,
                'description': prop.description
            }
            if prop.initial is not None:
                result['particle_states'][state_name]['initial'] = prop.initial
        
        for state_name, prop in self.species_states.items():
            result['species_states'][state_name] = {
                'type': prop.type.value,
                'min': prop.min,
                'max': prop.max,
                'description': prop.description
            }
            if prop.initial is not None:
                result['species_states'][state_name]['initial'] = prop.initial
        
        for state_name, prop in self.pixel_states.items():
            result['pixel_states'][state_name] = {
                'type': prop.type.value,
                'min': prop.min,
                'max': prop.max,
                'description': prop.description
            }
            if prop.initial is not None:
                result['pixel_states'][state_name]['initial'] = prop.initial
        
        if self.temporal_config:
            result['temporal_config'] = {
                'day_duration': self.temporal_config.day_duration,
                'time_units': self.temporal_config.time_units,
                'frame_rate': self.temporal_config.frame_rate,
                'time_scale': self.temporal_config.time_scale
            }
        
        return result


class BehaviorType(str, Enum):
    SINGLE = "SINGLE"
    INTERACTION = "INTERACTION"


class BehaviorClassification(BaseModel):
    behavior_type: BehaviorType = Field(description="Type of behavior (SINGLE or INTERACTION)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level of classification")
    reasoning: str = Field(description="Explanation for the classification")


class SubBehavior(BaseModel):
    description: str = Field(description="Natural language description of this sub-behavior")
    weight: float = Field(ge=0.1, le=2.0, description="Weight/importance of this sub-behavior")
    relationship: str = Field(description="Relationship to other sub-behaviors (e.g., 'primary', 'secondary')")


class BehaviorDecomposition(BaseModel):
    should_decompose: bool = Field(description="Whether this behavior should be decomposed")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Complexity score (0=simple, 1=very complex)")
    reasoning: str = Field(description="Explanation for decomposition decision")
    sub_behaviors: List[SubBehavior] = Field(default_factory=list, description="List of sub-behaviors if decomposed")