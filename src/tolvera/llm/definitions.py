from pydantic import BaseModel, Field, validator, field_validator, root_validator, conint, confloat
from typing import List, Optional, Literal
from enum import Enum

class ParticleShape(str, Enum):
    CIRCLE = "circle"
    POINT = "point"
    LINE = "line"
    RECT = "rect"

class BackgroundBehavior(str, Enum):
    DIFFUSE = "diffuse"
    CLEAR = "clear"
    NONE = "none"

class SpeciesConfig(BaseModel):
    species_index: conint(ge=0)
    color: Optional[str] = "random"
    
    # From slime.py
    slime_sense_angle: Optional[confloat(ge=0.0, le=3.14)] = None
    slime_sense_dist: Optional[confloat(ge=1.0)] = None
    slime_move_angle: Optional[confloat(ge=0.0, le=3.14)] = None
    slime_move_step: Optional[confloat(ge=0.1)] = None

class RenderConfig(BaseModel):
    background_behavior: BackgroundBehavior = BackgroundBehavior.DIFFUSE
    diffuse_rate: Optional[confloat(ge=0.0, le=1.0)] = None
    particle_shape: ParticleShape = ParticleShape.CIRCLE

    @root_validator(skip_on_failure=True)
    def check_background_dependencies(cls, values):
        if values.get('background_behavior') == BackgroundBehavior.DIFFUSE and values.get('diffuse_rate') is None:
            values['diffuse_rate'] = 0.95
        return values

class FlockRuleConfig(BaseModel):
    species_from: conint(ge=0)
    species_to: conint(ge=0)
    separate: Optional[confloat(ge=0.0, le=1.0)] = None
    align: Optional[confloat(ge=0.0, le=1.0)] = None
    cohere: Optional[confloat(ge=0.0, le=1.0)] = None
    radius: Optional[confloat(ge=0.01, le=1.0)] = None

class FlockVeraConfig(BaseModel):
    module_name: Literal["flock"] = "flock"
    rules: Optional[List[FlockRuleConfig]] = None


class SlimeVeraConfig(BaseModel):
    module_name: Literal["slime"] = "slime"
    evaporate_rate: confloat(ge=0.0, le=1.0) = 0.95
    trail_brightness: confloat(ge=0.0) = 1.0

class SketchConfig(BaseModel):
    sketch_name: str = Field(..., min_length=1)
    sketch_type: Literal["flock", "slime"] = ...
    particle_count: conint(ge=10) = 1024
    species_count: conint(ge=1) = 3
    
    window_width: conint(gt=0) = 1920
    window_height: conint(gt=0) = 1080
    global_speed: confloat(gt=0.0) = 1.0
    num_substeps: conint(ge=1) = 1
    
    render_config: RenderConfig
    flock_config: Optional[FlockVeraConfig] = None
    slime_config: Optional[SlimeVeraConfig] = None
    species_configs: Optional[List[SpeciesConfig]] = None

    # --- Validations ---
    @field_validator('render_config', mode='before')
    def ensure_render_config(cls, v):
        return RenderConfig() if v is None else v

    @validator('species_configs', always=True)
    def populate_species_configs(cls, v, values):
        species_count = values.get('species_count', 3)
        
        if v is None:
            return [SpeciesConfig(species_index=i) for i in range(species_count)]
            
        config_dict = {}
        indices = set()
        
        for item in v:
            if isinstance(item, dict):
                item = SpeciesConfig(**item)
                
            idx = item.species_index
            
            if not 0 <= idx < species_count: # Validate index for species info
                raise ValueError(f"species_index {idx} out of bounds")
            if idx in indices:
                raise ValueError(f"Duplicate species_index {idx}")
                
            config_dict[idx] = item
            indices.add(idx)
            
        return [config_dict.get(i, SpeciesConfig(species_index=i)) for i in range(species_count)]

    @validator('flock_config', always=True)
    def validate_flock_rules(cls, v, values):
        if not v or not v.rules:
            return v
            
        species_count = values.get('species_count', 3)
        validated_rules = []
        
        for rule in v.rules:
            if isinstance(rule, dict): 
                rule = FlockRuleConfig(**rule)
                
            if not 0 <= rule.species_from < species_count: # Indices for species are tricky for the LLM, so we validate it here
                raise ValueError(f"species_from {rule.species_from} out of bounds")
            if not 0 <= rule.species_to < species_count:
                raise ValueError(f"species_to {rule.species_to} out of bounds")
                
            validated_rules.append(rule)
            
        v.rules = validated_rules
        return v

    @root_validator(skip_on_failure=True)
    def ensure_config_consistency(cls, values):
        sketch_type = values.get('sketch_type')
        
        if sketch_type == "flock":
            if values.get('flock_config') is None:
                values['flock_config'] = FlockVeraConfig()
            values['slime_config'] = None
        elif sketch_type == "slime":
            if values.get('slime_config') is None:
                values['slime_config'] = SlimeVeraConfig()
            values['flock_config'] = None
            
        return values