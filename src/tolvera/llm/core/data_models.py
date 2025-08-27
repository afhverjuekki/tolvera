from __future__ import annotations
from typing import List, Dict, Optional, Literal, Union, Tuple, Any
from pydantic import BaseModel, Field, validator


class KeyValuePair(BaseModel):
    """Represents a single key-value pair for Gemini-compatible dictionary representation."""
    key: str = Field(description="The key of the entry")
    value: Any = Field(description="The value of the entry")


class VectorExpression(BaseModel):
    x: str = Field(description="X component (valid Taichi expression)")
    y: str = Field(description="Y component (valid Taichi expression)")


class StateAccess(BaseModel):
    var_name: str = Field(description="Local variable name")
    category: Literal["global", "particle", "species"]
    state_name: str = Field(description="State property name")
    index_expr: str = Field(description="Index expression (0, particle_idx, or species)")


class ConditionalForce(BaseModel):
    condition: str = Field(description="Condition expression")
    force_if_true: VectorExpression = Field(description="Force when condition is true")
    force_if_false: Optional[VectorExpression] = Field(default=None, description="Force when condition is false")


class ForceComputation(BaseModel):
    state_accesses: List[StateAccess] = Field(default_factory=list)
    helper_variables: List[KeyValuePair] = Field(
        default_factory=list,
        description="Helper variable declarations as key-value pairs"
    )
    conditional_forces: List[ConditionalForce] = Field(default_factory=list)
    base_force: VectorExpression = Field(
        default_factory=lambda: VectorExpression(x="0.0", y="0.0"),
        description="Base force before conditionals"
    )
    description: str = Field(default="", description="Description of the force computation")


class DrawingOperation(BaseModel):
    operation: Literal["set", "line", "circle", "rect"]
    args: List[str] = Field(description="Arguments for the drawing operation")
    color: str = Field(description="Color expression (ti.Vector or ti.math.vec4)")


class DrawingComputation(BaseModel):
    state_accesses: List[StateAccess] = Field(default_factory=list)
    helper_variables: List[KeyValuePair] = Field(
        default_factory=list,
        description="Helper variable declarations as key-value pairs"
    )
    conditionals: List[ConditionalForce] = Field(default_factory=list)
    drawing_operations: List[DrawingOperation] = Field(
        default_factory=list,
        description="List of drawing operations to perform"
    )


class ExpertFunction(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    is_interaction: bool = False
    is_drawing: bool = False
    draw_order: Literal["pre", "post"] = "post"
    computation: Union[ForceComputation, DrawingComputation]
    weight: float = Field(default=1.0, ge=0.1, le=2.0)
    applies_to_species: Optional[List[int]] = Field(
        default=None,
        description="Species IDs this expert applies to (None means all species)"
    )
    code: Optional[str] = Field(
        default=None,
        description="Pre-generated Taichi function code (if available)"
    )




class StateDefinition(BaseModel):
    name: str = Field(description="Name of the state variable")
    category: Literal["global", "particle", "species"] = Field(description="Category of the state")
    type: str = Field(description="Taichi type (e.g., ti.f32, ti.math.vec2)")
    min: Union[float, List[float]] = Field(description="Minimum value(s)")
    max: Union[float, List[float]] = Field(description="Maximum value(s)")
    description: str = Field(default="")
    initial: Optional[Union[float, List[float]]] = None
    temporal_update: Optional["TemporalUpdate"] = Field(
        default=None,
        description="Temporal update rule for this state"
    )
    
    @validator('min', 'max', 'initial')
    def validate_vector_bounds(cls, v, values):
        if 'type' in values and 'vec' in values['type']:
            if isinstance(v, (int, float)):
                # Convert scalar to appropriate vector
                if 'vec2' in values['type']:
                    return [v, v]
                elif 'vec3' in values['type']:
                    return [v, v, v]
                elif 'vec4' in values['type']:
                    return [v, v, v, v]
        return v


class SpeciesNameMapping(BaseModel):
    """Maps a species ID to its semantic name."""
    species_id: int = Field(description="Species ID")
    name: str = Field(description="Semantic name for the species")


class SpeciesBehaviorMapping(BaseModel):
    """Maps a species ID to its behaviors."""
    species_id: int = Field(description="Species ID")
    behaviors: List[str] = Field(description="List of behaviors for this species")


class SpeciesColorMapping(BaseModel):
    """Maps a species ID to its RGBA color."""
    species_id: int = Field(description="Species ID")
    rgba: List[float] = Field(description="RGBA color values [r, g, b, a]")


class SpeciesConfiguration(BaseModel):
    species_ids: List[int] = Field(description="List of species IDs to use")
    species_names: Optional[List[SpeciesNameMapping]] = Field(
        default=None,
        description="Mapping of species IDs to semantic names"
    )
    interaction_pairs: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Pairs of species that interact"
    )
    species_behaviors: Optional[List[SpeciesBehaviorMapping]] = Field(
        default=None,
        description="Species-specific behaviors"
    )
    colors: Optional[List[SpeciesColorMapping]] = Field(
        default=None,
        description="RGBA colors for each species"
    )
    requires_all_species: bool = Field(
        default=False,
        description="Whether the behavior applies to all species"
    )


class ExpertSpeciesCondition(BaseModel):
    """Maps an expert name to its species conditions."""
    expert_name: str = Field(description="Expert function name")
    species_ids: List[int] = Field(description="Species IDs this expert applies to")


class IntegrationKernel(BaseModel):
    single_experts: List[str] = Field(description="Names of single-particle experts")
    interaction_experts: List[str] = Field(description="Names of interaction experts")
    species_conditions: Optional[List[ExpertSpeciesCondition]] = Field(
        default=None,
        description="Species conditions for each expert"
    )


class StateUpdateMapping(BaseModel):
    """Maps a state name to its update expression."""
    state_name: str = Field(description="Name of the state to update")
    update_expression: str = Field(description="Expression for updating the state")


class TemporalUpdate(BaseModel):
    """Defines temporal state updates."""
    frame_updates: List[StateUpdateMapping] = Field(
        default_factory=list,
        description="Frame-based updates as state name to update expression mappings"
    )
    day_duration: float = Field(default=10.0, description="Day duration in seconds")
    update_expression: Optional[str] = Field(
        default=None,
        description="Expression for updating the state (e.g., 'value *= 0.99', 'value += 0.1')"
    )
    update_condition: Optional[str] = Field(
        default=None,
        description="Condition for when to apply update (e.g., 'if species == 0', 'if vel.norm() > 10')"
    )
    update_frequency: int = Field(
        default=1,
        description="Update every N frames (1 = every frame)"
    )
    affects_behavior: Optional[str] = Field(
        default=None,
        description="How state affects behavior (e.g., 'if value < 20: vel *= 0.5')"
    )
    coupling_strength: float = Field(
        default=1.0,
        description="Strength of behavioral coupling (0.0 to 1.0)"
    )


class BehaviorSynthesisRequest(BaseModel):
    description: str
    available_states: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Available states by category"
    )
    screen_size: Tuple[int, int] = (800, 600)
    species_count: int = 5
    particle_count: int = 300


class HelperFunctionMapping(BaseModel):
    """Maps a helper function name to its code."""
    function_name: str = Field(description="Name of the helper function")
    code: str = Field(description="Code implementation of the helper function")


class BehaviorSynthesisResponse(BaseModel):
    experts: List[ExpertFunction]
    states_needed: List[StateDefinition] = Field(
        default_factory=list,
        description="States to create by category"
    )
    species_config: SpeciesConfiguration
    integration_kernel: IntegrationKernel
    temporal_update: Optional[TemporalUpdate] = None
    helper_functions: Optional[List[HelperFunctionMapping]] = Field(
        default=None,
        description="Helper functions generated alongside experts"
    )
