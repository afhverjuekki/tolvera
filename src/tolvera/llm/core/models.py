from typing import List, Dict, Optional, Literal, Union, Tuple
from pydantic import BaseModel, Field, validator


class VectorExpression(BaseModel):
    x: str = Field(description="X component (valid Taichi expression)")
    y: str = Field(description="Y component (valid Taichi expression)")
    
    def to_code(self) -> str:
        return f"ti.math.vec2({self.x}, {self.y})"


class StateAccess(BaseModel):
    var_name: str = Field(description="Local variable name")
    category: Literal["global", "particle", "species"]
    state_name: str = Field(description="State property name")
    index_expr: str = Field(description="Index expression (0, particle_idx, or species)")
    
    def to_code(self) -> str:
        return f"{self.var_name} = tv.s.llm_{self.category}.field[{self.index_expr}].{self.state_name}"


class ConditionalForce(BaseModel):
    condition: str = Field(description="Condition expression")
    force_if_true: VectorExpression = Field(description="Force when condition is true")
    force_if_false: Optional[VectorExpression] = Field(default=None, description="Force when condition is false")
    
    def to_code(self) -> str:
        if self.force_if_false:
            return f"({self.force_if_true.to_code()} if {self.condition} else {self.force_if_false.to_code()})"
        else:
            return f"({self.force_if_true.to_code()} if {self.condition} else ti.math.vec2(0.0, 0.0))"


class ForceComputation(BaseModel):
    state_accesses: List[StateAccess] = Field(default_factory=list)
    helper_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Helper variable declarations (name -> expression)"
    )
    conditional_forces: List[ConditionalForce] = Field(default_factory=list)
    base_force: VectorExpression = Field(
        default_factory=lambda: VectorExpression(x="0.0", y="0.0"),
        description="Base force before conditionals"
    )
    
    def to_code(self, indent: int = 1) -> str:
        ind = "    " * indent
        lines = []
        
        # State accesses
        for state in self.state_accesses:
            lines.append(state.to_code())
        
        # Helper variables
        for var, expr in self.helper_variables.items():
            lines.append(f"{var} = {expr}")
        
        # Start with base force
        lines.append(f"force = {self.base_force.to_code()}")
        
        # Add conditional forces
        for cond_force in self.conditional_forces:
            lines.append(f"force += {cond_force.to_code()}")
        
        return f"\n{ind}".join(lines)


class DrawingOperation(BaseModel):
    operation: Literal["set", "line", "circle", "rect"]
    args: List[str] = Field(description="Arguments for the drawing operation")
    color: str = Field(description="Color expression (ti.Vector or ti.math.vec4)")
    
    def to_code(self) -> str:
        if self.operation == "set":
            return f"px.set({', '.join(self.args)}, {self.color})"
        elif self.operation == "line":
            return f"px.line({', '.join(self.args)}, {self.color})"
        elif self.operation == "circle":
            return f"px.circle({', '.join(self.args)}, {self.color})"
        elif self.operation == "rect":
            return f"px.rect({', '.join(self.args)}, {self.color})"


class DrawingComputation(BaseModel):
    state_accesses: List[StateAccess] = Field(default_factory=list)
    helper_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Helper variable declarations"
    )
    conditionals: List[ConditionalForce] = Field(default_factory=list)
    drawing_operations: List[DrawingOperation] = Field(
        default_factory=list,
        description="List of drawing operations to perform"
    )
    
    def to_code(self, indent: int = 1) -> str:
        ind = "    " * indent
        lines = []
        
        # State accesses
        for state in self.state_accesses:
            lines.append(state.to_code())
        
        # Helper variables
        for var, expr in self.helper_variables.items():
            lines.append(f"{var} = {expr}")
        
        # Drawing operations
        for op in self.drawing_operations:
            lines.append(op.to_code())
        
        # Conditionals (if any)
        for cond in self.conditionals:
            cond_lines = cond.to_code().split('\n')
            lines.extend(cond_lines)
        
        return f"\n{ind}".join(lines)


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
    
    def to_code(self) -> str:
        if self.is_drawing:
            # Drawing expert function
            if self.is_interaction:
                params = "px: ti.template(), p1: ti.template(), p2: ti.template()"
                param_extraction = ""
            else:
                params = "px: ti.template(), p: ti.template(), i: ti.i32"
                param_extraction = ""
            
            computation_code = self.computation.to_code()
            
            return f'''@ti.func
def expert_{self.name}({params}):
    # Extract particle properties
    pos = p.pos if not self.is_interaction else p1.pos
    vel = p.vel if not self.is_interaction else p1.vel
    mass = p.mass if not self.is_interaction else p1.mass
    species = p.species if not self.is_interaction else p1.species
    
    {computation_code}'''
        else:
            # Force expert function (existing logic)
            if self.is_interaction:
                params = "p1: ti.template(), p2: ti.template()"
                param_extraction = """
    # Extract particle properties
    pos = p1.pos
    vel = p1.vel  
    mass = p1.mass
    species = p1.species"""
            else:
                params = "pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32"
                param_extraction = ""
            
            computation_code = self.computation.to_code()
            
            return f'''@ti.func
def expert_{self.name}({params}) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0){param_extraction}
    
    {computation_code}
    
    return force'''


class TemporalUpdate(BaseModel):
    """Defines how a state changes over time."""
    update_expression: str = Field(
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


class StateDefinition(BaseModel):
    name: str = Field(description="Name of the state variable")
    category: Literal["global", "particle", "species"] = Field(description="Category of the state")
    type: str = Field(description="Taichi type (e.g., ti.f32, ti.math.vec2)")
    min: Union[float, List[float]] = Field(description="Minimum value(s)")
    max: Union[float, List[float]] = Field(description="Maximum value(s)")
    description: str = Field(default="")
    initial: Optional[Union[float, List[float]]] = None
    temporal_update: Optional[TemporalUpdate] = Field(
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


class SpeciesConfiguration(BaseModel):
    species_ids: List[int] = Field(description="List of species IDs to use")
    species_names: Optional[Dict[int, str]] = Field(
        default=None,
        description="Mapping of species IDs to semantic names"
    )
    interaction_pairs: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Pairs of species that interact"
    )
    species_behaviors: Optional[Dict[int, List[str]]] = Field(
        default=None,
        description="Species-specific behaviors"
    )
    colors: Optional[Dict[int, List[float]]] = Field(
        default=None,
        description="RGBA colors for each species"
    )
    requires_all_species: bool = Field(
        default=False,
        description="Whether the behavior applies to all species"
    )
    
    def get_init_code(self, particle_count: int, screen_size: Tuple[int, int]) -> str:
        num_species = len(self.species_ids)
        
        # Import ColorResolver for default colors
        from ..core.color_resolver import ColorResolver
        color_resolver = ColorResolver()
        default_colors_dict = color_resolver.get_default_species_colors(max(8, num_species))
        
        # Convert to list format for backward compatibility
        default_colors = [default_colors_dict[i] for i in range(max(8, num_species))]
        
        # Build initialization code
        init_code = ""
        
        # Add species mapping if multiple species
        if num_species > 1:
            init_code += f"# Species mapping\nspecies_map = ti.field(dtype=ti.i32, shape={num_species})\n"
            for i, sid in enumerate(self.species_ids):
                init_code += f"species_map[{i}] = {sid}\n"
            init_code += "\n"
        
        # Add species comments if names are available
        if self.species_names:
            init_code += "# Species configuration:\n"
            for sid, name in self.species_names.items():
                init_code += f"# Species {sid}: {name}\n"
            init_code += "\n"
        
        init_code += f"""@ti.kernel
def init_particles():
    for i in range(tv.pn):
        tv.p.field[i].active = 1.0
        tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
        tv.p.field[i].vel = ti.Vector([
            (ti.random() - 0.5) * 100.0,
            (ti.random() - 0.5) * 100.0
        ])
        tv.p.field[i].size = 5.0
        tv.p.field[i].mass = 1.0
        # Assign species cyclically
        tv.p.field[i].species = {self.species_ids[0] if num_species == 1 else f"species_map[i % {num_species}]"}

init_particles()
"""
        
        # Set colors with semantic awareness
        color_code = "\n# Set species colors\n"
        for idx, species_id in enumerate(self.species_ids):
            if self.colors and species_id in self.colors:
                color = self.colors[species_id]
            elif idx < len(default_colors):
                color = default_colors[idx]
            else:
                # Generate distinct color using golden ratio
                import colorsys
                hue = (idx * 0.618033988749895) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                color = [rgb[0], rgb[1], rgb[2], 1.0]
            color_code += f"tv.s.species.field[{species_id}].rgba = {color}\n"
        
        return init_code + color_code


class IntegrationKernel(BaseModel):
    single_experts: List[str] = Field(description="Names of single-particle experts")
    interaction_experts: List[str] = Field(description="Names of interaction experts")
    species_conditions: Optional[Dict[str, List[int]]] = Field(
        default=None,
        description="Species conditions for each expert"
    )
    
    def to_code(self) -> str:
        single_calls = []
        for expert_name in self.single_experts:
            if self.species_conditions and expert_name in self.species_conditions:
                species_list = self.species_conditions[expert_name]
                conditions = " or ".join([f"species == {s}" for s in species_list])
                single_calls.append(
                    f"        if {conditions}:\n"
                    f"            total_force += expert_{expert_name}(pos, vel, mass, species, i) * 1.0"
                )
            else:
                single_calls.append(
                    f"        total_force += expert_{expert_name}(pos, vel, mass, species, i) * 1.0"
                )
        
        single_calls_str = "\n".join(single_calls) if single_calls else "        # No single-particle experts"
        
        interaction_calls = []
        for expert_name in self.interaction_experts:
            interaction_calls.append(
                f"                total_force += expert_{expert_name}(p1, p2) * 1.0"
            )
        
        interaction_calls_str = "\n".join(interaction_calls) if interaction_calls else "                # No interaction experts"
        
        kernel_code = f'''@ti.kernel
def apply_all_experts():
    dt = 0.016
    
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            # Get current particle state
            pos = tv.p.field[i].pos
            vel = tv.p.field[i].vel
            mass = tv.p.field[i].mass
            species = tv.p.field[i].species
            
            # Initialize total force
            total_force = ti.math.vec2(0.0, 0.0)
            
            # Apply single-particle experts
{single_calls_str}
'''
        
        # Add interaction loop only if we have interaction experts
        if self.interaction_experts:
            kernel_code += f'''
            # Apply interaction experts
            for j in range(tv.pn):
                if i != j and tv.p.field[j].active > 0:
                    p1 = tv.p.field[i]
                    p2 = tv.p.field[j]
                    
{interaction_calls_str}
'''
        
        kernel_code += '''
            # Update velocity with damping
            tv.p.field[i].vel = tv.p.field[i].vel * 0.99 + total_force * dt
            
            # Update position
            new_pos = tv.p.field[i].pos + tv.p.field[i].vel * dt
            
            # Simple boundary handling
            if new_pos.x < 0 or new_pos.x > tv.x:
                tv.p.field[i].vel.x *= -0.8
                new_pos.x = ti.math.clamp(new_pos.x, 0, tv.x)
            if new_pos.y < 0 or new_pos.y > tv.y:
                tv.p.field[i].vel.y *= -0.8
                new_pos.y = ti.math.clamp(new_pos.y, 0, tv.y)
            
            tv.p.field[i].pos = new_pos'''
        
        return kernel_code


class TemporalUpdate(BaseModel):
    frame_updates: Dict[str, str] = Field(
        default_factory=dict,
        description="Frame-based updates (state_name -> update_expression)"
    )
    day_duration: float = Field(default=10.0, description="Day duration in seconds")
    
    def to_code(self) -> str:
        if not self.frame_updates:
            return "@ti.kernel\ndef update_temporal_states():\n    pass"
        
        update_lines = []
        for state_name, update_expr in self.frame_updates.items():
            update_lines.append(f"    tv.s.llm_global.field[0].{state_name} = {update_expr}")
        
        return f'''@ti.kernel
def update_temporal_states():
    frame = tv.ctx.i[None]
    fps = 60.0
    day_frames = {self.day_duration} * fps
    
{chr(10).join(update_lines)}'''


class BehaviorSynthesisRequest(BaseModel):
    description: str
    available_states: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Available states by category"
    )
    screen_size: Tuple[int, int] = (800, 600)
    species_count: int = 5
    particle_count: int = 300


class BehaviorSynthesisResponse(BaseModel):
    experts: List[ExpertFunction]
    states_needed: List[StateDefinition] = Field(
        default_factory=list,
        description="States to create by category"
    )
    species_config: SpeciesConfiguration
    integration_kernel: IntegrationKernel
    temporal_update: Optional[TemporalUpdate] = None
    helper_functions: Optional[Dict[str, str]] = Field(
        default=None,
        description="Helper functions generated alongside experts"
    )
    
    def get_init_code(self) -> str:
        return self.species_config.get_init_code(300, (800, 600))
    
    def get_state_code(self) -> str:
        # This will be implemented by the state manager
        return "# State initialization handled by UnifiedStateManager"


# Allow forward references
ForceComputation.model_rebuild()
# ConditionalBlock has been replaced with ConditionalForce