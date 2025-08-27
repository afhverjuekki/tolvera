"""
Template Renderer Module - Centralized Jinja2 template rendering for code generation
"""
import logging
import os
from typing import List, Dict, Optional, Any, Tuple
from jinja2 import Environment, FileSystemLoader
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TemplateRenderer:
    """
    Centralized template rendering for all code generation.
    Replaces the generation directory modules with direct template access.
    """
    
    def __init__(self):
        # Set up Jinja2 environment with whitespace control
        templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            trim_blocks=True,  # Remove first newline after block
            lstrip_blocks=True  # Strip leading whitespace from blocks
        )
        logger.debug(f"TemplateRenderer initialized with templates from: {templates_dir}")
    
    def render_integration_kernel(
        self,
        single_expert_names: List[str],
        interaction_expert_names: List[str],
        expert_weights: Dict[str, float],
        species_conditions: Optional[Dict[str, List[int]]] = None,
        species_config: Optional[Any] = None,
        visual_expert_names: Optional[List[str]] = None
    ) -> str:
        """
        Render an integration kernel that applies all experts.
        
        Args:
            single_expert_names: Names of single-particle experts
            interaction_expert_names: Names of interaction experts
            expert_weights: Weight for each expert
            species_conditions: Optional mapping of expert names to species IDs they apply to
            species_config: Optional species configuration with interaction pairs
            visual_expert_names: Optional list of visual expert names
            
        Returns:
            Generated kernel code as string
        """
        # Build species-specific expert mapping using explicit species_conditions
        species_expert_map = {}
        all_species_experts = []  # Experts that apply to all species
        
        if species_conditions:
            # Use the explicit species conditions passed from behavior agent
            for expert_name in single_expert_names:
                if expert_name in species_conditions:
                    # This expert has specific species it applies to
                    species_list = species_conditions[expert_name]
                    if species_list:  # Not None and not empty
                        for species_id in species_list:
                            if species_id not in species_expert_map:
                                species_expert_map[species_id] = []
                            species_expert_map[species_id].append(expert_name)
                    else:
                        # Empty list means apply to all
                        all_species_experts.append(expert_name)
                else:
                    # No species condition = apply to all
                    all_species_experts.append(expert_name)
        else:
            # No species conditions provided, all experts apply to all species
            all_species_experts = single_expert_names.copy()
        
        # Check if we have ANY species-specific experts
        has_species_specific = bool(species_expert_map)
        
        # Load and render template
        template = self.env.get_template('kernel/integration_kernel.j2')
        
        return template.render(
            single_expert_names=single_expert_names,
            interaction_expert_names=interaction_expert_names,
            expert_weights=expert_weights,
            species_conditions=species_conditions,
            species_config=species_config,
            species_expert_map=species_expert_map,
            all_species_experts=all_species_experts,
            has_species_specific=has_species_specific
        )
    
    def render_utility_kernel(
        self,
        utility_expert_names: List[str],
        function_name: str = "update_utilities"
    ) -> str:
        """
        Render a kernel that calls utility expert functions.
        
        Args:
            utility_expert_names: Names of utility expert functions
            function_name: Name of the utility kernel
            
        Returns:
            Generated utility kernel code
        """
        if not utility_expert_names:
            return ""
        
        template = self.env.get_template('kernel/utility_kernel.j2')
        
        return template.render(
            function_name=function_name,
            utility_expert_names=utility_expert_names
        )
    
    def render_drawing_kernel(
        self,
        visual_expert_names: List[str],
        function_name: str = "draw"
    ) -> str:
        """
        Render a drawing kernel that calls visual expert functions.
        
        Args:
            visual_expert_names: Names of visual expert functions
            function_name: Name of the drawing kernel
            
        Returns:
            Generated drawing kernel code
        """
        if not visual_expert_names:
            return ""
        
        template = self.env.get_template('kernel/drawing_kernel.j2')
        
        return template.render(
            function_name=function_name,
            visual_expert_names=visual_expert_names
        )
    
    def render_pure_drawing_kernel(
        self,
        drawing_code: str,
        function_name: str = "draw"
    ) -> str:
        """
        Render a kernel for pure drawing behaviors without particles.
        
        Args:
            drawing_code: The drawing code to execute
            function_name: Name of the drawing function
            
        Returns:
            Generated drawing kernel code
        """
        # Split drawing code into lines for template processing
        drawing_code_lines = drawing_code.split('\n')
        
        template = self.env.get_template('kernel/pure_drawing_kernel.j2')
        
        return template.render(
            function_name=function_name,
            drawing_code_lines=drawing_code_lines
        )
    
    def render_pixel_diffusion_kernel(self) -> str:
        """Render pixel diffusion kernel."""
        template = self.env.get_template('kernel/pixel_diffusion.j2')
        return template.render()
    
    def render_pixel_decay_kernel(self) -> str:
        """Render pixel decay kernel."""
        template = self.env.get_template('kernel/pixel_decay.j2')
        return template.render()
    
    def render_pixel_deposition_kernel(self) -> str:
        """Render pixel deposition kernel."""
        template = self.env.get_template('kernel/pixel_deposition.j2')
        return template.render()
    
    def render_sketch(
        self,
        description: str,
        experts: List[str],
        kernel: str,
        init_code: str,
        state_code: str = "",
        temporal_code: str = "",
        config_code: str = "",
        utility_code: str = "",
        utility_kernel: str = "",
        drawing_code: str = "",
        drawing_kernel: str = "",
        respawn_code: str = "",
        environmental_fields: str = "",
        pre_draw_calls: List[str] = None,
        post_draw_calls: List[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        has_non_visual_experts: bool = True
    ) -> str:
        """
        Render complete sketch file.
        
        Args:
            description: Behavior description
            experts: List of expert function code strings
            kernel: Integration kernel code
            init_code: Particle initialization code
            state_code: State initialization code
            temporal_code: Temporal update code
            config_code: Additional configuration code
            utility_code: Utility expert functions
            utility_kernel: Kernel that calls utility experts
            drawing_code: Drawing behavior functions
            drawing_kernel: Drawing kernel that calls visual experts
            respawn_code: Respawn functions for food/resources
            environmental_fields: Environmental fields
            pre_draw_calls: Drawing calls before particles
            post_draw_calls: Drawing calls after particles
            metadata: Optional metadata to include
            has_non_visual_experts: Whether there are non-visual experts
            
        Returns:
            Complete sketch code as string
        """
        # Build update calls - CORRECT ORDER IS CRITICAL
        update_calls = []
        
        # 1. Always apply utility experts first (handles temporal updates, state updates)
        if utility_kernel and "update_utilities" in utility_kernel:
            update_calls.append("update_utilities()  # Execute utility functions")
        
        # Only include particle physics if we have force experts
        if has_non_visual_experts:
            # 2. Apply expert behaviors to calculate forces
            update_calls.append("apply_all_experts()")
            
            # 3. Update physics (positions, velocities, boundaries)
            update_calls.append("tv.p()")
        
        # 4. Apply drawing behaviors if present
        if drawing_kernel and "draw" in drawing_kernel:
            update_calls.append("draw()  # Execute visual behaviors")
        
        # 5. Check for respawn (e.g., food particles)
        if respawn_code and "respawn_food" in respawn_code:
            update_calls.append("respawn_food()")
        
        # Format drawing calls
        if pre_draw_calls is None:
            pre_draw_calls = ["# No pre-draw effects"]
        if post_draw_calls is None:
            post_draw_calls = ["# No post-draw effects"]
        
        # Enhanced config code to always set kwargs properly
        if not config_code or config_code == "# No additional configuration":
            # Always generate proper kwargs configuration
            config_code = """# Configure Tölvera parameters
    # Default configuration - modify kwargs as needed
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080
    if 'pn' not in kwargs:
        kwargs['pn'] = 1000
    if 'sn' not in kwargs:
        kwargs['sn'] = 1  # Default to 1 species"""
        
        # Determine whether to render particles
        if has_non_visual_experts:
            render_particles_call = "tv.px.particles(tv.p, tv.s.species())"
        else:
            render_particles_call = "# No particle rendering (only visual behaviors)"
        
        template = self.env.get_template('sketch/main_sketch.j2')
        
        return template.render(
            description=description,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_code=config_code.strip(),
            init_code=init_code.strip() if init_code else "# No initialization code",
            state_code=state_code.strip() if state_code else "# No custom states needed",
            environmental_fields=environmental_fields if environmental_fields else "# No environmental fields",
            expert_code="\n\n".join(experts) if experts else "# No expert functions",
            kernel_code=kernel,
            temporal_code=temporal_code if temporal_code else "# No temporal updates needed",
            utility_code=utility_code if utility_code else "# No utility functions",
            utility_kernel=utility_kernel if utility_kernel else "# No utility kernel",
            drawing_code=drawing_code if drawing_code else "# No drawing functions",
            drawing_kernel=drawing_kernel if drawing_kernel else "# No drawing kernel",
            respawn_code=respawn_code if respawn_code else "# No respawn functions",
            pre_draw_calls=pre_draw_calls,
            post_draw_calls=post_draw_calls,
            update_calls=update_calls,
            render_particles_call=render_particles_call
        )
    
    def render_pure_drawing_sketch(
        self,
        description: str,
        drawing_code: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render a pure drawing sketch without particle systems.
        
        Args:
            description: Description of what to draw
            drawing_code: The drawing kernel/function code
            metadata: Optional metadata
            
        Returns:
            Complete pure drawing sketch code
        """
        template = self.env.get_template('sketch/pure_drawing_sketch.j2')
        
        return template.render(
            description=description,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            drawing_code=drawing_code.strip()
        )
    
    def render_drawing_instructions(
        self,
        interaction_type: str,
        description: str,
        function_signature: str,
        additional_access: str,
        example_code: str,
        draw_order_info: str = ""
    ) -> str:
        """
        Render drawing instructions for LLM prompts.
        
        Args:
            interaction_type: Type of interaction (e.g., "Single Particle", "Particle-Particle")
            description: Description of what to draw
            function_signature: The function signature template
            additional_access: Additional access patterns
            example_code: Example drawing code
            draw_order_info: Optional draw order information
            
        Returns:
            Rendered drawing instructions
        """
        template = self.env.get_template('drawing/drawing_instructions.j2')
        
        return template.render(
            interaction_type=interaction_type,
            description=description,
            function_signature=function_signature,
            additional_access=additional_access,
            example_code=example_code,
            draw_order_info=draw_order_info
        )
    
    # ========== Data Model Code Generation Methods ==========
    # These methods generate code from data model structures,
    # enforcing separation between data definition and code generation
    
    def render_vector_expression(self, expr: Any) -> str:
        """
        Render a VectorExpression data model to Taichi code.
        
        Args:
            expr: VectorExpression instance
            
        Returns:
            Taichi vector code
        """
        return f"ti.math.vec2({expr.x}, {expr.y})"
    
    def render_state_access(self, access: Any) -> str:
        """
        Render a StateAccess data model to assignment code.
        
        Args:
            access: StateAccess instance
            
        Returns:
            State access assignment code
        """
        return f"{access.var_name} = tv.s.llm_{access.category}.field[{access.index_expr}].{access.state_name}"
    
    def render_conditional_force(self, force: Any) -> str:
        """
        Render a ConditionalForce data model to conditional expression.
        
        Args:
            force: ConditionalForce instance
            
        Returns:
            Conditional force expression
        """
        if force.force_if_false:
            force_true = self.render_vector_expression(force.force_if_true)
            force_false = self.render_vector_expression(force.force_if_false)
            return f"({force_true} if {force.condition} else {force_false})"
        else:
            force_true = self.render_vector_expression(force.force_if_true)
            return f"({force_true} if {force.condition} else ti.math.vec2(0.0, 0.0))"
    
    def render_force_computation(self, comp: Any, indent: int = 1) -> str:
        """
        Render a ForceComputation data model to computation code.
        
        Args:
            comp: ForceComputation instance
            indent: Indentation level
            
        Returns:
            Force computation code
        """
        ind = "    " * indent
        lines = []
        
        # State accesses
        for state in comp.state_accesses:
            lines.append(self.render_state_access(state))
        
        # Helper variables
        for kv_pair in comp.helper_variables:
            lines.append(f"{kv_pair.key} = {kv_pair.value}")
        
        # Start with base force
        base_force = self.render_vector_expression(comp.base_force)
        lines.append(f"force = {base_force}")
        
        # Add conditional forces
        for cond_force in comp.conditional_forces:
            lines.append(f"force += {self.render_conditional_force(cond_force)}")
        
        return f"\n{ind}".join(lines)
    
    def render_drawing_operation(self, op: Any) -> str:
        """
        Render a DrawingOperation data model to drawing code.
        
        Args:
            op: DrawingOperation instance
            
        Returns:
            Drawing operation code
        """
        if op.operation == "set":
            return f"px.set({', '.join(op.args)}, {op.color})"
        elif op.operation == "line":
            return f"px.line({', '.join(op.args)}, {op.color})"
        elif op.operation == "circle":
            return f"px.circle({', '.join(op.args)}, {op.color})"
        elif op.operation == "rect":
            return f"px.rect({', '.join(op.args)}, {op.color})"
        return ""
    
    def render_drawing_computation(self, comp: Any, indent: int = 1) -> str:
        """
        Render a DrawingComputation data model to drawing code.
        
        Args:
            comp: DrawingComputation instance
            indent: Indentation level
            
        Returns:
            Drawing computation code
        """
        ind = "    " * indent
        lines = []
        
        # State accesses
        for state in comp.state_accesses:
            lines.append(self.render_state_access(state))
        
        # Helper variables
        for kv_pair in comp.helper_variables:
            lines.append(f"{kv_pair.key} = {kv_pair.value}")
        
        # Drawing operations
        for op in comp.drawing_operations:
            lines.append(self.render_drawing_operation(op))
        
        # Conditionals (if any)
        for cond in comp.conditionals:
            cond_lines = self.render_conditional_force(cond).split('\n')
            lines.extend(cond_lines)
        
        return f"\n{ind}".join(lines)
    
    def render_expert_function(self, expert: Any) -> str:
        """
        Render an ExpertFunction data model to Taichi function code.
        
        Args:
            expert: ExpertFunction instance
            
        Returns:
            Complete expert function code
        """
        # Check if expert has custom code field (for pre-generated code)
        if hasattr(expert, 'code') and expert.code:
            return expert.code
            
        if expert.is_drawing:
            # Drawing expert function
            if expert.is_interaction:
                params = "px: ti.template(), p1: ti.template(), p2: ti.template()"
            else:
                params = "px: ti.template(), p: ti.template(), i: ti.i32"
            
            computation_code = self.render_drawing_computation(expert.computation)
            
            return f'''@ti.func
def expert_{expert.name}({params}):
    # Extract particle properties
    pos = p.pos if not {expert.is_interaction} else p1.pos
    vel = p.vel if not {expert.is_interaction} else p1.vel
    mass = p.mass if not {expert.is_interaction} else p1.mass
    species = p.species if not {expert.is_interaction} else p1.species
    
    {computation_code}'''
        else:
            # Force expert function
            if expert.is_interaction:
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
            
            computation_code = self.render_force_computation(expert.computation)
            
            return f'''@ti.func
def expert_{expert.name}({params}) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0){param_extraction}
    
    {computation_code}
    
    return force'''
    
    def render_species_init_code(
        self,
        config: Any,
        particle_count: int,
        screen_size: Tuple[int, int]
    ) -> str:
        """
        Render species configuration to initialization code.
        
        Args:
            config: SpeciesConfiguration instance
            particle_count: Number of particles
            screen_size: Screen dimensions
            
        Returns:
            Initialization code string
        """
        num_species = len(config.species_ids)
        
        # Get default colors from ColorResolver
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
            for i, sid in enumerate(config.species_ids):
                init_code += f"species_map[{i}] = {sid}\n"
            init_code += "\n"
        
        # Add species comments if names are available
        if config.species_names:
            init_code += "# Species configuration:\n"
            for mapping in config.species_names:
                init_code += f"# Species {mapping.species_id}: {mapping.name}\n"
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
        tv.p.field[i].speed = 20.0
        # Assign species cyclically
        tv.p.field[i].species = {config.species_ids[0] if num_species == 1 else f"species_map[i % {num_species}]"}

init_particles()
"""
        
        # Set colors with semantic awareness
        color_code = "\n# Set species colors\n"
        
        # Build a dict for quick lookup from the list format
        colors_dict = {}
        if config.colors:
            for mapping in config.colors:
                colors_dict[mapping.species_id] = mapping.rgba
        
        for idx, species_id in enumerate(config.species_ids):
            if species_id in colors_dict:
                color = colors_dict[species_id]
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
    
    def render_temporal_update_kernel(self, update: Any) -> str:
        """
        Render a TemporalUpdate data model to kernel code.
        
        Args:
            update: TemporalUpdate instance
            
        Returns:
            Temporal update kernel code
        """
        if not update.frame_updates:
            return "@ti.kernel\ndef update_temporal_states():\n    pass"
        
        update_lines = []
        for mapping in update.frame_updates:
            update_lines.append(f"    tv.s.llm_global.field[0].{mapping.state_name} = {mapping.update_expression}")
        
        return f'''@ti.kernel
def update_temporal_states():
    frame = tv.ctx.i[None]
    fps = 60.0
    day_frames = {update.day_duration} * fps
    
{chr(10).join(update_lines)}'''