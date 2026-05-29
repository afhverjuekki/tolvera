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
    
    def _clean_code_section(self, code: str) -> str:
        """Remove trailing quotes from code sections that might break the file."""
        if not code:
            return code
        
        # Remove trailing triple quotes and variations
        code = code.rstrip()
        
        # Check for various quote patterns at the end
        patterns_to_remove = ['"""', "'''", '""', "''", '"', "'"]
        for pattern in patterns_to_remove:
            if code.endswith(pattern):
                code = code[:-len(pattern)].rstrip()
                # Silently remove trailing quotes
        
        return code
    
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
        # Detect musical intent once — paired with the SuperCollider companion
        # patch emitted by src/tolvera/llm/sc/emitter.py; both sides share the
        # has_musical_intent gate so OSC addresses always match.
        from ..sc.emitter import OSC_SENDER_BLOCK, has_musical_intent
        musical = has_musical_intent(description)

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

        # 6. OSC metrics kernel — only when musical intent fires
        if musical:
            update_calls.append("_compute_osc_metrics()  # OSC senders")

        osc_sender_code = OSC_SENDER_BLOCK if musical else ""
        
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

        # When OSC senders are injected, ensure the Tölvera instance has osc enabled
        if musical:
            config_code = config_code + "\n    if 'osc' not in kwargs:\n        kwargs['osc'] = True  # Required for OSC senders to SuperCollider companion"
        
        # Determine whether to render particles
        if has_non_visual_experts:
            render_particles_call = "tv.px.particles(tv.p, tv.s.species())"
        else:
            render_particles_call = "# No particle rendering (only visual behaviors)"
        
        # Clean all code sections to remove trailing quotes
        cleaned_experts = [self._clean_code_section(e) for e in experts] if experts else []
        
        template = self.env.get_template('sketch/final_sketch.j2')
        
        return template.render(
            description=description,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_code=self._clean_code_section(config_code).strip(),
            init_code=self._clean_code_section(init_code).strip() if init_code else "# No initialization code",
            state_code=self._clean_code_section(state_code).strip() if state_code else "# No custom states needed",
            environmental_fields=self._clean_code_section(environmental_fields) if environmental_fields else "# No environmental fields",
            expert_code="\n\n".join(cleaned_experts) if cleaned_experts else "# No expert functions",
            kernel_code=self._clean_code_section(kernel),
            temporal_code=self._clean_code_section(temporal_code) if temporal_code else "# No temporal updates needed",
            utility_code=self._clean_code_section(utility_code) if utility_code else "# No utility functions",
            utility_kernel=self._clean_code_section(utility_kernel) if utility_kernel else "# No utility kernel",
            drawing_code=self._clean_code_section(drawing_code) if drawing_code else "# No drawing functions",
            drawing_kernel=self._clean_code_section(drawing_kernel) if drawing_kernel else "# No drawing kernel",
            respawn_code=self._clean_code_section(respawn_code) if respawn_code else "# No respawn functions",
            osc_sender_code=osc_sender_code,
            pre_draw_calls=pre_draw_calls,
            post_draw_calls=post_draw_calls,
            update_calls=update_calls,
            render_particles_call=render_particles_call
        )
    
    # ========== Data Model Code Generation Methods ==========
    # These methods generate code from data model structures,
    # enforcing separation between data definition and code generation
    
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