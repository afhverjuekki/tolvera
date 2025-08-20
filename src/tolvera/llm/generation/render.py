"""
Dynamic Render Generator

Generates adaptive render loop sequences based on behavior pattern types.
Different patterns require different render sequences for optimal visualization.
"""

import logging
from typing import List, Dict, Optional, Any
from jinja2 import Environment, FileSystemLoader
import os
from ..core.behavior_requirements import BehaviorRequirements

logger = logging.getLogger(__name__)


class RenderLoopGenerator:
    """Generates dynamic render loops adapted to behavior patterns."""
    
    # Pattern-specific render sequences
    PATTERN_SEQUENCES = {
        "cellular_automaton": [
            "clear_inactive",
            "update_utilities",
            "apply_all_experts",
            "update_visuals",
            "render_particles"
        ],
        "physarum": [
            "decay_pheromones",
            "diffuse_pheromones",
            "apply_all_experts",
            "deposit_trails",
            "render_particles"
        ],
        "ecosystem": [
            "update_resources",
            "apply_all_experts",
            "handle_interactions",
            "update_energy",
            "render_particles"
        ],
        "swarm": [
            "apply_all_experts",
            "render_trails",
            "render_particles"
        ],
        "reaction_diffusion": [
            "diffuse_chemicals",
            "react_chemicals",
            "apply_all_experts",
            "render_particles"
        ],
        "ant_colony": [
            "decay_pheromones",
            "apply_all_experts",
            "deposit_pheromones",
            "render_particles"
        ],
        "neural": [
            "update_synapses",
            "propagate_signals",
            "apply_all_experts",
            "render_connections",
            "render_particles"
        ],
        "growth": [
            "apply_all_experts",
            "branch_growth",
            "render_branches",
            "render_particles"
        ],
        "particle_system": [
            "apply_all_experts",
            "render_particles"
        ],
        "pure_drawing": [
            "clear",
            "draw"
        ]
    }
    
    def __init__(self):
        """Initialize the render loop generator."""
        logger.info("Initialized RenderLoopGenerator")
        
        # Set up Jinja2 environment
        templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
        self.env = Environment(loader=FileSystemLoader(templates_dir))
    
    def generate_render_loop(
        self,
        requirements: BehaviorRequirements,
        available_kernels: List[str],
        has_drawing_behaviors: bool = False
    ) -> str:
        """
        Generate a render loop based on behavior requirements.
        
        Args:
            requirements: Behavior requirements analysis
            available_kernels: List of available kernel function names
            has_drawing_behaviors: Whether drawing behaviors are present
            
        Returns:
            Generated render loop code
        """
        pattern_type = requirements.pattern_type
        render_sequence = requirements.render_sequence or self.PATTERN_SEQUENCES.get(
            pattern_type, 
            self.PATTERN_SEQUENCES["particle_system"]
        )
        
        logger.info(f"Generating render loop for pattern: {pattern_type}")
        logger.info(f"Render sequence: {render_sequence}")
        
        # Load and render template using Jinja2
        template = self.env.get_template('render/render_loop.j2')
        
        return template.render(
            pattern_type=pattern_type,
            pixel_field=requirements.pixel_field,
            available_kernels=available_kernels,
            has_drawing_behaviors=has_drawing_behaviors
        )
    
    def generate_simple_render_loop(self) -> str:
        """Generate a simple default render loop."""
        # Load and render template using Jinja2
        template = self.env.get_template('render/simple_render_loop.j2')
        return template.render()
    
    def get_required_operations(self, pattern_type: str) -> List[str]:
        """
        Get list of required operations for a pattern type.
        
        Args:
            pattern_type: The detected pattern type
            
        Returns:
            List of operation names needed
        """
        sequence = self.PATTERN_SEQUENCES.get(pattern_type, [])
        
        # Map operations to kernel names
        operation_map = {
            "clear_inactive": None,  # Built-in
            "update_utilities": "update_utilities",
            "apply_all_experts": "apply_all_experts",
            "update_visuals": None,  # Built-in
            "render_particles": None,  # Built-in
            "decay_pheromones": "decay_pheromones",
            "diffuse_pheromones": "diffuse_pheromones",
            "deposit_trails": "deposit_trails",
            "deposit_pheromones": "deposit_trails",
            "update_resources": "update_resources",
            "handle_interactions": "handle_interactions",
            "update_energy": "update_energy",
            "render_trails": "render_trails",
            "diffuse_chemicals": "diffuse_chemicals",
            "react_chemicals": "react_chemicals",
            "update_synapses": "update_synapses",
            "propagate_signals": "propagate_signals",
            "render_connections": "render_connections",
            "branch_growth": "branch_growth",
            "render_branches": "render_branches"
        }
        
        required_ops = []
        for op in sequence:
            kernel_name = operation_map.get(op)
            if kernel_name:
                required_ops.append(kernel_name)
        
        return required_ops
    
    def generate_update_calls(self, operations: List[str]) -> List[str]:
        """
        Generate the update call sequence for render loop.
        
        Args:
            operations: List of operation/kernel names
            
        Returns:
            List of code lines for update calls
        """
        calls = []
        
        for op in operations:
            if op == "apply_all_experts":
                calls.append("apply_all_experts()")
                calls.append("tv.p()")  # After experts, for boundaries
            elif op == "update_utilities":
                calls.append("update_utilities()")
            elif op in ["decay_pheromones", "diffuse_pheromones", "deposit_trails"]:
                calls.append(f"{op}()")
            elif op == "render_particles":
                calls.append("tv.px.particles(tv.p, tv.s.species())")
        
        return calls
    
    def generate_pure_drawing_render_loop(self, drawing_function_name: str = "draw") -> str:
        """
        Generate a render loop for pure drawing behaviors without particles.
        
        Args:
            drawing_function_name: Name of the drawing function to call
            
        Returns:
            Generated render loop code
        """
        # Load and render template using Jinja2
        template = self.env.get_template('render/pure_drawing_render_loop.j2')
        
        return template.render(
            drawing_function_name=drawing_function_name
        )