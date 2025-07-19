"""
State Code Generator for generating Tölvera state initialization code.

This module provides utilities to generate Python code for initializing
Tölvera states using Jinja2 templates.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template

logger = logging.getLogger(__name__)


class StateCodeGenerator:
    """Generates Python code for Tölvera state initialization."""
    
    def __init__(self):
        """Initialize the code generator with Jinja2 environment."""
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.template = self.env.get_template("state_init.j2")
    
    def generate_state_init_code(self, state_spec: Dict[str, Any]) -> str:
        """
        Generate Python code for initializing Tölvera states.
        
        Args:
            state_spec: State specification containing global_states, 
                       particle_states, and/or species_states
        
        Returns:
            Generated Python code as a string
        """
        # Ensure types are rendered as strings without quotes
        formatted_spec = self._format_state_spec(state_spec)
        
        return self.template.render(states=formatted_spec)
    
    def _format_state_spec(self, state_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format state specification for template rendering.
        
        Ensures that Taichi types are properly formatted as code strings.
        """
        formatted = {}
        
        for category in ['global_states', 'particle_states', 'species_states']:
            if category in state_spec and state_spec[category]:
                formatted[category] = {}
                for name, info in state_spec[category].items():
                    formatted[category][name] = {
                        'type': info['type'],  # Already a string like "ti.f32"
                        'min': info['min'],
                        'max': info['max']
                    }
        
        return formatted
    
    @staticmethod
    def generate_from_experts(poe_system) -> str:
        """
        Generate state initialization code from PoE system experts.
        
        Args:
            poe_system: PoE behavior system with experts
            
        Returns:
            Generated Python code as a string
        """
        generator = StateCodeGenerator()
        
        # Collect all state specifications from all experts
        combined_state_spec = {
            'global_states': {},
            'particle_states': {},
            'species_states': {}
        }
        
        found_states = False
        for expert in poe_system.experts:
            state_spec = expert.metadata.get('state_spec', {})
            logger.info(f"Checking expert {expert.name} for states: {state_spec}")
            
            # Merge states from this expert into combined spec
            for category in ['global_states', 'particle_states', 'species_states']:
                if category in state_spec and state_spec[category]:
                    combined_state_spec[category].update(state_spec[category])
                    found_states = True
        
        if found_states:
            logger.info(f"Generating combined state initialization code: {combined_state_spec}")
            return generator.generate_state_init_code(combined_state_spec)
        
        logger.warning("No states found in any expert metadata")
        return ""  # No states to generate