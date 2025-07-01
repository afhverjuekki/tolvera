"""
Simplified integration layer between PoE behavior system and Tölvera.

This module provides the glue between the PoE expert system and Tölvera's
particle system, handling force application without mouse tracking.
"""

from typing import Dict, Any, List
import logging

from .poe_core import PoEBehaviorSystem, SimpleProgrammaticExpert
from .poe_experts import ExpertManager
from .poe_ollama import PoEExpertSynthesizer

logger = logging.getLogger(__name__)


class TolveraBehaviorAgent:
    """
    Simplified integration of PoE behavior system with Tölvera particle system.
    
    This version focuses on pure particle behaviors without mouse interaction.
    """
    
    def __init__(self, tolvera_instance):
        self.tv = tolvera_instance
        self.poe_system = PoEBehaviorSystem(tolvera_instance)
        self.expert_manager = ExpertManager()
        
        # Simple context without mouse
        self.context = {
            "world_width": float(tolvera_instance.x),
            "world_height": float(tolvera_instance.y),
            "time": 0.0,
            "dt": 0.016
        }
        
        logger.info(f"Initialized TolveraBehaviorAgent with {tolvera_instance.pn} particles")
    
    def add_expert_from_code(self, name: str, code: str, weight: float = 1.0):
        """Add an expert from raw code string."""
        expert = SimpleProgrammaticExpert(name, code, weight)
        self.poe_system.add_expert(expert)
        self.expert_manager.add_expert(name, expert)
        logger.info(f"Added expert from code: {name}")
        return expert
    
    async def add_expert_from_description(self, description: str, synthesizer: PoEExpertSynthesizer, weight: float = 1.0):
        """Generate and add expert from natural language description."""
        
        # Set kernel synthesizer if not already set
        if self.poe_system._kernel_synthesizer is None:
            self.poe_system.set_kernel_synthesizer(synthesizer)
        
        # Synthesize expert
        result = await synthesizer.synthesize_expert(description)
        
        if result["success"]:
            expert = SimpleProgrammaticExpert(
                name=result["name"],
                code=result["code"],
                weight=weight
            )
            expert.metadata["description"] = description
            expert.metadata["raw_llm_response"] = result.get("raw_response", "")
            
            # Add expert to system
            self.poe_system.add_expert(expert)
            self.expert_manager.add_expert(result["name"], expert)
            
            # Regenerate integration kernel
            kernel_success = await self.poe_system._regenerate_kernel()
            if not kernel_success:
                logger.warning(f"Expert {result['name']} added but kernel regeneration failed")
            
            logger.info(f"Successfully added synthesized expert: {result['name']}")
            return expert
        else:
            logger.error(f"Failed to synthesize expert: {result['errors']}")
            raise ValueError(f"Expert synthesis failed: {result['errors']}")
    
    def update(self, dt: float = 0.016):
        """Main update method called in render loop."""
        # Update context
        self.context["dt"] = dt
        self.context["time"] += dt
        
        # Compute and apply forces from all experts
        self.poe_system.compute_and_apply_forces(dt)
    
    def set_expert_weight(self, expert_name: str, weight: float):
        """Update the weight of a specific expert."""
        self.poe_system.set_expert_weight(expert_name, weight)
    
    def get_expert_info(self) -> List[Dict[str, Any]]:
        """Get information about all experts."""
        return self.poe_system.get_expert_info()
    
    def clear_all_experts(self):
        """Remove all experts from the system."""
        self.poe_system.clear_experts()
        self.expert_manager.clear_all()
        logger.info("Cleared all experts from agent")