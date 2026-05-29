"""Expert function registry management.

This module provides the registry system for managing synthesized expert
functions, tracking their metadata, types, and associations.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ExpertInfo:
    """Information about a registered expert function.
    
    Attributes:
        name (str): Name of the expert function.
        description (str): Human-readable description.
        weight (float): Weight in integration kernel.
        expert_type (str): Type ('single', 'interaction', 'drawing', 'utility').
        code (str): Generated Taichi code.
        draw_order (Optional[str]): 'pre' or 'post' for drawing experts.
        applies_to_species (Optional[List[int]]): Species IDs this expert affects.
    """
    name: str
    description: str
    weight: float
    expert_type: str  # 'single', 'interaction', 'drawing', 'utility'
    code: str
    draw_order: Optional[str] = None  # 'pre' or 'post' for drawing experts
    applies_to_species: Optional[List[int]] = None  # Species IDs this expert applies to


class ExpertRegistry:
    """Registry for managing expert functions.
    
    This class maintains a central registry of all synthesized expert
    functions, tracking their types, weights, and species associations.
    """
    
    def __init__(self):
        """Initialize the expert registry."""
        self.experts: List[ExpertInfo] = []
        self.weights: Dict[str, float] = {}
    
    def register(self, expert_info: ExpertInfo) -> None:
        """Register a new expert.
        
        Args:
            expert_info: Information about the expert to register
        """
        self.experts.append(expert_info)
        self.weights[expert_info.name] = expert_info.weight
    
    def get_by_type(self, expert_type: str) -> List[ExpertInfo]:
        """Get all experts of a specific type.
        
        Args:
            expert_type: Type of experts to retrieve
            
        Returns:
            List of experts matching the specified type
        """
        return [e for e in self.experts if e.expert_type == expert_type]
    
    def get_kernel_params(self) -> Dict[str, Any]:
        """Get parameters needed for kernel generation.
        
        Returns:
            Dictionary containing expert names, weights, and species conditions
        """
        single_experts = self.get_by_type('single')
        interaction_experts = self.get_by_type('interaction')
        visual_experts = self.get_by_type('visual') + self.get_by_type('drawing')
        utility_experts = self.get_by_type('utility')
        
        # Build species conditions mapping
        species_conditions = {}
        for expert in self.experts:
            if expert.applies_to_species is not None:
                species_conditions[expert.name] = expert.applies_to_species
        
        return {
            'single_expert_names': [e.name for e in single_experts],
            'interaction_expert_names': [e.name for e in interaction_experts],
            'visual_expert_names': [e.name for e in visual_experts],
            'utility_expert_names': [e.name for e in utility_experts],
            'expert_weights': self.weights,
            'species_conditions': species_conditions
        }
    
    def clear(self) -> None:
        """Clear all registered experts."""
        self.experts.clear()
        self.weights.clear()
    
    def get_expert_info_list(self) -> List[Dict[str, Any]]:
        """Get information about all registered experts.
        
        Returns:
            List of dictionaries containing expert information
        """
        return [
            {
                'name': e.name,
                'description': e.description,
                'weight': e.weight,
                'expert_type': e.expert_type
            }
            for e in self.experts
        ]