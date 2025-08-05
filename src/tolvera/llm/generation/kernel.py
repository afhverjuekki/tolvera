import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class IntegrationKernelGenerator:
    """
    Generates integration kernels that combine multiple expert behaviors.
    """
    
    def generate(
        self,
        single_expert_names: List[str],
        interaction_expert_names: List[str],
        expert_weights: Dict[str, float],
        tolvera_instance: Any,
        species_conditions: Optional[Dict[str, List[int]]] = None,
        species_config: Optional[Any] = None
    ) -> str:
        """
        Generate an integration kernel that applies all experts.
        
        Args:
            single_expert_names: Names of single-particle experts
            interaction_expert_names: Names of interaction experts
            expert_weights: Weight for each expert
            tolvera_instance: Tölvera instance for configuration
            species_conditions: Optional mapping of expert names to species IDs they apply to
            species_config: Optional species configuration with interaction pairs
            
        Returns:
            Generated kernel code as string
        """
        lines = ["@ti.kernel", "def apply_all_experts():"]
        
        # Main particle loop
        lines.append("    for i in range(tv.pn):")
        lines.append("        if tv.p.field[i].active > 0:")
        
        # Extract particle properties
        lines.extend([
            "            pos = tv.p.field[i].pos",
            "            vel = tv.p.field[i].vel", 
            "            mass = tv.p.field[i].mass",
            "            species = tv.p.field[i].species",
            "",
            "            force = ti.math.vec2(0.0, 0.0)"
        ])
        
        # Apply single-particle experts
        if single_expert_names:
            lines.append("")
            lines.append("            # Apply single-particle experts")
            for expert_name in single_expert_names:
                weight = expert_weights.get(expert_name, 1.0)
                
                # Check if this expert has species conditions
                if species_conditions and expert_name in species_conditions:
                    species_list = species_conditions[expert_name]
                    if species_list:
                        conditions = " or ".join([f"species == {s}" for s in species_list])
                        lines.append(f"            if {conditions}:")
                        lines.append(f"                force += expert_{expert_name}(pos, vel, mass, species, i) * {weight}")
                    else:
                        lines.append(f"            force += expert_{expert_name}(pos, vel, mass, species, i) * {weight}")
                else:
                    lines.append(f"            force += expert_{expert_name}(pos, vel, mass, species, i) * {weight}")
        
        # Apply interaction experts
        if interaction_expert_names:
            lines.append("")
            lines.append("            # Apply interaction experts")
            lines.append("            for j in range(tv.pn):")
            lines.append("                if i != j and tv.p.field[j].active > 0:")
            
            # Add species variables for cleaner conditions
            if species_config and species_config.interaction_pairs:
                lines.append("                    species_j = tv.p.field[j].species")
            
            for expert_name in interaction_expert_names:
                weight = expert_weights.get(expert_name, 1.0)
                
                # Check if this expert should only apply to certain species pairs
                if species_config and species_config.interaction_pairs and expert_name in species_conditions:
                    # Generate conditions for interaction pairs
                    conditions = []
                    for s1, s2 in species_config.interaction_pairs:
                        conditions.append(f"(species == {s1} and species_j == {s2})")
                        if s1 != s2:  # Add reverse pair for symmetric interactions
                            conditions.append(f"(species == {s2} and species_j == {s1})")
                    
                    if conditions:
                        condition_str = " or ".join(conditions)
                        lines.append(f"                    if {condition_str}:")
                        lines.append(f"                        force += expert_{expert_name}(tv.p.field[i], tv.p.field[j]) * {weight}")
                    else:
                        lines.append(f"                    force += expert_{expert_name}(tv.p.field[i], tv.p.field[j]) * {weight}")
                else:
                    lines.append(f"                    force += expert_{expert_name}(tv.p.field[i], tv.p.field[j]) * {weight}")
        
        # Update velocity and position
        lines.extend([
            "",
            "            # Update velocity and position",
            "            tv.p.field[i].vel += force * 0.05",  # Increased for much better visibility
            "            tv.p.field[i].vel *= 0.95  # Reduced damping for more motion",
            "            tv.p.field[i].pos += tv.p.field[i].vel * 0.016",
            "",
            "            # Handle boundaries",
            "            tv.p.field[i].pos = tv.p.field[i].pos.fract() * ti.math.vec2(tv.x, tv.y)"
        ])
        
        return "\n".join(lines)
    
    def generate_from_response(
        self,
        integration_kernel_model: Any,
        expert_weights: Dict[str, float],
        tolvera_instance: Any
    ) -> str:
        """
        Generate kernel from an IntegrationKernel model object.
        
        Args:
            integration_kernel_model: IntegrationKernel model instance
            expert_weights: Weight for each expert
            tolvera_instance: Tölvera instance
            
        Returns:
            Generated kernel code
        """
        return self.generate(
            single_expert_names=integration_kernel_model.single_experts,
            interaction_expert_names=integration_kernel_model.interaction_experts,
            expert_weights=expert_weights,
            tolvera_instance=tolvera_instance
        )