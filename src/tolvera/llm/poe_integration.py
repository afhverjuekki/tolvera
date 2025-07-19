
"""
Integration layer between PoE behavior system and Tölvera.

This module provides the glue between the PoE expert system and Tölvera's particle system.
"""

from typing import Dict, Any, List, Optional
import logging

from .poe_core import PoEBehaviorSystem, SimpleProgrammaticExpert
from .poe_experts import ExpertManager
from .poe_synthesis import PoEExpertSynthesizer
from .species_manager import SpeciesManager
from .dynamic_state_manager import DynamicStateManager

logger = logging.getLogger(__name__)


class TolveraBehaviorAgent:

    def __init__(self, tolvera_instance):
        self.tv = tolvera_instance
        self.poe_system = PoEBehaviorSystem(tolvera_instance)
        self.expert_manager = ExpertManager()
        self.species_manager = SpeciesManager(tolvera_instance)
        self.current_boundary_mode = None  # Track boundary mode

        logger.info(
            f"Initialized TolveraBehaviorAgent with {tolvera_instance.pn} particles and {tolvera_instance.sn} species")

    def add_expert_from_code(self, name: str, code: str, weight: float = 1.0):
        expert = SimpleProgrammaticExpert(name, code, weight)
        self.poe_system.add_expert(expert)
        self.expert_manager.add_expert(name, expert)
        logger.info(f"Added expert from code: {name}")
        return expert

    async def add_expert_from_description(
            self,
            description: str,
            synthesizer: PoEExpertSynthesizer,
            weight: float = 1.0,
            use_decomposition: bool = None,
            use_states: bool = True):
        """
        Add an expert from a natural language description.
        
        Args:
            description: Natural language behavior description
            synthesizer: The synthesizer to use
            weight: Weight for the expert
            use_decomposition: Whether to use decomposition (None = use synthesizer default)
            use_states: Whether to analyze and create states (default True)
            
        Returns:
            The expert if single behavior, or first expert if decomposed
        """
        # Ensure synthesizer has access to Tölvera instance for state management
        if use_states and not synthesizer.state_manager:
            synthesizer.state_manager = DynamicStateManager(self.tv)
        # Check if we should use decomposition
        if use_decomposition is None:
            use_decomposition = synthesizer.enable_decomposition
        
        if use_decomposition and synthesizer.enable_decomposition:
            # Use composite behavior method which handles decomposition
            experts = await self.add_composite_behavior(description, synthesizer, weight)
            # Return the first expert for backward compatibility
            return experts[0] if experts else None

        # Step 1: Synthesize expert @ti.func
        logger.info(
            f"Step 1: Synthesizing expert function for: '{description}'")
        
        # Use state-aware synthesis if enabled
        if use_states:
            result = await synthesizer.synthesize_expert_with_states(description)
        else:
            # Use interaction synthesis method which automatically detects interaction keywords
            result = await synthesizer.synthesize_interaction_expert(description)

        if result["success"]:
            expert = SimpleProgrammaticExpert(
                name=result["name"],
                code=result["code"],
                weight=weight
            )
            expert.metadata["description"] = description
            expert.metadata["raw_llm_response"] = result.get(
                "raw_response", "")
            expert.metadata["is_interaction"] = result.get("is_interaction", False)
            expert.metadata["species_info"] = result.get("species_info", {})
            expert.metadata['state_spec'] = result.get('state_spec', {})
            expert.metadata['state_context'] = result.get('state_context', {})

            # Log the generated expert code
            logger.info(
                f"Generated expert '{result['name']}' for description: '{description}'")
            logger.debug(
                f"Generated code for '{result['name']}':{result['code']}")

            self.poe_system.add_expert(expert)
            self.expert_manager.add_expert(result["name"], expert)

            # Analyze boundary requirements if not already set
            if self.current_boundary_mode is None:
                self.current_boundary_mode = synthesizer.analyze_boundary_requirements(description)
            
            # Step 2: Regenerate integration @ti.kernel with all experts
            logger.info(
                f"Step 2: Regenerating integration kernel for {len(self.poe_system.experts)} experts")
            kernel_success = await self.poe_system.regenerate_integration_kernel(synthesizer, self.current_boundary_mode)
            if not kernel_success:
                logger.error(
                    f"Expert {result['name']} added but kernel regeneration failed")
                raise RuntimeError(
                    f"Failed to regenerate integration kernel after adding expert {result['name']}")

            logger.info(
                f"Successfully added expert {result['name']} and regenerated integration kernel")
            return expert
        else:
            logger.error(f"Failed to synthesize expert: {result['errors']}")
            raise ValueError(f"Expert synthesis failed: {result['errors']}")

    def _calculate_expert_weight(self, result: Dict[str, Any], base_weight: float) -> float:
        if 'adjusted_weight' in result:
            return base_weight * result['adjusted_weight']
        return base_weight * result.get('weight', 1.0)
    
    def _assign_expert_metadata(self, expert: SimpleProgrammaticExpert, result: Dict[str, Any], 
                               description: str, index: int, total_results: int) -> None:
        expert.metadata["description"] = result.get("description", description)
        expert.metadata["raw_llm_response"] = result.get("raw_response", "")
        expert.metadata["is_interaction"] = result.get("is_interaction", False)
        expert.metadata["species_info"] = result.get("species_info", {})
        # IMPORTANT: Include state information for decomposed behaviors
        expert.metadata['state_spec'] = result.get('state_spec', {})
        expert.metadata['state_context'] = result.get('state_context', {})
        
        if result.get("is_decomposed", False):
            expert.metadata["is_decomposed"] = True
            expert.metadata["decomposed_from"] = result.get("original_description", description)
            expert.metadata["sub_behavior_index"] = result.get("sub_behavior_index", index)
            expert.metadata["total_sub_behaviors"] = result.get("sub_behavior_count", total_results)
            expert.metadata["relationship"] = result.get("relationship", "independent")
    
    def _create_expert_from_result(self, result: Dict[str, Any], base_weight: float, 
                                  description: str, index: int, total_results: int) -> SimpleProgrammaticExpert:
        actual_weight = self._calculate_expert_weight(result, base_weight)
        
        expert = SimpleProgrammaticExpert(
            name=result["name"],
            code=result["code"],
            weight=actual_weight
        )
        
        self._assign_expert_metadata(expert, result, description, index, total_results)
        
        logger.info(f"Generated expert '{result['name']}' with weight {actual_weight:.2f}")
        logger.debug(f"Generated code for '{result['name']}':\n{result['code']}")
        
        return expert
    
    def _rollback_experts(self, experts_to_remove: List[SimpleProgrammaticExpert]) -> None:
        for expert in experts_to_remove:
            self.poe_system.experts.remove(expert)
            self.expert_manager.experts.pop(expert.name, None)
    
    async def add_composite_behavior(
            self,
            description: str,
            synthesizer: PoEExpertSynthesizer,
            base_weight: float = 1.0) -> List[SimpleProgrammaticExpert]:
        """
        Add a potentially complex behavior that may be decomposed into multiple experts.
        
        Args:
            description: Natural language behavior description (may be complex)
            synthesizer: The synthesizer to use (should have decomposition enabled)
            base_weight: Base weight to apply (will be distributed among sub-behaviors)
            
        Returns:
            List of experts that were added
        """
        logger.info(f"Adding composite behavior: '{description}'")
        
        # Analyze boundary requirements if not already set
        if self.current_boundary_mode is None:
            self.current_boundary_mode = synthesizer.analyze_boundary_requirements(description)
        
        results = await synthesizer.synthesize_with_decomposition(description)
        
        if not results:
            raise ValueError("No results from synthesis")
        
        # Check if we got multiple sub-behaviors
        if len(results) > 1:
            logger.info(f"Behavior decomposed into {len(results)} sub-behaviors")
        
        # Process results and create experts
        added_experts, failed_count = self._process_synthesis_results(
            results, base_weight, description
        )
        
        if not added_experts:
            raise ValueError(f"All {len(results)} sub-behaviors failed to synthesize")
        
        # Regenerate integration kernel
        await self._regenerate_kernel_or_rollback(synthesizer, added_experts)
        
        logger.info(f"Successfully added {len(added_experts)} experts from composite behavior "
                   f"('{description}') with {failed_count} failures")
        
        return added_experts
    
    def _process_synthesis_results(self, results: List[Dict[str, Any]], 
                                 base_weight: float, description: str) -> tuple:
        added_experts = []
        failed_count = 0
        
        for i, result in enumerate(results):
            if result["success"]:
                expert = self._create_expert_from_result(
                    result, base_weight, description, i, len(results)
                )
                
                self.poe_system.add_expert(expert)
                self.expert_manager.add_expert(result["name"], expert)
                added_experts.append(expert)
            else:
                failed_count += 1
                logger.error(f"Failed to synthesize sub-behavior {i+1}: {result.get('errors', ['Unknown error'])}")
        
        return added_experts, failed_count
    
    async def _regenerate_kernel_or_rollback(self, synthesizer: PoEExpertSynthesizer, 
                                           added_experts: List[SimpleProgrammaticExpert]) -> None:
        logger.info(f"Regenerating integration kernel for {len(self.poe_system.experts)} experts")
        kernel_success = await self.poe_system.regenerate_integration_kernel(synthesizer, self.current_boundary_mode)
        
        if not kernel_success:
            logger.error("Kernel regeneration failed after adding composite behavior")
            self._rollback_experts(added_experts)
            raise RuntimeError("Failed to regenerate integration kernel after adding composite behavior")

    def set_expert_weight(self, expert_name: str, weight: float):
        self.poe_system.set_expert_weight(expert_name, weight)

    def get_expert_info(self) -> List[Dict[str, Any]]:
        return self.poe_system.get_expert_info()

    def clear_all_experts(self):
        self.poe_system.clear_experts()
        self.expert_manager.clear_all()
        logger.info("Cleared all experts from agent")
    
    def get_species_requirements(self) -> tuple:
        behaviors = []
        for expert in self.poe_system.experts:
            behaviors.append({
                'description': expert.metadata.get('description', ''),
                'species_info': expert.metadata.get('species_info', {}),
                'is_interaction': expert.metadata.get('is_interaction', False)
            })
        
        return self.species_manager.analyze_species_requirements(behaviors)
    
    def get_species_initialization_code(self, species_ids: List[int]) -> str:
        return self.species_manager.get_species_initialization_code(species_ids)
    
    def set_boundary_mode(self, mode: str):
        from .boundary_manager import BoundaryMode
        mode_map = {
            'none': BoundaryMode.NONE,
            'wrap': BoundaryMode.WRAP,
            'bounce': BoundaryMode.BOUNCE,
            'absorb': BoundaryMode.ABSORB
        }
        if mode.lower() in mode_map:
            self.current_boundary_mode = mode_map[mode.lower()]
            logger.info(f"Set boundary mode to: {self.current_boundary_mode.value}")
        else:
            logger.warning(f"Invalid boundary mode: {mode}. Valid modes: {list(mode_map.keys())}")
    
    def get_boundary_mode(self) -> Optional[str]:
        if self.current_boundary_mode:
            return self.current_boundary_mode.value
        return None
