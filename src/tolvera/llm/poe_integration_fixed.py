"""
Fixed integration layer between PoE behavior system and Tölvera.

This module provides the glue between the PoE expert system and Tölvera's
particle system, handling force application and state management.
"""

import taichi as ti
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import asyncio

from .poe_core_fixed import PoEBehaviorSystemV2, SimpleProgrammaticExpert

logger = logging.getLogger(__name__)


class TolveraBehaviorAgentFixed:
    """
    Fixed integration of PoE behavior system with Tölvera particle system.
    
    This version properly executes dynamically generated expert code
    instead of routing to templates.
    """
    
    def __init__(self, tolvera_instance):
        self.tv = tolvera_instance
        self.poe_system = PoEBehaviorSystemV2(tolvera_instance)
        
        # Context for experts
        self.context = {
            "mouse_x": 0.0,
            "mouse_y": 0.0,
            "world_width": float(tolvera_instance.x),
            "world_height": float(tolvera_instance.y),
            "time": 0.0,
            "dt": 0.016
        }
        
        # Track update time
        self.last_update_time = 0.0
        
        logger.info(f"Initialized TolveraBehaviorAgentFixed with {tolvera_instance.pn} particles")
    
    def add_expert(self, expert: SimpleProgrammaticExpert):
        """Add a pre-compiled expert to the system."""
        self.poe_system.add_expert(expert)
    
    def add_expert_from_code(self, name: str, code: str, weight: float = 1.0):
        """Add an expert from raw code string."""
        expert = SimpleProgrammaticExpert(name, code, weight)
        self.add_expert(expert)
    
    async def add_expert_from_description(self, description: str, synthesizer, weight: float = 1.0):
        """Generate and add expert from natural language description."""
        # Use the synthesizer to generate code
        code = await synthesizer.synthesize_expert(description)
        
        # Extract function name from code
        import re
        match = re.search(r'def\s+(\w+)', code)
        name = match.group(1) if match else f"expert_{len(self.poe_system.experts)}"
        
        # Create and add expert
        expert = SimpleProgrammaticExpert(name, code, weight)
        self.add_expert(expert)
        
        logger.info(f"Added synthesized expert: {name}")
        return expert
    
    def update_context(self, dt: float = 0.016):
        """Update context with current state."""
        self.context["dt"] = dt
        self.context["time"] += dt
        
        # Update mouse position if available
        if hasattr(self.tv, 'mouse'):
            self.context["mouse_x"] = float(self.tv.mouse.x)
            self.context["mouse_y"] = float(self.tv.mouse.y)
    
    def update(self, dt: float = 0.016):
        """Main update method called in render loop."""
        # Update context
        self.update_context(dt)
        
        # Compute and apply forces from all experts
        self.poe_system.compute_and_apply_forces(self.context, dt)
    
    def set_expert_weight(self, expert_name: str, weight: float):
        """Update the weight of a specific expert."""
        for expert in self.poe_system.experts:
            if expert.name == expert_name:
                expert.weight = weight
                # Update in Taichi field
                idx = self.poe_system.experts.index(expert)
                self.poe_system.expert_weights[idx] = weight
                logger.info(f"Updated weight for {expert_name} to {weight}")
                return
        logger.warning(f"Expert {expert_name} not found")
    
    def get_expert_info(self) -> List[Dict[str, Any]]:
        """Get information about all experts."""
        info = []
        for i, expert in enumerate(self.poe_system.experts):
            info.append({
                "name": expert.name,
                "weight": expert.weight,
                "index": i,
                "code_preview": expert.code[:100] + "..." if len(expert.code) > 100 else expert.code
            })
        return info
    
    def optimize_weights(self, target_behavior_data: np.ndarray, learning_rate: float = 0.01):
        """
        Optimize expert weights based on target behavior data.
        
        This is a simplified gradient descent approach. In practice,
        you'd use more sophisticated optimization like L-BFGS.
        """
        # Get current particle positions
        current_positions = self.tv.p.field.pos.to_numpy()
        
        # Compute loss (simplified - just MSE of positions)
        loss = np.mean((current_positions - target_behavior_data) ** 2)
        
        # Gradient descent on weights (simplified)
        for i, expert in enumerate(self.poe_system.experts):
            # Numerical gradient estimation
            eps = 0.001
            
            # Forward difference
            original_weight = expert.weight
            expert.weight = original_weight + eps
            self.poe_system.expert_weights[i] = expert.weight
            
            # Recompute with perturbed weight
            self.update(0.016)
            new_positions = self.tv.p.field.pos.to_numpy()
            new_loss = np.mean((new_positions - target_behavior_data) ** 2)
            
            # Compute gradient
            gradient = (new_loss - loss) / eps
            
            # Update weight
            expert.weight = original_weight - learning_rate * gradient
            expert.weight = max(0.0, min(1.0, expert.weight))  # Clamp to [0, 1]
            self.poe_system.expert_weights[i] = expert.weight
        
        logger.info(f"Optimization step complete. Loss: {loss}")
        return loss


class AsyncTolveraBehaviorAgent(TolveraBehaviorAgentFixed):
    """Async version of the behavior agent for use with async synthesizers."""
    
    def __init__(self, tolvera_instance):
        super().__init__(tolvera_instance)
        self._synthesis_queue = asyncio.Queue()
        self._synthesis_task = None
    
    async def start_synthesis_worker(self):
        """Start background worker for expert synthesis."""
        self._synthesis_task = asyncio.create_task(self._synthesis_worker())
    
    async def _synthesis_worker(self):
        """Background worker that processes synthesis requests."""
        while True:
            try:
                description, synthesizer, weight = await self._synthesis_queue.get()
                await self.add_expert_from_description(description, synthesizer, weight)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Synthesis worker error: {e}")
    
    async def queue_expert_synthesis(self, description: str, synthesizer, weight: float = 1.0):
        """Queue an expert for background synthesis."""
        await self._synthesis_queue.put((description, synthesizer, weight))
    
    def stop(self):
        """Stop the synthesis worker."""
        if self._synthesis_task:
            self._synthesis_task.cancel()


# Convenience functions for common expert patterns

def create_attract_expert(name: str, target_x: float, target_y: float, 
                         strength: float = 50.0, max_distance: float = 200.0) -> SimpleProgrammaticExpert:
    """Create an attraction expert to a fixed point."""
    code = f"""@ti.func
def {name}(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    pos = tv.p.field[i].pos
    target = ti.Vector([{target_x}, {target_y}])
    diff = target - pos
    dist = diff.norm()
    
    force = ti.Vector([0.0, 0.0])
    if 1.0 < dist < {max_distance}:
        force = diff.normalized() * ({strength} / dist)
    
    return force
"""
    return SimpleProgrammaticExpert(name, code)


def create_repel_expert(name: str, center_x: float, center_y: float,
                       strength: float = 30.0, min_distance: float = 150.0) -> SimpleProgrammaticExpert:
    """Create a repulsion expert from a fixed point."""
    code = f"""@ti.func
def {name}(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    pos = tv.p.field[i].pos
    center = ti.Vector([{center_x}, {center_y}])
    diff = pos - center
    dist = diff.norm()
    
    force = ti.Vector([0.0, 0.0])
    if 1.0 < dist < {min_distance}:
        force = diff.normalized() * ({strength} / (dist * 0.1))
    
    return force
"""
    return SimpleProgrammaticExpert(name, code)


def create_gravity_expert(name: str, gravity_strength: float = 9.8) -> SimpleProgrammaticExpert:
    """Create a gravity expert that pulls particles downward."""
    code = f"""@ti.func
def {name}(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Simple downward gravity
    return ti.Vector([0.0, {gravity_strength}])
"""
    return SimpleProgrammaticExpert(name, code)


def create_vortex_expert(name: str, center_x: float, center_y: float,
                        strength: float = 0.5, clockwise: bool = True) -> SimpleProgrammaticExpert:
    """Create a vortex/swirl expert around a point."""
    direction = -1.0 if clockwise else 1.0
    code = f"""@ti.func
def {name}(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    pos = tv.p.field[i].pos
    center = ti.Vector([{center_x}, {center_y}])
    diff = pos - center
    dist = diff.norm()
    
    force = ti.Vector([0.0, 0.0])
    if 10.0 < dist < 300.0:
        # Perpendicular to radial direction
        tangent = ti.Vector([-diff[1], diff[0]]).normalized()
        force = tangent * ({strength} * {direction})
    
    return force
"""
    return SimpleProgrammaticExpert(name, code)