"""Integration layer connecting PoE system with Tölvera.

This module provides the TolveraBehaviorAgent class that integrates the
PoE behavior system with Tölvera's particle system and render loop.
"""

import taichi as ti
import numpy as np
from typing import Dict, Any, Optional, List, Callable
import logging
import asyncio

from .poe_core import PoEBehaviorSystem, SimpleProgrammaticExpert

logger = logging.getLogger(__name__)


class TolveraBehaviorAgent:
    """Integrates PoE behavior system with Tölvera's update cycle.
    
    This class manages the connection between the abstract PoE expert
    system and Tölvera's concrete particle implementation.
    """
    
    def __init__(self, tolvera_instance):
        """Initialize the behavior agent.
        
        Args:
            tolvera_instance: Tölvera instance to attach to
        """
        self.tv = tolvera_instance
        self.poe_system = PoEBehaviorSystem(tolvera_instance)
        
        # Runtime context for experts
        self.context = {
            "mouse_x": 0.0,
            "mouse_y": 0.0,
            "mouse_pressed": False,
            "time": 0.0,
            "dt": 0.016,
            "world_width": float(tolvera_instance.x),
            "world_height": float(tolvera_instance.y),
        }
        
        # Create behavior state in Tölvera
        self._create_behavior_state()
        
        # Setup integration kernels
        self._setup_kernels()
        
        # Track if we need to recompile combined kernel
        self._needs_recompile = True
        
        # Initialize builtin expert flags
        self._enable_mouse_attraction = False
        self._enable_boundary = False
        self._enable_separation = False
        self._mouse_attraction_weight = 1.0
        self._boundary_weight = 1.0
        self._separation_weight = 1.0
        
        logger.info("Initialized TolveraBehaviorAgent")
    
    def _create_behavior_state(self):
        """Create behavior-specific state in Tölvera's state system."""
        try:
            self.tv.s.behavior = {
                "state": {
                    "mouse_x": (ti.f32, 0.0, self.context["world_width"]),
                    "mouse_y": (ti.f32, 0.0, self.context["world_height"]),
                    "time": (ti.f32, 0.0, 1000000.0),
                    "active": (ti.i32, 0, 1)
                },
                "shape": 1,
                "osc": ("set", "get"),
                "randomise": False
            }
            logger.debug("Created behavior state")
        except Exception as e:
            logger.error(f"Failed to create behavior state: {e}")
            raise
    
    def _setup_kernels(self):
        """Setup Taichi kernels for force application."""
        
        @ti.kernel
        def apply_mouse_attraction(
            particles: ti.template(),
            forces: ti.template(),
            mouse_x: ti.f32,
            mouse_y: ti.f32,
            strength: ti.f32,
            radius: ti.f32
        ):
            """Apply mouse attraction force to particles."""
            mouse_pos = ti.Vector([mouse_x, mouse_y])
            
            for i in particles:
                if particles[i].active == 0:
                    continue
                    
                pos = particles[i].pos
                to_mouse = mouse_pos - pos
                dist = to_mouse.norm()
                
                if dist > 1.0 and dist < radius:
                    # Linear attraction force
                    force = to_mouse.normalized() * (strength / dist)
                    forces[i] += force
        
        @ti.kernel
        def apply_boundary_repulsion(
            particles: ti.template(),
            forces: ti.template(),
            world_width: ti.f32,
            world_height: ti.f32,
            margin: ti.f32,
            strength: ti.f32
        ):
            """Apply boundary repulsion to keep particles in bounds."""
            for i in particles:
                if particles[i].active == 0:
                    continue
                    
                pos = particles[i].pos
                force = ti.Vector([0.0, 0.0])
                
                # Left/right boundaries
                if pos[0] < margin:
                    force[0] += strength * (margin - pos[0]) / margin
                elif pos[0] > world_width - margin:
                    force[0] -= strength * (pos[0] - (world_width - margin)) / margin
                
                # Top/bottom boundaries
                if pos[1] < margin:
                    force[1] += strength * (margin - pos[1]) / margin
                elif pos[1] > world_height - margin:
                    force[1] -= strength * (pos[1] - (world_height - margin)) / margin
                    
                forces[i] += force
        
        @ti.kernel
        def apply_particle_separation(
            particles: ti.template(),
            forces: ti.template(),
            separation_dist: ti.f32,
            strength: ti.f32
        ):
            """Apply separation force between particles."""
            for i in particles:
                if particles[i].active == 0:
                    continue
                    
                pos_i = particles[i].pos
                
                for j in range(i + 1, particles.shape[0]):
                    if particles[j].active == 0:
                        continue
                        
                    pos_j = particles[j].pos
                    diff = pos_i - pos_j
                    dist = diff.norm()
                    
                    if 0 < dist < separation_dist:
                        # Repulsion force
                        force = diff.normalized() * (strength / dist)
                        forces[i] += force
                        forces[j] -= force  # Newton's third law
        
        # Store kernel references
        self.kernel_mouse_attraction = apply_mouse_attraction
        self.kernel_boundary_repulsion = apply_boundary_repulsion
        self.kernel_particle_separation = apply_particle_separation
        
        @ti.kernel
        def update_particle_positions(particles: ti.template(), dt: ti.f32):
            """Update particle positions based on their velocities."""
            for i in particles:
                if particles[i].active > 0:
                    particles[i].pos += particles[i].vel * dt
        
        self.kernel_update_positions = update_particle_positions
    
    def add_expert_from_code(self, name: str, code: str, weight: float = 1.0) -> bool:
        """Add an expert from code string.
        
        Args:
            name: Expert name
            code: Python code defining the expert
            weight: Expert weight
            
        Returns:
            True if expert was added successfully
        """
        expert = SimpleProgrammaticExpert(name, code, weight)
        if expert.compile(self.poe_system.namespace):
            if self.poe_system.add_expert(expert):
                self._needs_recompile = True
                return True
        return False
    
    def add_builtin_expert(self, expert_type: str, weight: float = 1.0) -> bool:
        """Add a built-in expert type.
        
        Args:
            expert_type: Type of expert ("mouse_attraction", "boundary", "separation")
            weight: Expert weight
            
        Returns:
            True if expert was added
        """
        # For now, we'll use a flag-based approach since dynamic kernel
        # compilation is challenging in Taichi
        if expert_type == "mouse_attraction":
            self._enable_mouse_attraction = True
            self._mouse_attraction_weight = weight
            logger.info(f"Enabled mouse attraction (weight: {weight})")
            return True
        elif expert_type == "boundary":
            self._enable_boundary = True
            self._boundary_weight = weight
            logger.info(f"Enabled boundary repulsion (weight: {weight})")
            return True
        elif expert_type == "separation":
            self._enable_separation = True
            self._separation_weight = weight
            logger.info(f"Enabled particle separation (weight: {weight})")
            return True
        else:
            logger.warning(f"Unknown expert type: {expert_type}")
            return False
    
    def update_context(self):
        """Update the runtime context for experts."""
        # Update mouse position
        if hasattr(self.tv.ctx, 'i') and hasattr(self.tv.ctx, 'gui'):
            try:
                # Get mouse position from Tölvera context (already in pixel coordinates)
                mouse_x = self.tv.ctx.i.field[0].x
                mouse_y = self.tv.ctx.i.field[0].y
                
                # Only update if we have valid mouse coordinates
                if mouse_x >= 0 and mouse_y >= 0:
                    self.context["mouse_x"] = float(mouse_x)
                    self.context["mouse_y"] = float(mouse_y)
                
                # Check mouse button press
                self.context["mouse_pressed"] = self.tv.ctx.i.field[0].s == 1
                
                # Debug logging
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 0
                    
                if self._debug_counter % 60 == 0:  # Log every second at 60fps
                    logger.debug(f"Mouse pos: ({self.context['mouse_x']:.1f}, {self.context['mouse_y']:.1f})")
            except Exception as e:
                # Try alternate method using Taichi window
                try:
                    if hasattr(self.tv, 'ti') and hasattr(self.tv.ti, 'window'):
                        mouse_pos = self.tv.ti.window.get_cursor_pos()
                        # Taichi's get_cursor_pos() returns normalized coordinates (0-1)
                        self.context["mouse_x"] = mouse_pos[0] * self.tv.x
                        self.context["mouse_y"] = mouse_pos[1] * self.tv.y  # Don't invert Y
                        self.context["mouse_pressed"] = self.tv.ti.window.is_pressed(ti.ui.LMB)
                except Exception as e2:
                    # Fallback - just keep previous values
                    pass
        
        # Update time
        self.context["time"] += self.context["dt"]
        
        # Update behavior state
        if hasattr(self.tv.s, 'behavior'):
            self.tv.s.behavior.field[0].mouse_x = self.context["mouse_x"]
            self.tv.s.behavior.field[0].mouse_y = self.context["mouse_y"]
            self.tv.s.behavior.field[0].time = self.context["time"]
    
    def apply_expert_forces(self):
        """Apply all active expert forces to particles.
        
        This is the main method called during the render loop to
        compute and apply behavior forces.
        """
        # Clear force accumulator
        self.poe_system.clear_forces()
        
        # Apply built-in experts (simplified approach for now)
        if hasattr(self, '_enable_mouse_attraction') and self._enable_mouse_attraction:
            # Log only occasionally to avoid spam
            if hasattr(self, '_debug_counter') and self._debug_counter % 120 == 0:
                logger.debug(f"Applying mouse attraction at ({self.context['mouse_x']:.1f}, {self.context['mouse_y']:.1f})")
            
            self.kernel_mouse_attraction(
                self.tv.p.field,
                self.poe_system.force_accumulator,
                self.context["mouse_x"],
                self.context["mouse_y"],
                strength=5.0 * self._mouse_attraction_weight,  # Reasonable strength
                radius=300.0  # Reasonable radius
            )
        
        if hasattr(self, '_enable_boundary') and self._enable_boundary:
            self.kernel_boundary_repulsion(
                self.tv.p.field,
                self.poe_system.force_accumulator,
                self.context["world_width"],
                self.context["world_height"],
                margin=50.0,
                strength=0.5 * self._boundary_weight
            )
        
        if hasattr(self, '_enable_separation') and self._enable_separation:
            self.kernel_particle_separation(
                self.tv.p.field,
                self.poe_system.force_accumulator,
                separation_dist=30.0,
                strength=0.1 * self._separation_weight
            )
        
        # Apply custom experts
        # Note: This is where we apply dynamically generated experts
        if len(self.poe_system.experts) > 0:
            logger.debug(f"Applying {len(self.poe_system.experts)} custom experts")
            self.poe_system.compute_expert_forces(self.context)
    
    def update(self, dt: float = 0.016):
        """Main update method called in Tölvera render loop.
        
        Args:
            dt: Time step
        """
        # Update context
        self.context["dt"] = dt
        self.update_context()
        
        # Apply expert forces
        self.apply_expert_forces()
        
        # Apply forces to particles
        self.poe_system.apply_forces_to_particles(dt, damping=0.98)
        
        # Update positions based on velocities
        self.kernel_update_positions(self.tv.p.field, dt)
        
        # Debug: Check if any forces were applied (only occasionally)
        if hasattr(self, '_debug_counter') and self._debug_counter % 300 == 0:  # Every 5 seconds at 60fps
            # Sample a few particles to check their velocities
            vel_sum = 0.0
            for i in range(min(10, self.tv.pn)):
                if self.tv.p.field[i].active > 0:
                    vel = self.tv.p.field[i].vel
                    vel_sum += (vel[0]**2 + vel[1]**2)**0.5
            if vel_sum > 0.01:
                logger.debug(f"Particles moving! Avg velocity: {vel_sum/10:.4f}")
        
        # NOTE: Don't call tv.p.update() as it may override our force applications
        # Let the render loop handle position updates
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status.
        
        Returns:
            Status dictionary
        """
        status = {
            "context": self.context,
            "poe_system": self.poe_system.get_status(),
            "builtin_experts": {
                "mouse_attraction": getattr(self, '_enable_mouse_attraction', False),
                "boundary": getattr(self, '_enable_boundary', False),
                "separation": getattr(self, '_enable_separation', False),
            }
        }
        return status


class AsyncTolveraBehaviorAgent(TolveraBehaviorAgent):
    """Async version of behavior agent for LLM integration.
    
    This version supports async operations for expert synthesis
    and other LLM-based features.
    """
    
    def __init__(self, tolvera_instance):
        """Initialize async behavior agent."""
        super().__init__(tolvera_instance)
        self.synthesis_queue = asyncio.Queue()
        self.synthesis_task = None
    
    async def add_expert_from_description(
        self, 
        description: str, 
        synthesizer,
        weight: float = 1.0
    ) -> bool:
        """Add an expert from natural language description.
        
        Args:
            description: Natural language behavior description
            synthesizer: Expert synthesizer instance
            weight: Expert weight
            
        Returns:
            True if expert was added successfully
        """
        try:
            # Generate expert code
            expert = await synthesizer.synthesize_expert(description)
            
            if expert:
                expert.weight = weight
                if self.poe_system.add_expert(expert):
                    self._needs_recompile = True
                    logger.info(f"Added expert from description: {expert.name}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to add expert from description: {e}")
            
        return False
    
    async def batch_add_experts(
        self,
        descriptions: List[str],
        synthesizer,
        weights: Optional[List[float]] = None
    ) -> List[bool]:
        """Add multiple experts from descriptions.
        
        Args:
            descriptions: List of behavior descriptions
            synthesizer: Expert synthesizer instance
            weights: Optional list of weights
            
        Returns:
            List of success flags
        """
        if weights is None:
            weights = [1.0] * len(descriptions)
            
        results = []
        for desc, weight in zip(descriptions, weights):
            success = await self.add_expert_from_description(desc, synthesizer, weight)
            results.append(success)
            
        return results