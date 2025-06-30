"""Pre-built expert examples for common particle behaviors.

This module provides a library of example experts that demonstrate
various particle behaviors and can be used as templates or references.
"""

from typing import Dict, List
from .poe_core import SimpleProgrammaticExpert


class ExpertLibrary:
    """Library of pre-built behavior experts."""
    
    @staticmethod
    def get_expert(name: str) -> SimpleProgrammaticExpert:
        """Get a pre-built expert by name.
        
        Args:
            name: Expert name
            
        Returns:
            Expert instance
        """
        experts = ExpertLibrary.get_all_experts()
        if name in experts:
            return experts[name]
        raise ValueError(f"Unknown expert: {name}")
    
    @staticmethod
    def get_all_experts() -> Dict[str, SimpleProgrammaticExpert]:
        """Get all available pre-built experts.
        
        Returns:
            Dictionary mapping names to experts
        """
        experts = {}
        
        # Mouse attraction expert
        experts["mouse_attraction"] = SimpleProgrammaticExpert(
            name="mouse_attraction",
            code="""@ti.func
def mouse_attraction(particles: ti.template(), i: ti.i32, mouse_x: ti.f32, mouse_y: ti.f32) -> ti.math.vec2:
    '''Attract particles towards mouse position'''
    pos = particles[i].pos
    mouse_pos = ti.Vector([mouse_x, mouse_y])
    
    to_mouse = mouse_pos - pos
    dist = to_mouse.norm()
    
    force = ti.Vector([0.0, 0.0])
    if 0 < dist < 300.0:  # Influence radius
        # Inverse square law with cutoff
        strength = 50.0 / (dist * dist + 1.0)
        force = to_mouse.normalized() * strength
    
    return force""",
            weight=1.0
        )
        
        # Mouse repulsion expert
        experts["mouse_repulsion"] = SimpleProgrammaticExpert(
            name="mouse_repulsion",
            code="""@ti.func
def mouse_repulsion(particles: ti.template(), i: ti.i32, mouse_x: ti.f32, mouse_y: ti.f32) -> ti.math.vec2:
    '''Repel particles from mouse when too close'''
    pos = particles[i].pos
    mouse_pos = ti.Vector([mouse_x, mouse_y])
    
    from_mouse = pos - mouse_pos
    dist = from_mouse.norm()
    
    force = ti.Vector([0.0, 0.0])
    if 0 < dist < 50.0:  # Repulsion radius
        # Strong repulsion when very close
        strength = 100.0 / (dist + 1.0)
        force = from_mouse.normalized() * strength
    
    return force""",
            weight=1.0
        )
        
        # Boundary avoidance expert
        experts["boundary_avoidance"] = SimpleProgrammaticExpert(
            name="boundary_avoidance",
            code="""@ti.func
def boundary_avoidance(particles: ti.template(), i: ti.i32, width: ti.f32, height: ti.f32) -> ti.math.vec2:
    '''Keep particles away from screen edges'''
    pos = particles[i].pos
    margin = 50.0
    strength = 1.0
    
    force = ti.Vector([0.0, 0.0])
    
    # Horizontal boundaries
    if pos[0] < margin:
        force[0] = strength * (1.0 - pos[0] / margin)
    elif pos[0] > width - margin:
        force[0] = -strength * (1.0 - (width - pos[0]) / margin)
    
    # Vertical boundaries
    if pos[1] < margin:
        force[1] = strength * (1.0 - pos[1] / margin)
    elif pos[1] > height - margin:
        force[1] = -strength * (1.0 - (height - pos[1]) / margin)
    
    return force""",
            weight=0.5
        )
        
        # Particle separation expert
        experts["particle_separation"] = SimpleProgrammaticExpert(
            name="particle_separation",
            code="""@ti.func
def particle_separation(particles: ti.template(), i: ti.i32) -> ti.math.vec2:
    '''Maintain minimum distance between particles'''
    pos_i = particles[i].pos
    force = ti.Vector([0.0, 0.0])
    
    separation_radius = 30.0
    max_force = 0.5
    
    # Check nearby particles
    for j in range(particles.shape[0]):
        if i != j and particles[j].active > 0:
            pos_j = particles[j].pos
            diff = pos_i - pos_j
            dist = diff.norm()
            
            if 0 < dist < separation_radius:
                # Repulsion force
                repulsion = diff.normalized() * (1.0 - dist / separation_radius) * max_force
                force += repulsion
    
    return force""",
            weight=0.3
        )
        
        # Flocking alignment expert
        experts["flocking_alignment"] = SimpleProgrammaticExpert(
            name="flocking_alignment",
            code="""@ti.func
def flocking_alignment(particles: ti.template(), i: ti.i32) -> ti.math.vec2:
    '''Align velocity with nearby particles'''
    pos_i = particles[i].pos
    vel_i = particles[i].vel
    
    avg_vel = ti.Vector([0.0, 0.0])
    neighbor_count = 0
    neighbor_radius = 80.0
    
    # Find neighbors
    for j in range(particles.shape[0]):
        if i != j and particles[j].active > 0:
            pos_j = particles[j].pos
            dist = (pos_i - pos_j).norm()
            
            if dist < neighbor_radius:
                avg_vel += particles[j].vel
                neighbor_count += 1
    
    force = ti.Vector([0.0, 0.0])
    if neighbor_count > 0:
        avg_vel /= neighbor_count
        # Steering force towards average velocity
        desired_vel = avg_vel.normalized() * vel_i.norm()
        force = (desired_vel - vel_i) * 0.1
    
    return force""",
            weight=0.5
        )
        
        # Flocking cohesion expert
        experts["flocking_cohesion"] = SimpleProgrammaticExpert(
            name="flocking_cohesion",
            code="""@ti.func
def flocking_cohesion(particles: ti.template(), i: ti.i32) -> ti.math.vec2:
    '''Move towards center of nearby particles'''
    pos_i = particles[i].pos
    
    center_of_mass = ti.Vector([0.0, 0.0])
    neighbor_count = 0
    neighbor_radius = 100.0
    
    # Find neighbors
    for j in range(particles.shape[0]):
        if i != j and particles[j].active > 0:
            pos_j = particles[j].pos
            dist = (pos_i - pos_j).norm()
            
            if dist < neighbor_radius:
                center_of_mass += pos_j
                neighbor_count += 1
    
    force = ti.Vector([0.0, 0.0])
    if neighbor_count > 0:
        center_of_mass /= neighbor_count
        # Force towards center of mass
        to_center = center_of_mass - pos_i
        force = to_center * 0.01
    
    return force""",
            weight=0.3
        )
        
        # Random walk expert
        experts["random_walk"] = SimpleProgrammaticExpert(
            name="random_walk",
            code="""@ti.func
def random_walk(particles: ti.template(), i: ti.i32) -> ti.math.vec2:
    '''Add random perturbations to movement'''
    # Use particle index and position for pseudo-random seed
    pos = particles[i].pos
    seed = i + int(pos[0] * 100) + int(pos[1] * 100)
    
    # Simple pseudo-random based on seed
    angle = (seed * 0.1234 + ti.random()) * 6.28318
    strength = 0.2
    
    force = ti.Vector([
        ti.cos(angle) * strength,
        ti.sin(angle) * strength
    ])
    
    return force""",
            weight=0.1
        )
        
        # Circular orbit expert
        experts["circular_orbit"] = SimpleProgrammaticExpert(
            name="circular_orbit",
            code="""@ti.func
def circular_orbit(particles: ti.template(), i: ti.i32, center_x: ti.f32, center_y: ti.f32) -> ti.math.vec2:
    '''Create circular orbital motion around a center point'''
    pos = particles[i].pos
    center = ti.Vector([center_x, center_y])
    
    to_particle = pos - center
    dist = to_particle.norm()
    
    force = ti.Vector([0.0, 0.0])
    
    if dist > 10.0:  # Avoid singularity
        # Tangential force for circular motion
        tangent = ti.Vector([-to_particle[1], to_particle[0]]).normalized()
        
        # Centripetal force to maintain orbit
        radial = -to_particle.normalized() * 0.1
        
        # Combine forces
        force = tangent * 0.5 + radial
    
    return force""",
            weight=0.7
        )
        
        # Gravity well expert
        experts["gravity_well"] = SimpleProgrammaticExpert(
            name="gravity_well",
            code="""@ti.func
def gravity_well(particles: ti.template(), i: ti.i32, well_x: ti.f32, well_y: ti.f32) -> ti.math.vec2:
    '''Simulate gravitational attraction to a point'''
    pos = particles[i].pos
    well_pos = ti.Vector([well_x, well_y])
    
    to_well = well_pos - pos
    dist = to_well.norm()
    
    force = ti.Vector([0.0, 0.0])
    
    if dist > 5.0:  # Avoid singularity
        # Gravity with inverse square law
        g_constant = 100.0
        force = to_well.normalized() * (g_constant / (dist * dist))
        
        # Add some damping when very close
        if dist < 30.0:
            force *= (dist / 30.0)
    
    return force""",
            weight=0.8
        )
        
        # Velocity damping expert
        experts["velocity_damping"] = SimpleProgrammaticExpert(
            name="velocity_damping",
            code="""@ti.func
def velocity_damping(particles: ti.template(), i: ti.i32) -> ti.math.vec2:
    '''Apply damping force opposite to velocity'''
    vel = particles[i].vel
    damping_factor = 0.05
    
    # Force opposite to velocity
    force = -vel * damping_factor
    
    return force""",
            weight=0.2
        )
        
        # Add metadata to all experts
        for expert in experts.values():
            expert.metadata["category"] = ExpertLibrary._categorize_expert(expert.name)
            expert.metadata["description"] = ExpertLibrary._get_description(expert.name)
        
        return experts
    
    @staticmethod
    def _categorize_expert(name: str) -> str:
        """Categorize expert by name."""
        categories = {
            "mouse": ["mouse_attraction", "mouse_repulsion"],
            "boundary": ["boundary_avoidance"],
            "flocking": ["flocking_alignment", "flocking_cohesion", "particle_separation"],
            "motion": ["circular_orbit", "random_walk"],
            "physics": ["gravity_well", "velocity_damping"],
        }
        
        for category, names in categories.items():
            if name in names:
                return category
        return "other"
    
    @staticmethod
    def _get_description(name: str) -> str:
        """Get human-readable description for expert."""
        descriptions = {
            "mouse_attraction": "Particles are attracted to the mouse cursor",
            "mouse_repulsion": "Particles are repelled from the mouse when too close",
            "boundary_avoidance": "Particles avoid the edges of the screen",
            "particle_separation": "Particles maintain minimum distance from each other",
            "flocking_alignment": "Particles align their velocity with neighbors",
            "flocking_cohesion": "Particles move towards the center of nearby groups",
            "random_walk": "Particles move with random perturbations",
            "circular_orbit": "Particles orbit around a central point",
            "gravity_well": "Particles are attracted by gravity to a point",
            "velocity_damping": "Particle velocities are gradually reduced",
        }
        return descriptions.get(name, "Custom particle behavior")
    
    @staticmethod
    def get_categories() -> List[str]:
        """Get all available categories."""
        return ["mouse", "boundary", "flocking", "motion", "physics", "other"]
    
    @staticmethod
    def get_experts_by_category(category: str) -> List[SimpleProgrammaticExpert]:
        """Get all experts in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of experts in that category
        """
        all_experts = ExpertLibrary.get_all_experts()
        return [
            expert for expert in all_experts.values()
            if expert.metadata.get("category") == category
        ]
    
    @staticmethod
    def create_composite_behavior(
        expert_names: List[str],
        weights: List[float] = None
    ) -> List[SimpleProgrammaticExpert]:
        """Create a composite behavior from multiple experts.
        
        Args:
            expert_names: List of expert names to combine
            weights: Optional weights for each expert
            
        Returns:
            List of weighted experts
        """
        if weights is None:
            weights = [1.0] * len(expert_names)
            
        experts = []
        for name, weight in zip(expert_names, weights):
            expert = ExpertLibrary.get_expert(name)
            expert.weight = weight
            experts.append(expert)
            
        return experts