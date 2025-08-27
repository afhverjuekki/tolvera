"""
Particle Life Simulation in Tölvera
A multi-species particle system with attraction/repulsion interactions
"""

import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    """Main function for Particle Life simulation."""
    # === Configuration ===
    # Set particle and species counts
    if 'species' not in kwargs:
        kwargs['species'] = 5  # 5 different species for interesting interactions
    if 'particles' not in kwargs:
        kwargs['particles'] = 1000  # More particles for rich emergence
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080
    
    # Create Tölvera instance
    tv = Tolvera(**kwargs)
    
    # === Particle Initialization ===
    @ti.kernel
    def init_particles_random():
        """Initialize particles randomly across the screen with species assignment."""
        particles_per_species = tv.pn // tv.sn
        remaining = tv.pn % tv.sn
        
        particle_idx = 0
        
        for species_id in range(tv.sn):
            count = particles_per_species + (1 if species_id < remaining else 0)
            
            for i in range(count):
                if particle_idx < tv.pn:
                    tv.p.field[particle_idx].active = 1.0
                    # Keep particles away from edges initially
                    tv.p.field[particle_idx].pos = ti.Vector([
                        100.0 + ti.random() * (tv.x - 200.0),
                        100.0 + ti.random() * (tv.y - 200.0)
                    ])
                    # Reduced initial velocity
                    tv.p.field[particle_idx].vel = ti.Vector([
                        (ti.random() - 0.5) * 50.0,
                        (ti.random() - 0.5) * 50.0
                    ])
                    tv.p.field[particle_idx].size = 3.0
                    tv.p.field[particle_idx].mass = 1.0
                    tv.p.field[particle_idx].speed = 1.0
                    tv.p.field[particle_idx].species = species_id
                    
                    particle_idx += 1
    
    init_particles_random()
    
    # Set species colors - rainbow spectrum
    colors = [
        [1.0, 0.2, 0.2, 1.0],  # Red
        [1.0, 0.7, 0.2, 1.0],  # Orange
        [0.2, 1.0, 0.2, 1.0],  # Green
        [0.2, 0.7, 1.0, 1.0],  # Cyan
        [0.5, 0.2, 1.0, 1.0],  # Purple
    ]
    
    for i in range(min(tv.sn, len(colors))):
        tv.s.species.field[i].rgba = colors[i]
    
    # === State Initialization ===
    # Global states for simulation parameters
    if 'llm_global' not in tv.s:
        tv.s.set('llm_global', {
            'state': {
                'max_force': (ti.f32, 10.0, 500.0),
                'damping': (ti.f32, 0.9, 0.999),
            },
            'shape': 1,
            'osc': ('get', 'set'),
            'randomise': False
        })
    
    # Species interaction matrix (2D for species-to-species interactions)
    if 'llm_species' not in tv.s:
        tv.s.set('llm_species', {
            'state': {
                'attraction_force': (ti.f32, -200.0, 200.0),  # Can be negative (repulsion) or positive (attraction)
                'interaction_radius': (ti.f32, 20.0, 200.0),
                'repulsion_radius': (ti.f32, 5.0, 30.0),
            },
            'shape': (tv.sn, tv.sn),  # 2D matrix for species interactions
            'osc': ('get', 'set'),
            'randomise': True
        })
    
    # Initialize global parameters
    tv.s.llm_global.field[0].max_force = 200.0
    tv.s.llm_global.field[0].damping = 0.98
    
    # Initialize species interaction matrix with interesting patterns
    @ti.kernel
    def init_interaction_matrix():
        for s1 in range(tv.sn):
            for s2 in range(tv.sn):
                # More controlled attraction/repulsion forces
                tv.s.llm_species.field[s1, s2].attraction_force = (ti.random() - 0.5) * 100.0
                
                # Interaction radius varies by species pair
                tv.s.llm_species.field[s1, s2].interaction_radius = 60.0 + ti.random() * 80.0
                
                # Close-range repulsion to prevent overlap
                tv.s.llm_species.field[s1, s2].repulsion_radius = 15.0 + ti.random() * 10.0
    
    init_interaction_matrix()
    
    # === Particle Force Experts ===
    @ti.func
    def particle_life_interaction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Calculate attraction/repulsion forces based on species interaction matrix."""
        force = ti.math.vec2(0.0, 0.0)
        
        for j in range(tv.pn):
            if particle_idx != j and tv.p.field[j].active > 0:
                other_pos = tv.p.field[j].pos
                other_species = tv.p.field[j].species
                
                # Get interaction parameters from the matrix
                attraction = tv.s.llm_species.field[species, other_species].attraction_force
                interaction_radius = tv.s.llm_species.field[species, other_species].interaction_radius
                repulsion_radius = tv.s.llm_species.field[species, other_species].repulsion_radius
                
                # Calculate distance and direction
                diff = other_pos - pos
                dist = diff.norm()
                
                # Declare force_magnitude and direction BEFORE conditionals
                direction = ti.math.vec2(0.0, 0.0)
                force_magnitude = 0.0
                
                if dist > 0.001 and dist < interaction_radius:
                    direction = diff / dist
                    
                    # Apply forces based on distance
                    if dist < repulsion_radius:
                        # Strong repulsion at close range
                        force_magnitude = -300.0 * (1.0 - dist / repulsion_radius)
                    else:
                        # Attraction/repulsion based on matrix
                        normalized_dist = (dist - repulsion_radius) / (interaction_radius - repulsion_radius)
                        # Smooth falloff with distance
                        force_magnitude = attraction * (1.0 - normalized_dist)
                    
                    force += direction * force_magnitude
        
        # Clamp total force to prevent instability
        max_force = tv.s.llm_global.field[0].max_force
        force_norm = force.norm()
        
        # Declare clamped_force BEFORE conditional
        clamped_force = force
        
        if force_norm > max_force and force_norm > 0.001:
            clamped_force = (force / force_norm) * max_force
        
        return clamped_force
    
    @ti.func
    def friction_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Apply velocity-dependent friction for stability."""
        friction_coefficient = 0.5
        force = -vel * friction_coefficient
        return force
    
    # === Integration Kernel ===
    @ti.kernel
    def apply_all_experts():
        """Main physics kernel that applies all forces and updates particles."""
        dt = 0.016  # 60 FPS timestep
        damping = tv.s.llm_global.field[0].damping
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                vel = tv.p.field[i].vel
                mass = tv.p.field[i].mass
                species = tv.p.field[i].species
                
                # Accumulate all forces
                total_force = ti.math.vec2(0.0, 0.0)
                
                # Main particle life interaction
                total_force += particle_life_interaction(pos, vel, mass, species, i)
            
                
                # Friction for stability
                total_force += friction_force(pos, vel, mass, species, i)
                
                # Declare acceleration BEFORE conditional
                acceleration = ti.math.vec2(0.0, 0.0)
                
                # Update velocity (F = ma)
                if mass > 0:
                    acceleration = total_force / mass
                else:
                    acceleration = total_force
                    
                tv.p.field[i].vel += acceleration * dt
                
                # Apply damping
                tv.p.field[i].vel *= damping
                
                # Update position
                tv.p.field[i].pos += tv.p.field[i].vel * tv.p.field[i].speed * dt
    
    # === Drawing Functions ===
    @ti.func
    def draw_interaction_lines(i: ti.i32):
        """Draw lines showing strong interactions between particles."""
        if tv.p.field[i].active > 0:
            pos_i = tv.p.field[i].pos
            species_i = tv.p.field[i].species
            
            for j in range(tv.pn):
                if i < j and tv.p.field[j].active > 0:  # Avoid duplicate lines
                    pos_j = tv.p.field[j].pos
                    species_j = tv.p.field[j].species
                    
                    dist = (pos_j - pos_i).norm()
                    interaction_radius = tv.s.llm_species.field[species_i, species_j].interaction_radius
                    
                    if dist < interaction_radius * 0.5:  # Only show close interactions
                        attraction = tv.s.llm_species.field[species_i, species_j].attraction_force
                        
                        # Declare color variable BEFORE conditional
                        color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                        
                        # Color based on attraction/repulsion
                        if attraction > 0:
                            # Green for attraction
                            color = ti.math.vec4(0.0, 0.3, 0.0, 0.1)
                        else:
                            # Red for repulsion
                            color = ti.math.vec4(0.3, 0.0, 0.0, 0.1)
                        
                        tv.px.line(pos_i.x, pos_i.y, pos_j.x, pos_j.y, color)
    
    # === Drawing Kernel ===
    @ti.kernel
    def draw_visuals():
        """Kernel for visual effects and drawing."""
        # Draw interaction lines for first 100 particles to avoid overload
        for i in range(min(100, tv.pn)):
            draw_interaction_lines(i)
    
    # === Render Loop ===
    @tv.render
    def _():
        tv.px.diffuse(0.95)  # Slight trail effect
        
        # Draw visual effects
        draw_visuals()
        
        # Apply physics
        apply_all_experts()
        tv.p()
        
        # Render particles
        tv.px.particles(tv.p, tv.s.species())
        
        return tv.px

if __name__ == "__main__":
    run(main)