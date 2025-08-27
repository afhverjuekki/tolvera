"""
Boids Flocking Simulation in Tölvera
Classic artificial life algorithm demonstrating emergent flocking behavior
"""

import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    """Main function for Boids flocking simulation."""
    # === Configuration ===
    if 'species' not in kwargs:
        kwargs['species'] = 3  # Different flocks with different behaviors
    if 'particles' not in kwargs:
        kwargs['particles'] = 800  # Number of boids
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080
    
    # Create Tölvera instance
    tv = Tolvera(**kwargs)
    
    # === Particle Initialization ===
    @ti.kernel
    def init_particles_flocks():
        """Initialize boids in separate flocks."""
        particles_per_species = tv.pn // tv.sn
        remaining = tv.pn % tv.sn
        
        particle_idx = 0
        
        for species_id in range(tv.sn):
            count = particles_per_species + (1 if species_id < remaining else 0)
            
            # Create a clustered starting position for each flock
            cluster_center = ti.Vector([
                200.0 + (tv.x - 400.0) * ti.random(),
                200.0 + (tv.y - 400.0) * ti.random()
            ])
            
            for i in range(count):
                if particle_idx < tv.pn:
                    # Position boids in a cluster
                    angle = ti.random() * 2.0 * 3.14159
                    radius = ti.random() * 100.0
                    
                    tv.p.field[particle_idx].active = 1.0
                    tv.p.field[particle_idx].pos = cluster_center + ti.Vector([
                        ti.cos(angle) * radius,
                        ti.sin(angle) * radius
                    ])
                    
                    # Give initial velocity in random direction
                    vel_angle = ti.random() * 2.0 * 3.14159
                    vel_mag = 50.0 + ti.random() * 50.0
                    tv.p.field[particle_idx].vel = ti.Vector([
                        ti.cos(vel_angle) * vel_mag,
                        ti.sin(vel_angle) * vel_mag
                    ])
                    
                    tv.p.field[particle_idx].size = 4.0
                    tv.p.field[particle_idx].mass = 1.0
                    tv.p.field[particle_idx].speed = 1.0
                    tv.p.field[particle_idx].species = species_id
                    
                    particle_idx += 1
    
    init_particles_flocks()
    
    # Set species colors
    colors = [
        [0.9, 0.7, 0.2, 1.0],
        [0.8, 0.2, 0.8, 1.0],
        [0.2, 0.7, 0.9, 1.0],
    ]
    
    for i in range(min(tv.sn, len(colors))):
        tv.s.species.field[i].rgba = colors[i]
    
    # === State Initialization ===
    # Global parameters for flocking behavior
    if 'llm_global' not in tv.s:
        tv.s.set('llm_global', {
            'state': {
                'perception_radius': (ti.f32, 20.0, 200.0),
                'separation_radius': (ti.f32, 10.0, 50.0),
                'max_speed': (ti.f32, 50.0, 300.0),
                'max_force': (ti.f32, 100.0, 500.0),
            },
            'shape': 1,
            'osc': ('get', 'set'),
            'randomise': False
        })
    
    # Species-specific flocking parameters
    if 'llm_species' not in tv.s:
        tv.s.set('llm_species', {
            'state': {
                'separation_weight': (ti.f32, 0.5, 3.0),
                'alignment_weight': (ti.f32, 0.5, 2.0),
                'cohesion_weight': (ti.f32, 0.5, 2.0),
                'inter_species_avoidance': (ti.f32, 0.0, 5.0),
            },
            'shape': tv.sn,
            'osc': ('get', 'set'),
            'randomise': True
        })
    
    # Per-particle neighbor tracking
    if 'llm_particles' not in tv.s:
        tv.s.set('llm_particles', {
            'state': {
                'neighbor_count': (ti.i32, 0, 50),
                'flock_heading': (ti.math.vec2, -1.0, 1.0),
            },
            'shape': tv.pn,
            'osc': ('get',),
            'randomise': False
        })
    
    # Initialize global parameters
    tv.s.llm_global.field[0].perception_radius = 80.0
    tv.s.llm_global.field[0].separation_radius = 30.0
    tv.s.llm_global.field[0].max_speed = 200.0
    tv.s.llm_global.field[0].max_force = 300.0
    
    # Initialize species parameters with variation
    @ti.kernel
    def init_species_params():
        for s in range(tv.sn):
            # Each species has slightly different flocking behavior
            tv.s.llm_species.field[s].separation_weight = 1.5 + ti.random() * 0.5
            tv.s.llm_species.field[s].alignment_weight = 1.0 + ti.random() * 0.5
            tv.s.llm_species.field[s].cohesion_weight = 1.0 + ti.random() * 0.5
            tv.s.llm_species.field[s].inter_species_avoidance = 2.0 + ti.random() * 1.0
    
    init_species_params()
    
    # === Particle Force Experts ===
    @ti.func
    def separation_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Steer to avoid crowding local flockmates."""
        force = ti.math.vec2(0.0, 0.0)
        separation_radius = tv.s.llm_global.field[0].separation_radius
        count = 0
        
        for j in range(tv.pn):
            if particle_idx != j and tv.p.field[j].active > 0:
                other_pos = tv.p.field[j].pos
                diff = pos - other_pos
                dist = diff.norm()
                
                # Declare normalized_diff before conditional
                normalized_diff = ti.math.vec2(0.0, 0.0)
                
                if dist > 0.001 and dist < separation_radius:
                    # Repel from nearby boids
                    normalized_diff = diff / dist
                    # Weight by inverse distance (closer = stronger repulsion)
                    force += normalized_diff / dist
                    count += 1
        
        # Normalize and apply species weight
        result_force = ti.math.vec2(0.0, 0.0)
        if count > 0:
            force = force / ti.cast(count, ti.f32)
            force_norm = force.norm()
            if force_norm > 0.001:
                # Normalize and scale
                force = (force / force_norm) * tv.s.llm_global.field[0].max_speed
                # Apply steering force
                result_force = force - vel
                # Apply species-specific weight
                result_force *= tv.s.llm_species.field[species].separation_weight
        
        return result_force
    
    @ti.func
    def alignment_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Steer towards the average heading of local flockmates."""
        force = ti.math.vec2(0.0, 0.0)
        avg_vel = ti.math.vec2(0.0, 0.0)
        perception_radius = tv.s.llm_global.field[0].perception_radius
        count = 0
        
        for j in range(tv.pn):
            if particle_idx != j and tv.p.field[j].active > 0:
                other_species = tv.p.field[j].species
                
                # Only align with same species
                if other_species == species:
                    other_pos = tv.p.field[j].pos
                    dist = (other_pos - pos).norm()
                    
                    if dist > 0.001 and dist < perception_radius:
                        avg_vel += tv.p.field[j].vel
                        count += 1
        
        # Calculate steering force
        result_force = ti.math.vec2(0.0, 0.0)
        if count > 0:
            avg_vel = avg_vel / ti.cast(count, ti.f32)
            avg_vel_norm = avg_vel.norm()
            if avg_vel_norm > 0.001:
                # Normalize to max speed
                avg_vel = (avg_vel / avg_vel_norm) * tv.s.llm_global.field[0].max_speed
                # Calculate steering force
                result_force = avg_vel - vel
                # Apply species-specific weight
                result_force *= tv.s.llm_species.field[species].alignment_weight
        
        return result_force
    
    @ti.func
    def cohesion_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Steer to move toward the average position of local flockmates."""
        force = ti.math.vec2(0.0, 0.0)
        center = ti.math.vec2(0.0, 0.0)
        perception_radius = tv.s.llm_global.field[0].perception_radius
        count = 0
        
        for j in range(tv.pn):
            if particle_idx != j and tv.p.field[j].active > 0:
                other_species = tv.p.field[j].species
                
                # Only cohere with same species
                if other_species == species:
                    other_pos = tv.p.field[j].pos
                    dist = (other_pos - pos).norm()
                    
                    if dist > 0.001 and dist < perception_radius:
                        center += other_pos
                        count += 1
        
        # Steer towards center of mass
        result_force = ti.math.vec2(0.0, 0.0)
        if count > 0:
            center = center / ti.cast(count, ti.f32)
            # Desired velocity towards center
            desired = center - pos
            desired_norm = desired.norm()
            if desired_norm > 0.001:
                # Normalize and scale to max speed
                desired = (desired / desired_norm) * tv.s.llm_global.field[0].max_speed
                # Steering force
                result_force = desired - vel
                # Apply species-specific weight
                result_force *= tv.s.llm_species.field[species].cohesion_weight
        
        return result_force
    
    @ti.func
    def inter_species_avoidance(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Avoid boids from other species."""
        force = ti.math.vec2(0.0, 0.0)
        avoidance_radius = tv.s.llm_global.field[0].perception_radius * 0.7
        count = 0
        
        for j in range(tv.pn):
            if particle_idx != j and tv.p.field[j].active > 0:
                other_species = tv.p.field[j].species
                
                # Only avoid different species
                if other_species != species:
                    other_pos = tv.p.field[j].pos
                    diff = pos - other_pos
                    dist = diff.norm()
                    
                    # Declare normalized_diff before conditional
                    normalized_diff = ti.math.vec2(0.0, 0.0)
                    
                    if dist > 0.001 and dist < avoidance_radius:
                        normalized_diff = diff / dist
                        # Stronger avoidance for different species
                        force += normalized_diff / dist
                        count += 1
        
        # Apply inter-species avoidance weight
        result_force = ti.math.vec2(0.0, 0.0)
        if count > 0:
            force = force / ti.cast(count, ti.f32)
            result_force = force * tv.s.llm_species.field[species].inter_species_avoidance * 100.0
        
        return result_force
    
    @ti.func
    def wander_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Add slight randomness for more organic movement."""
        # Use particle index for consistent randomness
        angle = (ti.random() - 0.5 + particle_idx * 0.001) * 0.5
        wander = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * 20.0
        return wander
    
    # === Integration Kernel ===
    @ti.kernel
    def apply_all_experts():
        """Apply flocking behaviors to all boids."""
        dt = 0.032  # Adjusted for faster simulation
        max_force = tv.s.llm_global.field[0].max_force
        max_speed = tv.s.llm_global.field[0].max_speed
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                vel = tv.p.field[i].vel
                mass = tv.p.field[i].mass
                species = tv.p.field[i].species
                
                # Accumulate all forces
                total_force = ti.math.vec2(0.0, 0.0)
                
                # Classic boid behaviors
                total_force += separation_force(pos, vel, mass, species, i)
                total_force += alignment_force(pos, vel, mass, species, i)
                total_force += cohesion_force(pos, vel, mass, species, i)
                
                # Additional behaviors
                total_force += inter_species_avoidance(pos, vel, mass, species, i)
                total_force += wander_force(pos, vel, mass, species, i)
                
                # Limit total force
                force_norm = total_force.norm()
                limited_force = total_force
                if force_norm > max_force:
                    limited_force = (total_force / force_norm) * max_force
                
                # Apply force to velocity
                acceleration = limited_force / mass if mass > 0 else limited_force
                new_vel = vel + acceleration * dt
                
                # Limit speed
                vel_norm = new_vel.norm()
                if vel_norm > max_speed:
                    new_vel = (new_vel / vel_norm) * max_speed
                
                # Update velocity and position
                tv.p.field[i].vel = new_vel
                tv.p.field[i].pos += new_vel * tv.p.field[i].speed * dt
    
    # === Drawing Functions ===
    @ti.func
    def draw_boid_trails(i: ti.i32):
        """Draw velocity trails for boids."""
        if tv.p.field[i].active > 0:
            pos = tv.p.field[i].pos
            vel = tv.p.field[i].vel
            species = tv.p.field[i].species
            
            # Draw a line showing velocity direction
            vel_norm = vel.norm()
            if vel_norm > 0.001:
                # Scale velocity for visualization
                vel_scaled = (vel / vel_norm) * 20.0
                end_pos = pos - vel_scaled  # Trail behind boid
                
                # Faded color for trail
                color = tv.s.species.field[species].rgba * 0.3
                tv.px.line(pos.x, pos.y, end_pos.x, end_pos.y, color)
    
    # === Drawing Kernel ===
    @ti.kernel
    def draw_visuals():
        """Draw visual effects for boids."""
        # Draw trails for all boids
        for i in range(tv.pn):
            draw_boid_trails(i)
    
    # === Render Loop ===
    @tv.render
    def _():
        tv.px.diffuse(0.98)  # Slight trail effect
        
        # Draw visual effects
        draw_visuals()
        
        # Apply flocking behaviors
        apply_all_experts()
        tv.p()
        
        # Render particles as triangles pointing in velocity direction
        tv.px.particles(tv.p, tv.s.species())
        
        return tv.px

if __name__ == "__main__":
    run(main)