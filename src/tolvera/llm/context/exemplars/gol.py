"""
Game of Life Particles Simulation in Tölvera
A particle-based adaptation of Conway's Game of Life with continuous movement
"""

import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    """Main function for Game of Life particles simulation."""
    # === Configuration ===
    if 'species' not in kwargs:
        kwargs['species'] = 3  # Different life forms with varying rules
    if 'particles' not in kwargs:
        kwargs['particles'] = 2000  # Number of cells/particles
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080
    
    # Create Tölvera instance
    tv = Tolvera(**kwargs)
    
    # === Particle Initialization ===
    @ti.kernel
    def init_particles_grid():
        """Initialize particles in a grid-like pattern with some randomness."""
        # Create a rough grid with some noise
        grid_size = ti.cast(ti.sqrt(tv.pn), ti.i32)
        cell_width = tv.x / grid_size
        cell_height = tv.y / grid_size
        
        for i in range(tv.pn):
            # Calculate grid position
            grid_x = i % grid_size
            grid_y = i // grid_size
            
            # Add noise to position
            noise_x = (ti.random() - 0.5) * cell_width * 0.8
            noise_y = (ti.random() - 0.5) * cell_height * 0.8
            
            x = (grid_x + 0.5) * cell_width + noise_x
            y = (grid_y + 0.5) * cell_height + noise_y
        
            
            tv.p.field[i].pos = ti.Vector([x, y])
            
            # Small random velocity for drifting
            tv.p.field[i].vel = ti.Vector([
                (ti.random() - 0.5) * 10.0,
                (ti.random() - 0.5) * 10.0
            ])
            
            # Random initial alive/dead state (active field)
            tv.p.field[i].active = 1.0 if ti.random() > 0.6 else 0.0
            
            # Size based on state
            tv.p.field[i].size = 6.0 if tv.p.field[i].active > 0 else 2.0
            tv.p.field[i].mass = 1.0
            tv.p.field[i].speed = 0.5
            tv.p.field[i].species = i % tv.sn
    
    init_particles_grid()
    
    # Set species colors - different life forms
    colors = [
        [0.2, 1.0, 0.4, 1.0],  # Green life
        [1.0, 0.4, 0.2, 1.0],  # Red life
        [0.2, 0.6, 1.0, 1.0],  # Blue life
    ]
    
    for i in range(min(tv.sn, len(colors))):
        tv.s.species.field[i].rgba = colors[i]
    
    # === State Initialization ===
    # Global parameters
    if 'llm_global' not in tv.s:
        tv.s.set('llm_global', {
            'state': {
                'neighbor_radius': (ti.f32, 20.0, 100.0),
                'birth_threshold_min': (ti.i32, 2, 4),
                'birth_threshold_max': (ti.i32, 3, 5),
                'survival_threshold_min': (ti.i32, 2, 4),
                'survival_threshold_max': (ti.i32, 3, 5),
                'update_rate': (ti.f32, 0.0, 1.0),
            },
            'shape': 1,
            'osc': ('get', 'set'),
            'randomise': False
        })
    
    # Species-specific Game of Life rules
    if 'llm_species' not in tv.s:
        tv.s.set('llm_species', {
            'state': {
                'birth_min': (ti.i32, 2, 4),
                'birth_max': (ti.i32, 3, 5),
                'survival_min': (ti.i32, 1, 4),
                'survival_max': (ti.i32, 3, 6),
                'mutation_rate': (ti.f32, 0.0, 0.1),
            },
            'shape': tv.sn,
            'osc': ('get', 'set'),
            'randomise': False
        })
    
    # Per-particle state
    if 'llm_particle' not in tv.s:
        tv.s.set('llm_particle', {
            'state': {
                'alive_neighbors': (ti.i32, 0, 20),
                'next_state': (ti.f32, 0.0, 1.0),
                'age': (ti.f32, 0.0, 100.0),
                'energy': (ti.f32, 0.0, 100.0),
            },
            'shape': tv.pn,
            'osc': ('get',),
            'randomise': False
        })
    
    # Initialize global parameters
    tv.s.llm_global.field[0].neighbor_radius = 50.0
    tv.s.llm_global.field[0].update_rate = 0.1  # How often to update life/death
    
    # Initialize species with different Game of Life variants
    # Standard Conway's Game of Life
    tv.s.llm_species.field[0].birth_min = 3
    tv.s.llm_species.field[0].birth_max = 3
    tv.s.llm_species.field[0].survival_min = 2
    tv.s.llm_species.field[0].survival_max = 3
    tv.s.llm_species.field[0].mutation_rate = 0.01
    
    # HighLife variant
    tv.s.llm_species.field[1].birth_min = 3
    tv.s.llm_species.field[1].birth_max = 3
    tv.s.llm_species.field[1].survival_min = 2
    tv.s.llm_species.field[1].survival_max = 3
    tv.s.llm_species.field[1].mutation_rate = 0.02
    
    # Seeds variant
    tv.s.llm_species.field[2].birth_min = 2
    tv.s.llm_species.field[2].birth_max = 2
    tv.s.llm_species.field[2].survival_min = 2
    tv.s.llm_species.field[2].survival_max = 4
    tv.s.llm_species.field[2].mutation_rate = 0.03
    
    # === Utility Functions ===
    @ti.func
    def count_alive_neighbors(i: ti.i32) -> ti.i32:
        """Count alive neighbors within radius for particle i."""
        count = 0
        pos_i = tv.p.field[i].pos
        species_i = tv.p.field[i].species
        radius = tv.s.llm_global.field[0].neighbor_radius
        
        for j in range(tv.pn):
            if i != j:
                pos_j = tv.p.field[j].pos
                species_j = tv.p.field[j].species
                
                # Only count same species as neighbors
                if species_i == species_j:
                    dist = (pos_j - pos_i).norm()
                    if dist < radius and tv.p.field[j].active > 0.5:
                        count += 1
        
        return count
    
    @ti.func
    def calculate_next_state(i: ti.i32, neighbor_count: ti.i32) -> ti.f32:
        """Calculate next state based on Game of Life rules."""
        current_state = tv.p.field[i].active
        species = tv.p.field[i].species
        
        # Get species-specific rules
        birth_min = tv.s.llm_species.field[species].birth_min
        birth_max = tv.s.llm_species.field[species].birth_max
        survival_min = tv.s.llm_species.field[species].survival_min
        survival_max = tv.s.llm_species.field[species].survival_max
        
        # Declare next_state before conditionals
        next_state = 0.0
        
        if current_state > 0.5:
            # Cell is alive - check survival
            if neighbor_count >= survival_min and neighbor_count <= survival_max:
                next_state = 1.0  # Survive
            else:
                next_state = 0.0  # Die
        else:
            # Cell is dead - check birth
            if neighbor_count >= birth_min and neighbor_count <= birth_max:
                next_state = 1.0  # Birth
            else:
                next_state = 0.0  # Stay dead
        
        # Add mutation chance
        mutation_rate = tv.s.llm_species.field[species].mutation_rate
        if ti.random() < mutation_rate:
            next_state = 1.0 - next_state  # Flip state
        
        return next_state
    
    # === Utility Kernel ===
    @ti.kernel
    def update_life_states():
        """Update life states for all particles."""
        # First pass: count neighbors
        for i in range(tv.pn):
            neighbor_count = count_alive_neighbors(i)
            tv.s.llm_particle.field[i].alive_neighbors = neighbor_count
            tv.s.llm_particle.field[i].next_state = calculate_next_state(i, neighbor_count)
        
        # Second pass: update states
        for i in range(tv.pn):
            next_state = tv.s.llm_particle.field[i].next_state
            tv.p.field[i].active = next_state
            
            # Update visual properties based on state
            if next_state > 0.5:
                tv.p.field[i].size = 6.0
                tv.s.llm_particle.field[i].age += 1.0
                tv.s.llm_particle.field[i].energy = ti.min(100.0, tv.s.llm_particle.field[i].energy + 10.0)
            else:
                tv.p.field[i].size = 2.0
                tv.s.llm_particle.field[i].age = 0.0
                tv.s.llm_particle.field[i].energy = ti.max(0.0, tv.s.llm_particle.field[i].energy - 5.0)
    
    # === Force Experts ===
    @ti.func
    def attraction_to_life(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Dead cells slowly drift towards areas with life."""
        force = ti.math.vec2(0.0, 0.0)
        
        if tv.p.field[particle_idx].active < 0.5:  # Only for dead cells
            radius = tv.s.llm_global.field[0].neighbor_radius * 2.0
            center_of_life = ti.math.vec2(0.0, 0.0)
            life_count = 0
            
            for j in range(tv.pn):
                if particle_idx != j and tv.p.field[j].active > 0.5:
                    if tv.p.field[j].species == species:
                        other_pos = tv.p.field[j].pos
                        dist = (other_pos - pos).norm()
                        
                        if dist < radius:
                            center_of_life += other_pos
                            life_count += 1
            
            # Move towards center of nearby life
            if life_count > 0:
                center_of_life = center_of_life / ti.cast(life_count, ti.f32)
                direction = center_of_life - pos
                dir_norm = direction.norm()
                if dir_norm > 0.001:
                    force = (direction / dir_norm) * 20.0
        
        return force
    
    @ti.func
    def separation_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Gentle separation to prevent overlap."""
        force = ti.math.vec2(0.0, 0.0)
        min_dist = 15.0
        
        for j in range(tv.pn):
            if particle_idx != j:
                other_pos = tv.p.field[j].pos
                diff = pos - other_pos
                dist = diff.norm()
                
                if dist > 0.001 and dist < min_dist:
                    force += (diff / dist) * (1.0 - dist / min_dist) * 50.0
        
        return force
    
    @ti.func
    def drift_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Gentle random drift for organic movement."""
        angle = (ti.random() - 0.5 + particle_idx * 0.0001) * 2.0 * 3.14159
        magnitude = 5.0 if tv.p.field[particle_idx].active > 0.5 else 10.0
        force = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * magnitude
        return force
    
    # === Integration Kernel ===
    @ti.kernel
    def apply_all_experts():
        """Apply forces and update particle physics."""
        dt = 0.016
        damping = 0.95
        
        for i in range(tv.pn):
            pos = tv.p.field[i].pos
            vel = tv.p.field[i].vel
            mass = tv.p.field[i].mass
            species = tv.p.field[i].species
            
            # Accumulate forces
            total_force = ti.math.vec2(0.0, 0.0)
            
            total_force += attraction_to_life(pos, vel, mass, species, i)
            total_force += separation_force(pos, vel, mass, species, i)
            total_force += drift_force(pos, vel, mass, species, i)
            
            # Limit force
            force_norm = total_force.norm()
            max_force = 200.0
            limited_force = total_force
            if force_norm > max_force:
                limited_force = (total_force / force_norm) * max_force
            
            # Update velocity and position
            acceleration = limited_force / mass if mass > 0 else limited_force
            tv.p.field[i].vel += acceleration * dt
            tv.p.field[i].vel *= damping
            
            # Limit velocity
            vel_norm = tv.p.field[i].vel.norm()
            max_vel = 50.0
            if vel_norm > max_vel:
                tv.p.field[i].vel = (tv.p.field[i].vel / vel_norm) * max_vel
            
            tv.p.field[i].pos += tv.p.field[i].vel * tv.p.field[i].speed * dt
    
    # === Drawing Functions ===
    @ti.func
    def draw_life_connections(i: ti.i32):
        """Draw connections between nearby living cells."""
        if tv.p.field[i].active > 0.5:
            pos_i = tv.p.field[i].pos
            species_i = tv.p.field[i].species
            radius = tv.s.llm_global.field[0].neighbor_radius
            
            for j in range(tv.pn):
                if i < j and tv.p.field[j].active > 0.5:
                    if tv.p.field[j].species == species_i:
                        pos_j = tv.p.field[j].pos
                        dist = (pos_j - pos_i).norm()
                        
                        if dist < radius:
                            # Fade based on distance
                            alpha = (1.0 - dist / radius) * 0.2
                            color = tv.s.species.field[species_i].rgba * alpha
                            tv.px.line(pos_i.x, pos_i.y, pos_j.x, pos_j.y, color)
    
    @ti.func
    def draw_life_aura(i: ti.i32):
        """Draw an aura around old living cells."""
        if tv.p.field[i].active > 0.5:
            age = tv.s.llm_particle.field[i].age
            if age > 5.0:
                pos = tv.p.field[i].pos
                species = tv.p.field[i].species
                
                # Older cells get bigger auras
                aura_size = ti.cast(5.0 + age * 0.5, ti.i32)
                aura_size = ti.min(aura_size, 20)
                
                # Pulsing effect based on age
                pulse = ti.sin(age * 0.2) * 0.5 + 0.5
                alpha = 0.1 * pulse
                
                color = tv.s.species.field[species].rgba * alpha
                tv.px.circle(ti.cast(pos.x, ti.i32), ti.cast(pos.y, ti.i32), aura_size, color, 1)
    
    # === Drawing Kernel ===
    @ti.kernel
    def draw_visuals():
        """Draw visual effects."""
        # Draw connections between living cells
        for i in range(ti.min(tv.pn, 500)):  # Limit for performance
            draw_life_connections(i)
        
        # Draw auras for old cells
        for i in range(tv.pn):
            draw_life_aura(i)
    
    # Frame counter for update timing
    frame_count = ti.field(ti.i32, shape=())
    frame_count[None] = 0
    
    # === Render Loop ===
    @tv.render
    def _():
        tv.px.diffuse(0.97)  # Fade effect
        
        # Update life states periodically
        frame_count[None] += 1
        if frame_count[None] % 10 == 0:  # Update every 10 frames
            update_life_states()
        
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