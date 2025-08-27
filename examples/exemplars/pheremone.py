"""
Ant Colony Simulation in Tölvera
Ants forage for food using pheromone trails, demonstrating stigmergic communication
"""

import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    """Main function for Ant Colony simulation."""
    # === Configuration ===
    if 'species' not in kwargs:
        kwargs['species'] = 3  # Different ant colonies
    if 'particles' not in kwargs:
        kwargs['particles'] = 1200  # Number of ants
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080
    
    # Create Tölvera instance
    tv = Tolvera(**kwargs)
    
    # === Pheromone Fields ===
    # Two pheromone types: to-food and to-home
    pheromone_to_food = ti.field(dtype=ti.f32, shape=(tv.x, tv.y))
    pheromone_to_home = ti.field(dtype=ti.f32, shape=(tv.x, tv.y))
    
    # === Food Sources Field ===
    food_sources = ti.field(dtype=ti.f32, shape=(tv.x, tv.y))
    
    # === Particle Initialization ===
    @ti.kernel
    def init_particles_colonies():
        """Initialize ants at their colony homes."""
        particles_per_species = tv.pn // tv.sn
        remaining = tv.pn % tv.sn
        
        particle_idx = 0
        
        for species_id in range(tv.sn):
            count = particles_per_species + (1 if species_id < remaining else 0)
            
            # Each colony has a home position
            home_x = 200.0 + species_id * ((tv.x - 400.0) / max(tv.sn - 1, 1))
            home_y = tv.y - 200.0  # Colonies at bottom
            
            for i in range(count):
                if particle_idx < tv.pn:
                    # Start ants near their home
                    angle = ti.random() * 2.0 * 3.14159
                    radius = ti.random() * 30.0
                    
                    tv.p.field[particle_idx].active = 1.0
                    tv.p.field[particle_idx].pos = ti.Vector([
                        home_x + ti.cos(angle) * radius,
                        home_y + ti.sin(angle) * radius
                    ])
                    
                    # Random initial direction
                    heading = ti.random() * 2.0 * 3.14159
                    speed = 80.0
                    tv.p.field[particle_idx].vel = ti.Vector([
                        ti.cos(heading) * speed,
                        ti.sin(heading) * speed
                    ])
                    
                    tv.p.field[particle_idx].size = 3.0
                    tv.p.field[particle_idx].mass = 1.0
                    tv.p.field[particle_idx].speed = 1.0
                    tv.p.field[particle_idx].species = species_id
                    
                    particle_idx += 1
    
    init_particles_colonies()
    
    # Set species colors - different colonies
    colors = [
        [0.8, 0.2, 0.2, 1.0],  # Red colony
        [0.2, 0.8, 0.2, 1.0],  # Green colony
        [0.2, 0.2, 0.8, 1.0],  # Blue colony
    ]
    
    for i in range(min(tv.sn, len(colors))):
        tv.s.species.field[i].rgba = colors[i]
    
    # === Initialize Food Sources ===
    @ti.kernel
    def init_food_sources():
        """Place food sources in the environment."""
        # Create several food patches
        num_patches = 5
        for patch in range(num_patches):
            # Random position for each food patch
            center_x = 200.0 + ti.random() * (tv.x - 400.0)
            center_y = 100.0 + ti.random() * (tv.y - 400.0)
            
            # Create circular food patch
            patch_radius = 30.0 + ti.random() * 30.0
            
            for dx in range(-100, 101):
                for dy in range(-100, 101):
                    x = ti.cast(center_x + dx, ti.i32)
                    y = ti.cast(center_y + dy, ti.i32)
                    
                    if x >= 0 and x < tv.x and y >= 0 and y < tv.y:
                        dist = ti.sqrt(ti.cast(dx * dx + dy * dy, ti.f32))
                        if dist < patch_radius:
                            # Food amount decreases from center
                            food_sources[x, y] = 100.0 * (1.0 - dist / patch_radius)
    
    init_food_sources()
    
    # === State Initialization ===
    # Global parameters
    if 'llm_global' not in tv.s:
        tv.s.set('llm_global', {
            'state': {
                'pheromone_decay': (ti.f32, 0.95, 0.999),
                'pheromone_deposit': (ti.f32, 1.0, 20.0),
                'sensor_angle': (ti.f32, 0.2, 1.0),
                'sensor_distance': (ti.f32, 15.0, 40.0),
                'turn_speed': (ti.f32, 0.1, 0.5),
                'wander_strength': (ti.f32, 0.0, 0.3),
            },
            'shape': 1,
            'osc': ('get', 'set'),
            'randomise': False
        })
    
    # Species-specific parameters
    if 'llm_species' not in tv.s:
        tv.s.set('llm_species', {
            'state': {
                'move_speed': (ti.f32, 50.0, 150.0),
                'home_x': (ti.f32, 0.0, 1920.0),
                'home_y': (ti.f32, 0.0, 1080.0),
                'trail_strength': (ti.f32, 0.5, 2.0),
                'exploration_rate': (ti.f32, 0.0, 0.1),
            },
            'shape': tv.sn,
            'osc': ('get', 'set'),
            'randomise': False
        })
    
    # Per-particle state
    if 'llm_particles' not in tv.s:
        tv.s.set('llm_particles', {
            'state': {
                'has_food': (ti.f32, 0.0, 1.0),
                'heading': (ti.f32, 0.0, 6.28),
                'time_since_trail': (ti.f32, 0.0, 100.0),
            },
            'shape': tv.pn,
            'osc': ('get',),
            'randomise': False
        })
    
    # Initialize global parameters
    tv.s.llm_global.field[0].pheromone_decay = 0.99
    tv.s.llm_global.field[0].pheromone_deposit = 10.0
    tv.s.llm_global.field[0].sensor_angle = 0.4
    tv.s.llm_global.field[0].sensor_distance = 25.0
    tv.s.llm_global.field[0].turn_speed = 0.3
    tv.s.llm_global.field[0].wander_strength = 0.1
    
    # Initialize species homes and parameters
    @ti.kernel
    def init_species_params():
        for s in range(tv.sn):
            # Set home positions
            home_x = 200.0 + s * ((tv.x - 400.0) / max(tv.sn - 1, 1))
            tv.s.llm_species.field[s].home_x = home_x
            tv.s.llm_species.field[s].home_y = tv.y - 200.0
            
            # Vary parameters slightly between colonies
            tv.s.llm_species.field[s].move_speed = 70.0 + ti.random() * 30.0
            tv.s.llm_species.field[s].trail_strength = 1.0 + ti.random() * 0.5
            tv.s.llm_species.field[s].exploration_rate = 0.03 + ti.random() * 0.02
    
    init_species_params()
    
    # Initialize particle states
    @ti.kernel
    def init_particle_states():
        for i in range(tv.pn):
            tv.s.llm_particles.field[i].has_food = 0.0
            vel = tv.p.field[i].vel
            tv.s.llm_particles.field[i].heading = ti.atan2(vel.y, vel.x)
            tv.s.llm_particles.field[i].time_since_trail = 0.0
    
    init_particle_states()
    
    # === Helper Functions ===
    @ti.func
    def sample_pheromone(x: ti.f32, y: ti.f32, pheromone_type: ti.i32) -> ti.f32:
        """Sample pheromone strength at a position."""
        xi = ti.cast(x, ti.i32)
        yi = ti.cast(y, ti.i32)
        
        result = 0.0
        if xi >= 0 and xi < tv.x and yi >= 0 and yi < tv.y:
            if pheromone_type == 0:
                result = pheromone_to_food[xi, yi]
            else:
                result = pheromone_to_home[xi, yi]
        
        return result
    
    @ti.func
    def sample_food(x: ti.f32, y: ti.f32) -> ti.f32:
        """Sample food amount at a position."""
        xi = ti.cast(x, ti.i32)
        yi = ti.cast(y, ti.i32)
        
        result = 0.0
        if xi >= 0 and xi < tv.x and yi >= 0 and yi < tv.y:
            result = food_sources[xi, yi]
        
        return result
    
    @ti.func
    def sense_pheromones(pos: ti.math.vec2, heading: ti.f32, pheromone_type: ti.i32) -> ti.f32:
        """Sense pheromones and return turn direction."""
        sensor_dist = tv.s.llm_global.field[0].sensor_distance
        sensor_angle = tv.s.llm_global.field[0].sensor_angle
        
        # Three sensors
        left_angle = heading - sensor_angle
        center_angle = heading
        right_angle = heading + sensor_angle
        
        # Sample positions
        left_pos = pos + ti.Vector([ti.cos(left_angle), ti.sin(left_angle)]) * sensor_dist
        center_pos = pos + ti.Vector([ti.cos(center_angle), ti.sin(center_angle)]) * sensor_dist
        right_pos = pos + ti.Vector([ti.cos(right_angle), ti.sin(right_angle)]) * sensor_dist
        
        # Sample pheromone strengths
        left_val = sample_pheromone(left_pos.x, left_pos.y, pheromone_type)
        center_val = sample_pheromone(center_pos.x, center_pos.y, pheromone_type)
        right_val = sample_pheromone(right_pos.x, right_pos.y, pheromone_type)
        
        # Determine turn direction
        turn = 0.0
        turn_speed = tv.s.llm_global.field[0].turn_speed
        
        if center_val > left_val and center_val > right_val:
            turn = 0.0
        elif left_val > right_val:
            turn = -turn_speed
        elif right_val > left_val:
            turn = turn_speed
        else:
            # Random exploration
            turn = (ti.random() - 0.5) * turn_speed * 2.0
        
        return turn
    
    # === Particle Force Experts ===
    @ti.func
    def ant_navigation(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Navigate based on ant state (carrying food or searching)."""
        has_food = tv.s.llm_particles.field[particle_idx].has_food
        heading = tv.s.llm_particles.field[particle_idx].heading
        
        # Decide which pheromone to follow
        turn = 0.0
        if has_food > 0.5:
            # Has food - follow home pheromones
            turn = sense_pheromones(pos, heading, 1)
        else:
            # Searching - follow food pheromones
            turn = sense_pheromones(pos, heading, 0)
        
        # Add wandering for exploration
        wander = tv.s.llm_global.field[0].wander_strength
        exploration = tv.s.llm_species.field[species].exploration_rate
        if ti.random() < exploration:
            turn += (ti.random() - 0.5) * wander
        
        # Update heading
        new_heading = heading + turn
        tv.s.llm_particles.field[particle_idx].heading = new_heading
        
        # Calculate movement force
        move_speed = tv.s.llm_species.field[species].move_speed
        desired_vel = ti.Vector([
            ti.cos(new_heading) * move_speed,
            ti.sin(new_heading) * move_speed
        ])
        
        force = (desired_vel - vel) * 10.0
        return force
    
    @ti.func
    def home_attraction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Attraction to home when carrying food."""
        force = ti.math.vec2(0.0, 0.0)
        has_food = tv.s.llm_particles.field[particle_idx].has_food
        
        if has_food > 0.5:
            home_x = tv.s.llm_species.field[species].home_x
            home_y = tv.s.llm_species.field[species].home_y
            home_pos = ti.Vector([home_x, home_y])
            
            # Weak attraction to home
            direction = home_pos - pos
            dist = direction.norm()
            if dist > 1.0:
                force = (direction / dist) * 20.0
        
        return force
    
    @ti.func
    def boundary_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Keep ants within bounds."""
        force = ti.math.vec2(0.0, 0.0)
        margin = 30.0
        strength = 300.0
        
        if pos.x < margin:
            force.x += strength * (1.0 - pos.x / margin)
        elif pos.x > tv.x - margin:
            force.x -= strength * (1.0 - (tv.x - pos.x) / margin)
        
        if pos.y < margin:
            force.y += strength * (1.0 - pos.y / margin)
        elif pos.y > tv.y - margin:
            force.y -= strength * (1.0 - (tv.y - pos.y) / margin)
        
        return force
    
    # === State Update Kernel ===
    @ti.kernel
    def update_ant_states():
        """Update ant states based on environment."""
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                species = tv.p.field[i].species
                has_food = tv.s.llm_particles.field[i].has_food
                
                home_x = tv.s.llm_species.field[species].home_x
                home_y = tv.s.llm_species.field[species].home_y
                dist_to_home = ti.sqrt((pos.x - home_x) ** 2 + (pos.y - home_y) ** 2)
                
                if has_food > 0.5:
                    # Carrying food - check if at home
                    if dist_to_home < 30.0:
                        # Delivered food!
                        tv.s.llm_particles.field[i].has_food = 0.0
                        tv.p.field[i].size = 3.0
                        # Turn around
                        tv.s.llm_particles.field[i].heading += 3.14159
                else:
                    # Searching - check for food
                    food_here = sample_food(pos.x, pos.y)
                    if food_here > 1.0:
                        # Found food!
                        tv.s.llm_particles.field[i].has_food = 1.0
                        tv.p.field[i].size = 5.0
                        # Turn towards home
                        angle_to_home = ti.atan2(home_y - pos.y, home_x - pos.x)
                        tv.s.llm_particles.field[i].heading = angle_to_home
                        
                        # Deplete food source
                        xi = ti.cast(pos.x, ti.i32)
                        yi = ti.cast(pos.y, ti.i32)
                        if xi >= 0 and xi < tv.x and yi >= 0 and yi < tv.y:
                            food_sources[xi, yi] = max(0.0, food_sources[xi, yi] - 10.0)
    
    # === Pheromone Management ===
    @ti.kernel
    def deposit_pheromones():
        """Ants deposit pheromones based on their state."""
        deposit_amount = tv.s.llm_global.field[0].pheromone_deposit
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                species = tv.p.field[i].species
                has_food = tv.s.llm_particles.field[i].has_food
                trail_strength = tv.s.llm_species.field[species].trail_strength
                
                xi = ti.cast(pos.x, ti.i32)
                yi = ti.cast(pos.y, ti.i32)
                
                if xi >= 0 and xi < tv.x and yi >= 0 and yi < tv.y:
                    if has_food > 0.5:
                        # Deposit "to-food" pheromone when carrying food
                        pheromone_to_food[xi, yi] += deposit_amount * trail_strength
                    else:
                        # Deposit "to-home" pheromone when searching
                        pheromone_to_home[xi, yi] += deposit_amount * trail_strength * 0.5
    
    @ti.kernel
    def update_pheromones():
        """Decay and diffuse pheromones."""
        decay_rate = tv.s.llm_global.field[0].pheromone_decay
        
        for x in range(tv.x):
            for y in range(tv.y):
                # Decay both pheromone types
                pheromone_to_food[x, y] *= decay_rate
                pheromone_to_home[x, y] *= decay_rate
                
                # Cap maximum strength
                pheromone_to_food[x, y] = min(pheromone_to_food[x, y], 100.0)
                pheromone_to_home[x, y] = min(pheromone_to_home[x, y], 100.0)
    
    # === Integration Kernel ===
    @ti.kernel
    def apply_all_experts():
        """Apply forces and update ant movement."""
        dt = 0.016
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                vel = tv.p.field[i].vel
                mass = tv.p.field[i].mass
                species = tv.p.field[i].species
                
                # Accumulate forces
                total_force = ti.math.vec2(0.0, 0.0)
                
                # Main ant navigation
                total_force += ant_navigation(pos, vel, mass, species, i)
                
                # Home attraction when carrying food
                total_force += home_attraction(pos, vel, mass, species, i)
                
                # Boundaries
                total_force += boundary_force(pos, vel, mass, species, i)
                
                # Limit force
                force_norm = total_force.norm()
                max_force = 500.0
                limited_force = total_force
                if force_norm > max_force:
                    limited_force = (total_force / force_norm) * max_force
                
                # Update velocity
                acceleration = limited_force / mass if mass > 0 else limited_force
                tv.p.field[i].vel += acceleration * dt
                
                # Limit speed
                vel_norm = tv.p.field[i].vel.norm()
                max_speed = tv.s.llm_species.field[species].move_speed
                if vel_norm > max_speed:
                    tv.p.field[i].vel = (tv.p.field[i].vel / vel_norm) * max_speed
                
                # Update position
                tv.p.field[i].pos += tv.p.field[i].vel * tv.p.field[i].speed * dt
    
    # === Drawing Functions ===
    @ti.kernel
    def draw_pheromones():
        """Visualize pheromone trails."""
        for x in range(tv.x // 3):
            for y in range(tv.y // 3):
                x_coord = x * 3
                y_coord = y * 3
                
                if x_coord < tv.x and y_coord < tv.y:
                    # Sample both pheromone types
                    food_pheromone = pheromone_to_food[x_coord, y_coord]
                    home_pheromone = pheromone_to_home[x_coord, y_coord]
                    
                    if food_pheromone > 0.5 or home_pheromone > 0.5:
                        # Mix colors based on pheromone types
                        food_intensity = min(food_pheromone / 30.0, 1.0)
                        home_intensity = min(home_pheromone / 30.0, 1.0)
                        
                        color = ti.math.vec4(
                            home_intensity * 0.3,  # Red for home trail
                            food_intensity * 0.3,  # Green for food trail
                            0.1,
                            max(food_intensity, home_intensity) * 0.4
                        )
                        tv.px.rect(x_coord, y_coord, 3, 3, color)
    
    @ti.kernel
    def draw_food_sources():
        """Draw food sources."""
        for x in range(tv.x // 2):
            for y in range(tv.y // 2):
                x_coord = x * 2
                y_coord = y * 2
                
                if x_coord < tv.x and y_coord < tv.y:
                    food = food_sources[x_coord, y_coord]
                    if food > 0.1:
                        intensity = min(food / 50.0, 1.0)
                        color = ti.math.vec4(
                            intensity * 0.8,
                            intensity * 0.8,
                            intensity * 0.2,
                            1.0
                        )
                        tv.px.rect(x_coord, y_coord, 2, 2, color)
    
    @ti.kernel
    def draw_colonies():
        """Draw colony homes."""
        for s in range(tv.sn):
            home_x = ti.cast(tv.s.llm_species.field[s].home_x, ti.i32)
            home_y = ti.cast(tv.s.llm_species.field[s].home_y, ti.i32)
            
            # Draw home as a circle
            color = tv.s.species.field[s].rgba * 0.5
            tv.px.circle(home_x, home_y, 20, color, 2)
    
    # === Render Loop ===
    @tv.render
    def _():
        tv.px.diffuse(0.98)  # Slight fade
        
        # Draw environment
        draw_food_sources()
        draw_colonies()
        draw_pheromones()
        
        # Update states
        update_ant_states()
        deposit_pheromones()
        update_pheromones()
        
        # Apply movement
        apply_all_experts()
        tv.p()
        
        # Render ants
        tv.px.particles(tv.p, tv.s.species())
        
        return tv.px

if __name__ == "__main__":
    run(main)