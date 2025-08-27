"""
Slime Mold (Physarum) Simulation in Tölvera
Agents deposit and follow chemical trails, creating emergent transport networks
"""

import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    """Main function for Slime Mold simulation."""
    # === Configuration ===
    if 'species' not in kwargs:
        kwargs['species'] = 4  # Different mold colonies with varying behaviors
    if 'particles' not in kwargs:
        kwargs['particles'] = 5000  # Number of slime mold agents
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080
    
    # Create Tölvera instance
    tv = Tolvera(**kwargs)
    
    # === Trail Map Field ===
    # Create a field for chemical trails (pheromones)
    trail_field = ti.field(dtype=ti.f32, shape=(tv.x, tv.y))
    
    # === Particle Initialization ===
    @ti.kernel
    def init_particles_clusters():
        """Initialize slime mold agents in colony clusters."""
        particles_per_species = tv.pn // tv.sn
        remaining = tv.pn % tv.sn
        
        particle_idx = 0
        
        for species_id in range(tv.sn):
            count = particles_per_species + (1 if species_id < remaining else 0)
            
            # Create a circular cluster for each colony
            cluster_center = ti.Vector([
                200.0 + (tv.x - 400.0) * ti.random(),
                200.0 + (tv.y - 400.0) * ti.random()
            ])
            
            for i in range(count):
                if particle_idx < tv.pn:
                    # Position agents in a circular cluster
                    angle = ti.random() * 2.0 * 3.14159
                    radius = ti.random() * 50.0
                    
                    tv.p.field[particle_idx].active = 1.0
                    tv.p.field[particle_idx].pos = cluster_center + ti.Vector([
                        ti.cos(angle) * radius,
                        ti.sin(angle) * radius
                    ])
                    
                    # Random initial heading
                    heading = ti.random() * 2.0 * 3.14159
                    speed = 50.0 + ti.random() * 30.0
                    tv.p.field[particle_idx].vel = ti.Vector([
                        ti.cos(heading) * speed,
                        ti.sin(heading) * speed
                    ])
                    
                    tv.p.field[particle_idx].size = 2.0
                    tv.p.field[particle_idx].mass = 1.0
                    tv.p.field[particle_idx].speed = 1.0
                    tv.p.field[particle_idx].species = species_id
                    
                    particle_idx += 1
    
    init_particles_clusters()
    
    # Set species colors - different colonies
    colors = [
        [1.0, 1.0, 0.2, 1.0],  # Yellow colony
        [0.2, 1.0, 1.0, 1.0],  # Cyan colony
        [1.0, 0.2, 1.0, 1.0],  # Magenta colony
        [0.5, 1.0, 0.2, 1.0],  # Lime colony
    ]
    
    for i in range(min(tv.sn, len(colors))):
        tv.s.species.field[i].rgba = colors[i]
    
    # === State Initialization ===
    # Global parameters for slime mold behavior
    if 'llm_global' not in tv.s:
        tv.s.set('llm_global', {
            'state': {
                'sensor_angle': (ti.f32, 0.1, 1.0),  # Angle offset for sensors (radians)
                'sensor_distance': (ti.f32, 10.0, 50.0),  # How far sensors look
                'rotation_angle': (ti.f32, 0.1, 1.0),  # Turn angle when sensing
                'deposit_amount': (ti.f32, 1.0, 10.0),  # Trail deposition strength
                'decay_rate': (ti.f32, 0.95, 0.999),  # Trail decay
                'diffusion_rate': (ti.f32, 0.0, 0.5),  # Trail diffusion
            },
            'shape': 1,
            'osc': ('get', 'set'),
            'randomise': False
        })
    
    # Species-specific parameters
    if 'llm_species' not in tv.s:
        tv.s.set('llm_species', {
            'state': {
                'move_speed': (ti.f32, 30.0, 150.0),
                'sensor_width': (ti.f32, 0.2, 1.5),  # Sensor angle spread
                'attraction_strength': (ti.f32, 0.5, 2.0),  # How strongly attracted to trails
                'random_turn_chance': (ti.f32, 0.0, 0.1),  # Exploration vs exploitation
            },
            'shape': tv.sn,
            'osc': ('get', 'set'),
            'randomise': True
        })
    
    # Per-particle state
    if 'llm_particles' not in tv.s:
        tv.s.set('llm_particles', {
            'state': {
                'heading': (ti.f32, 0.0, 6.28),  # Current heading in radians
                'trail_strength': (ti.f32, 0.0, 10.0),  # Individual trail deposition
            },
            'shape': tv.pn,
            'osc': ('get',),
            'randomise': False
        })
    
    # Initialize global parameters
    tv.s.llm_global.field[0].sensor_angle = 0.5
    tv.s.llm_global.field[0].sensor_distance = 25.0
    tv.s.llm_global.field[0].rotation_angle = 0.3
    tv.s.llm_global.field[0].deposit_amount = 5.0
    tv.s.llm_global.field[0].decay_rate = 0.98
    tv.s.llm_global.field[0].diffusion_rate = 0.1
    
    # Initialize species parameters with variation
    @ti.kernel
    def init_species_params():
        for s in range(tv.sn):
            tv.s.llm_species.field[s].move_speed = 60.0 + ti.random() * 40.0
            tv.s.llm_species.field[s].sensor_width = 0.4 + ti.random() * 0.4
            tv.s.llm_species.field[s].attraction_strength = 1.0 + ti.random() * 0.5
            tv.s.llm_species.field[s].random_turn_chance = 0.02 + ti.random() * 0.03
    
    init_species_params()
    
    # Initialize particle headings
    @ti.kernel
    def init_particle_headings():
        for i in range(tv.pn):
            vel = tv.p.field[i].vel
            heading = ti.atan2(vel.y, vel.x)
            tv.s.llm_particles.field[i].heading = heading
            tv.s.llm_particles.field[i].trail_strength = 5.0
    
    init_particle_headings()
    
    # === Sensing Functions ===
    @ti.func
    def sample_trail(x: ti.f32, y: ti.f32) -> ti.f32:
        """Sample trail strength at a position."""
        xi = ti.cast(x, ti.i32)
        yi = ti.cast(y, ti.i32)
        
        # Declare result variable before conditional
        result = 0.0
        
        # Boundary check
        if xi >= 0 and xi < tv.x and yi >= 0 and yi < tv.y:
            result = trail_field[xi, yi]
        
        return result
    
    @ti.func
    def sense_trails(pos: ti.math.vec2, heading: ti.f32, species: ti.i32) -> ti.f32:
        """Sense chemical trails and return turn direction."""
        sensor_dist = tv.s.llm_global.field[0].sensor_distance
        sensor_angle = tv.s.llm_global.field[0].sensor_angle
        sensor_width = tv.s.llm_species.field[species].sensor_width
        
        # Three sensors: left, center, right
        left_angle = heading - sensor_angle * sensor_width
        center_angle = heading
        right_angle = heading + sensor_angle * sensor_width
        
        # Sample positions
        left_pos = pos + ti.Vector([ti.cos(left_angle), ti.sin(left_angle)]) * sensor_dist
        center_pos = pos + ti.Vector([ti.cos(center_angle), ti.sin(center_angle)]) * sensor_dist
        right_pos = pos + ti.Vector([ti.cos(right_angle), ti.sin(right_angle)]) * sensor_dist
        
        # Sample trail strengths
        left_val = sample_trail(left_pos.x, left_pos.y)
        center_val = sample_trail(center_pos.x, center_pos.y)
        right_val = sample_trail(right_pos.x, right_pos.y)
        
        # Determine turn direction
        turn = 0.0
        rotation_angle = tv.s.llm_global.field[0].rotation_angle
        
        if center_val > left_val and center_val > right_val:
            # Move forward
            turn = 0.0
        elif left_val > right_val:
            # Turn left
            turn = -rotation_angle
        elif right_val > left_val:
            # Turn right
            turn = rotation_angle
        else:
            # Random turn for exploration
            turn = (ti.random() - 0.5) * rotation_angle * 2.0
        
        return turn
    
    # === Particle Force Experts ===
    @ti.func
    def chemotaxis_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Follow chemical trails using sensory feedback."""
        heading = tv.s.llm_particles.field[particle_idx].heading
        
        # Sense and get turn direction
        turn = sense_trails(pos, heading, species)
        
        # Add random exploration
        if ti.random() < tv.s.llm_species.field[species].random_turn_chance:
            turn += (ti.random() - 0.5) * tv.s.llm_global.field[0].rotation_angle * 2.0
        
        # Update heading
        new_heading = heading + turn * tv.s.llm_species.field[species].attraction_strength
        tv.s.llm_particles.field[particle_idx].heading = new_heading
        
        # Calculate force towards new heading
        desired_vel = ti.Vector([
            ti.cos(new_heading),
            ti.sin(new_heading)
        ]) * tv.s.llm_species.field[species].move_speed
        
        force = (desired_vel - vel) * 10.0
        return force
    
    @ti.func
    def boundary_avoidance(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Avoid edges with soft boundaries."""
        force = ti.math.vec2(0.0, 0.0)
        margin = 50.0
        strength = 200.0
        
        # Left edge
        if pos.x < margin:
            force.x += strength * (1.0 - pos.x / margin)
        # Right edge
        elif pos.x > tv.x - margin:
            force.x -= strength * (1.0 - (tv.x - pos.x) / margin)
        
        # Top edge
        if pos.y < margin:
            force.y += strength * (1.0 - pos.y / margin)
        # Bottom edge
        elif pos.y > tv.y - margin:
            force.y -= strength * (1.0 - (tv.y - pos.y) / margin)
        
        return force
    
    @ti.func
    def colony_cohesion(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Weak attraction to same species agents."""
        force = ti.math.vec2(0.0, 0.0)
        perception_radius = 100.0
        center = ti.math.vec2(0.0, 0.0)
        count = 0
        
        for j in range(tv.pn):
            if particle_idx != j and tv.p.field[j].active > 0:
                if tv.p.field[j].species == species:
                    other_pos = tv.p.field[j].pos
                    dist = (other_pos - pos).norm()
                    
                    if dist < perception_radius:
                        center += other_pos
                        count += 1
        
        # Weak cohesion force
        if count > 0:
            center = center / ti.cast(count, ti.f32)
            direction = center - pos
            dir_norm = direction.norm()
            if dir_norm > 0.001:
                force = (direction / dir_norm) * 10.0
        
        return force
    
    # === Trail Management ===
    @ti.kernel
    def deposit_trails():
        """Agents deposit chemical trails at their positions."""
        deposit_amount = tv.s.llm_global.field[0].deposit_amount
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                xi = ti.cast(pos.x, ti.i32)
                yi = ti.cast(pos.y, ti.i32)
                
                # Deposit trail at current position
                if xi >= 0 and xi < tv.x and yi >= 0 and yi < tv.y:
                    strength = tv.s.llm_particles.field[i].trail_strength
                    trail_field[xi, yi] += deposit_amount * strength
    
    @ti.kernel
    def update_trails():
        """Decay and diffuse chemical trails."""
        decay_rate = tv.s.llm_global.field[0].decay_rate
        diffusion_rate = tv.s.llm_global.field[0].diffusion_rate
        
        # Create temporary field for diffusion
        for x in range(tv.x):
            for y in range(tv.y):
                # Decay
                current = trail_field[x, y] * decay_rate
                
                # Simple diffusion (average with neighbors)
                if diffusion_rate > 0:
                    neighbor_sum = 0.0
                    neighbor_count = 0
                    
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx = x + dx
                            ny = y + dy
                            if nx >= 0 and nx < tv.x and ny >= 0 and ny < tv.y:
                                neighbor_sum += trail_field[nx, ny]
                                neighbor_count += 1
                    
                    if neighbor_count > 0:
                        avg = neighbor_sum / ti.cast(neighbor_count, ti.f32)
                        current = current * (1.0 - diffusion_rate) + avg * diffusion_rate
                
                # Cap maximum trail strength
                trail_field[x, y] = ti.min(current, 100.0)
    
    # === Integration Kernel ===
    @ti.kernel
    def apply_all_experts():
        """Apply all forces and update slime mold agents."""
        dt = 0.016
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                vel = tv.p.field[i].vel
                mass = tv.p.field[i].mass
                species = tv.p.field[i].species
                
                # Accumulate forces
                total_force = ti.math.vec2(0.0, 0.0)
                
                # Main chemotaxis behavior
                total_force += chemotaxis_force(pos, vel, mass, species, i)
                
                # Boundary avoidance
                total_force += boundary_avoidance(pos, vel, mass, species, i)
                
                # Weak colony cohesion
                total_force += colony_cohesion(pos, vel, mass, species, i) * 0.1
                
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
    def draw_trail_field():
        """Visualize the chemical trail field."""
        for x in range(tv.x // 2):  # Sample every 2 pixels for performance
            for y in range(tv.y // 2):
                x_coord = x * 2
                y_coord = y * 2
                if x_coord < tv.x and y_coord < tv.y:
                    strength = trail_field[x_coord, y_coord]
                    if strength > 0.1:
                        # Map strength to color intensity
                        intensity = ti.min(strength / 20.0, 1.0)
                        color = ti.math.vec4(
                            intensity * 0.2,
                            intensity * 0.3,
                            intensity * 0.1,
                            intensity * 0.5
                        )
                        tv.px.rect(x_coord, y_coord, 2, 2, color)
    
    @ti.func
    def draw_agent_sensors(i: ti.i32):
        """Draw sensor rays for visualization."""
        if tv.p.field[i].active > 0 and i < 100:  # Only first 100 for performance
            pos = tv.p.field[i].pos
            heading = tv.s.llm_particles.field[i].heading
            species = tv.p.field[i].species
            
            sensor_dist = tv.s.llm_global.field[0].sensor_distance
            sensor_angle = tv.s.llm_global.field[0].sensor_angle
            sensor_width = tv.s.llm_species.field[species].sensor_width
            
            color = tv.s.species.field[species].rgba * 0.2
            
            # Draw left sensor ray
            left_angle = heading - sensor_angle * sensor_width
            left_end = pos + ti.Vector([ti.cos(left_angle), ti.sin(left_angle)]) * sensor_dist
            tv.px.line(pos.x, pos.y, left_end.x, left_end.y, color)
            
            # Draw center sensor ray
            center_end = pos + ti.Vector([ti.cos(heading), ti.sin(heading)]) * sensor_dist
            tv.px.line(pos.x, pos.y, center_end.x, center_end.y, color)
            
            # Draw right sensor ray
            right_angle = heading + sensor_angle * sensor_width
            right_end = pos + ti.Vector([ti.cos(right_angle), ti.sin(right_angle)]) * sensor_dist
            tv.px.line(pos.x, pos.y, right_end.x, right_end.y, color)
    
    # === Drawing Kernel ===
    @ti.kernel
    def draw_visuals():
        """Draw visual effects."""
        # Draw sensor rays for some agents
        for i in range(min(50, tv.pn)):
            draw_agent_sensors(i)
    
    # === Render Loop ===
    @tv.render
    def _():
        tv.px.diffuse(0.99)  # Very slow fade for trail persistence
        
        # Draw trail field
        draw_trail_field()
        
        # Draw visual debugging
        draw_visuals()
        
        # Update trails
        deposit_trails()
        update_trails()
        
        # Apply agent behaviors
        apply_all_experts()
        tv.p()
        
        # Render particles
        tv.px.particles(tv.p, tv.s.species())
        
        return tv.px

if __name__ == "__main__":
    run(main)