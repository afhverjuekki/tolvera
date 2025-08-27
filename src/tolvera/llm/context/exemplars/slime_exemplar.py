"""
Golden Exemplar: Slime Mold (Physarum) Simulation
=================================================
This exemplar demonstrates agent-based slime mold behavior with
pheromone trail deposition, multi-directional sensing, and gradient
following. Key patterns: pixel buffer usage, sensor sampling, trail diffusion.
"""

SLIME_EXEMPLAR = """
# Slime Mold (Physarum) - Agent-Based Trail System

## Core Slime Agent Behavior

This exemplar shows how to implement physarum-like slime mold behavior with
agents that deposit trails and follow chemical gradients in the environment.

```python
@ti.func
def slime_sense_and_turn(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, 
                         species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Sense pheromone gradient and turn toward highest concentration.'''
    
    # Sensor parameters per species
    sensor_angle = tv.s.llm_species.field[species].sensor_angle      # e.g., 0.4 rad
    sensor_distance = tv.s.llm_species.field[species].sensor_distance # e.g., 20.0 pixels
    turn_speed = tv.s.llm_species.field[species].turn_speed          # e.g., 0.1 rad/frame
    
    # Current heading from velocity
    heading = 0.0
    vel_norm = vel.norm()
    if vel_norm > 0.001:
        heading = ti.atan2(vel.y, vel.x)
    
    # Three sensor positions (left, center, right)
    left_angle = heading - sensor_angle
    center_angle = heading
    right_angle = heading + sensor_angle
    
    # Sample pheromone at sensor positions
    left_pos = pos + ti.math.vec2(ti.cos(left_angle), ti.sin(left_angle)) * sensor_distance
    center_pos = pos + ti.math.vec2(ti.cos(center_angle), ti.sin(center_angle)) * sensor_distance
    right_pos = pos + ti.math.vec2(ti.cos(right_angle), ti.sin(right_angle)) * sensor_distance
    
    # Read pheromone values from pixel buffer (with boundary checks)
    left_val = 0.0
    center_val = 0.0
    right_val = 0.0
    
    # Sample left sensor
    lx = ti.cast(left_pos.x, ti.i32)
    ly = ti.cast(left_pos.y, ti.i32)
    if 0 <= lx < tv.x and 0 <= ly < tv.y:
        # Sum all color channels for total pheromone
        color = tv.px.px.rgba[lx, ly]
        left_val = color.x + color.y + color.z
    
    # Sample center sensor
    cx = ti.cast(center_pos.x, ti.i32)
    cy = ti.cast(center_pos.y, ti.i32)
    if 0 <= cx < tv.x and 0 <= cy < tv.y:
        color = tv.px.px.rgba[cx, cy]
        center_val = color.x + color.y + color.z
    
    # Sample right sensor
    rx = ti.cast(right_pos.x, ti.i32)
    ry = ti.cast(right_pos.y, ti.i32)
    if 0 <= rx < tv.x and 0 <= ry < tv.y:
        color = tv.px.px.rgba[rx, ry]
        right_val = color.x + color.y + color.z
    
    # Determine turn direction based on sensor values
    # CRITICAL: Declare turn before conditionals
    turn = 0.0
    
    # Random exploration if no gradient detected
    if left_val == center_val and center_val == right_val:
        # Random turn for exploration
        turn = (ti.random() - 0.5) * turn_speed * 2.0
    elif left_val > center_val and left_val > right_val:
        # Turn left toward higher concentration
        turn = -turn_speed
    elif right_val > center_val and right_val > left_val:
        # Turn right toward higher concentration
        turn = turn_speed
    # else: continue straight (turn = 0)
    
    # Apply turn to velocity
    new_heading = heading + turn
    speed = tv.s.llm_species.field[species].move_speed  # e.g., 100.0
    
    # Calculate steering force toward new heading
    desired_vel = ti.math.vec2(ti.cos(new_heading), ti.sin(new_heading)) * speed
    force = (desired_vel - vel) * 10.0  # Steering strength
    
    return force
```

## Trail Deposition

```python
@ti.func
def deposit_pheromone(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32,
                      species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Deposit pheromone trail at current position.'''
    
    if tv.p.field[particle_idx].active > 0:
        x = ti.cast(pos.x, ti.i32)
        y = ti.cast(pos.y, ti.i32)
        
        if 0 <= x < tv.x and 0 <= y < tv.y:
            # Get species-specific trail color and strength
            deposition_rate = tv.s.llm_species.field[species].deposition_rate  # e.g., 0.5
            trail_color = tv.s.species.field[species].rgba
            
            # Deposit pheromone (additive)
            tv.px.px.rgba[x, y] += trail_color * deposition_rate
            
            # Optional: Deposit in small radius for thicker trails
            radius = 1
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < tv.x and 0 <= ny < tv.y:
                        dist = ti.sqrt(ti.cast(dx * dx + dy * dy, ti.f32))
                        if dist <= radius:
                            intensity = 1.0 - dist / ti.cast(radius + 1, ti.f32)
                            tv.px.px.rgba[nx, ny] += trail_color * deposition_rate * intensity * 0.5
    
    # Trail deposition doesn't apply force
    return ti.math.vec2(0.0, 0.0)
```

## Environment Diffusion and Decay

```python
@ti.kernel
def diffuse_and_decay_trails():
    '''Diffuse pheromone trails and apply decay over time.'''
    
    diffusion_rate = 0.1   # How much spreads to neighbors
    decay_rate = 0.99      # How fast trails fade
    
    # Create temporary buffer for diffusion
    for x, y in ti.ndrange(tv.x, tv.y):
        # Sample 3x3 neighborhood
        total = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        count = 0
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = (x + dx) % tv.x  # Toroidal wrap
                ny = (y + dy) % tv.y
                
                # Weight center pixel higher (blur kernel)
                weight = 1.0
                if dx == 0 and dy == 0:
                    weight = 4.0
                
                total += tv.px.px.rgba[nx, ny] * weight
                count += ti.cast(weight, ti.i32)
        
        # Apply diffusion and decay
        if count > 0:
            blurred = total / ti.cast(count, ti.f32)
            current = tv.px.px.rgba[x, y]
            
            # Mix original with blurred
            mixed = current * (1.0 - diffusion_rate) + blurred * diffusion_rate
            
            # Apply decay
            tv.px.px.rgba[x, y] = mixed * decay_rate
```

## State Setup for Slime

```python
# Global slime parameters
tv.s.llm_global = {
    'diffusion_rate': 0.1,   # Trail spreading
    'decay_rate': 0.98,      # Trail fade speed
    'sensor_offset': 0.4,    # Sensor angle offset (radians)
}

# Species-specific slime parameters
tv.s.llm_species = {
    'sensor_angle': (0.2, 0.6),        # Angle between sensors
    'sensor_distance': (10.0, 30.0),   # How far sensors look
    'turn_speed': (0.05, 0.2),         # Rotation rate
    'move_speed': (50.0, 150.0),       # Movement speed
    'deposition_rate': (0.1, 1.0),     # Trail strength
}

# Optional: Per-particle memory
tv.s.llm_particle = {
    'trail_strength': (0.0, 1.0),      # Individual deposition
    'last_sensor_val': (0.0, 1.0),     # Memory of last sense
}
```

## Integration Pattern

```python
@ti.kernel
def integrate_slime_behavior():
    '''Main slime mold integration kernel.'''
    dt = 0.016  # 60 FPS
    
    # First pass: Update agents
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            pos = tv.p.field[i].pos
            vel = tv.p.field[i].vel
            mass = tv.p.field[i].mass
            species = tv.p.field[i].species
            
            # Accumulate forces
            total_force = ti.math.vec2(0.0, 0.0)
            
            # Sense and turn toward pheromone
            total_force += slime_sense_and_turn(pos, vel, mass, species, i)
            
            # Random exploration
            total_force += ti.math.vec2(
                (ti.random() - 0.5) * 20.0,
                (ti.random() - 0.5) * 20.0
            )
            
            # Update velocity
            acceleration = total_force / mass if mass > 0 else total_force
            tv.p.field[i].vel += acceleration * dt
            
            # Limit speed
            vel_norm = tv.p.field[i].vel.norm()
            max_speed = tv.s.llm_species.field[species].move_speed
            if vel_norm > max_speed:
                tv.p.field[i].vel = (tv.p.field[i].vel / vel_norm) * max_speed
            
            # Update position
            tv.p.field[i].pos += tv.p.field[i].vel * dt
            
            # Toroidal boundary
            tv.p.field[i].pos.x = tv.p.field[i].pos.x % tv.x
            tv.p.field[i].pos.y = tv.p.field[i].pos.y % tv.y
    
    # Second pass: Deposit trails
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            deposit_pheromone(
                tv.p.field[i].pos,
                tv.p.field[i].vel,
                tv.p.field[i].mass,
                tv.p.field[i].species,
                i
            )
```

## Visual Enhancements

```python
@ti.func
def draw_sensor_debug(i: ti.i32):
    '''Visualize sensor positions for debugging.'''
    if tv.p.field[i].active > 0 and i < 10:  # Only first 10 for clarity
        pos = tv.p.field[i].pos
        vel = tv.p.field[i].vel
        species = tv.p.field[i].species
        
        vel_norm = vel.norm()
        if vel_norm > 0.001:
            heading = ti.atan2(vel.y, vel.x)
            
            sensor_angle = tv.s.llm_species.field[species].sensor_angle
            sensor_distance = tv.s.llm_species.field[species].sensor_distance
            
            # Draw sensor lines
            for angle_offset in [-sensor_angle, 0.0, sensor_angle]:
                angle = heading + angle_offset
                end_pos = pos + ti.math.vec2(ti.cos(angle), ti.sin(angle)) * sensor_distance
                
                # Color based on sensor position
                color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                if angle_offset < 0:
                    color = ti.math.vec4(1.0, 0.0, 0.0, 0.3)  # Left = red
                elif angle_offset > 0:
                    color = ti.math.vec4(0.0, 0.0, 1.0, 0.3)  # Right = blue
                else:
                    color = ti.math.vec4(0.0, 1.0, 0.0, 0.3)  # Center = green
                
                tv.px.line(pos.x, pos.y, end_pos.x, end_pos.y, color)
```

## Key Patterns Demonstrated

1. **Pixel Buffer as Environment**: Using pixel field for pheromone storage
2. **Multi-Sensor Sampling**: Three-point gradient detection
3. **Trail Deposition**: Writing to pixel buffer with additive blending
4. **Diffusion Algorithm**: Neighborhood averaging with decay
5. **Steering Behavior**: Turning based on sensory input
6. **Toroidal Boundaries**: Seamless world wrapping
7. **Species Variation**: Different sensor configurations per species
8. **Debug Visualization**: Sensor ray visualization
"""

# Export exemplar
__all__ = ['SLIME_EXEMPLAR']