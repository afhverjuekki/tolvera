MOVEMENT_PATTERNS = """
# Core Movement Patterns

## Seek/Attraction
```python
to_target = target - pos
dist = to_target.norm()
if dist > 1.0:  # Avoid division by zero
    force = (to_target / dist) * attraction_strength
```

## Flee/Repulsion
```python
away_from_target = pos - target
dist = away_from_target.norm()
if dist > 0.01 and dist < repulsion_radius:
    force = (away_from_target / dist) * repulsion_strength
```

## Wander/Random Walk
```python
angle = ti.random() * 2 * 3.14159
force = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * wander_strength
```

## Orbital Motion
```python
to_center = center - pos
dist = to_center.norm()
if dist > 1.0:
    # Perpendicular vector for tangent
    tangent = ti.math.vec2(-to_center.y, to_center.x) / dist
    force = tangent * orbital_speed
```

## Damped Spring
```python
to_target = target - pos
force = to_target * spring_constant - vel * damping
```

## Spiral Motion
```python
to_center = center - pos
dist = to_center.norm()
if dist > 1.0:
    # Combine radial and tangential components
    radial = (to_center / dist) * radial_strength
    tangent = ti.math.vec2(-to_center.y, to_center.x) / dist * tangent_strength
    force = radial + tangent
```
"""

FLOCKING_PATTERNS = """
# Flocking Behaviors (Boids)

## Alignment - Match neighbor velocities
```python
perceived_velocity = ti.math.vec2(0.0, 0.0)
neighbor_count = 0

for j in range(tv.pn):
    if i != j:
        other = tv.p.field[j]
        if other.species == species:
            diff = other.pos - pos
            if diff.norm() < perception_radius:
                perceived_velocity += other.vel
                neighbor_count += 1

if neighbor_count > 0:
    desired_vel = perceived_velocity / neighbor_count
    force = (desired_vel - vel) * alignment_strength
```

## Cohesion - Move toward group center
```python
center_of_mass = ti.math.vec2(0.0, 0.0)
neighbor_count = 0

for j in range(tv.pn):
    if i != j and tv.p.field[j].species == species:
        diff = tv.p.field[j].pos - pos
        if diff.norm() < perception_radius:
            center_of_mass += tv.p.field[j].pos
            neighbor_count += 1

if neighbor_count > 0:
    center_of_mass /= neighbor_count
    force = (center_of_mass - pos) * cohesion_strength
```

## Separation - Avoid crowding
```python
separation_force = ti.math.vec2(0.0, 0.0)

for j in range(tv.pn):
    if i != j:
        diff = pos - tv.p.field[j].pos
        dist = diff.norm()
        if dist > 0.01 and dist < separation_radius:
            # Stronger repulsion when closer
            separation_force += (diff / dist) * (1.0 / dist)

force = separation_force * separation_strength
```
"""

INTERACTION_PATTERNS = """
# Particle Interaction Patterns

## Chase/Predator-Prey
```python
# In interaction function (p1 chases p2)
if p1.species == predator_species and p2.species == prey_species:
    to_prey = p2.pos - p1.pos
    dist = to_prey.norm()
    if dist > 1.0 and dist < hunt_radius:
        force = (to_prey / dist) * chase_speed
```

## Mutual Repulsion
```python
# Both particles repel each other
if (p1.species == 0 and p2.species == 1) or (p1.species == 1 and p2.species == 0):
    diff = p1.pos - p2.pos
    dist = diff.norm()
    if dist > 0.01 and dist < repel_radius:
        force = (diff / dist) * (repel_strength / (dist + 1.0))
```

## Gravitational Attraction
```python
# Newton's law of gravitation
diff = p2.pos - p1.pos
dist_sq = diff.dot(diff)
if dist_sq > 1.0:  # Minimum distance to avoid singularity
    force = diff.normalized() * (G * p1.mass * p2.mass / dist_sq)
```

## Magnetic Dipole
```python
# Attraction/repulsion based on orientation
diff = p2.pos - p1.pos
dist = diff.norm()
if dist > 1.0 and dist < interaction_radius:
    # Assuming particles have orientation stored in velocity direction
    alignment = p1.vel.normalized().dot(p2.vel.normalized())
    force = diff.normalized() * (alignment * magnetic_strength / (dist * dist))
```

## Social Distancing
```python
# Maintain preferred distance between particles
diff = p2.pos - p1.pos
dist = diff.norm()
if dist > 0.01:
    deviation = dist - preferred_distance
    force = diff.normalized() * (-deviation * social_strength)
```
"""

TEMPORAL_PATTERNS = """
# Time-Based Behavior Patterns

## Day/Night Cycle
```python
# Assuming day_phase varies 0-1 (0=midnight, 0.5=noon)
day_phase = tv.s.llm_global.field[0].day_phase
if day_phase > 0.25 and day_phase < 0.75:  # Daytime
    activity_multiplier = 1.0 + ti.sin((day_phase - 0.25) * 2 * 3.14159)
else:  # Nighttime
    activity_multiplier = 0.1

force = base_force * activity_multiplier
```

## Energy-Based Movement
```python
energy = tv.s.llm_particle.field[particle_idx].energy
if energy > 20.0:
    # Active movement
    angle = ti.random() * 2 * 3.14159
    force = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * energy * 0.1
else:
    # Rest/minimal movement
    force = vel * -0.5  # Damping
```

## Periodic Behavior
```python
phase = tv.s.llm_particle.field[particle_idx].phase
oscillation = ti.sin(phase * 2 * 3.14159)
force = ti.math.vec2(oscillation * 100.0, 0.0)
```

## Tidal Forces
```python
# Two overlapping cycles
frame = tv.ctx.i[None]
tide1 = ti.sin(frame * 0.01) * 50.0
tide2 = ti.sin(frame * 0.007) * 30.0
force = ti.math.vec2(0.0, tide1 + tide2)
```

## Memory Decay
```python
# Gradually forget home position
home_pos = tv.s.llm_particle.field[particle_idx].home_pos
memory_strength = tv.s.llm_particle.field[particle_idx].memory
to_home = home_pos - pos
if memory_strength > 0.1:
    force = to_home * memory_strength * 0.5
    # Decay memory over time
    tv.s.llm_particle.field[particle_idx].memory *= 0.999
```
"""

CELLULAR_AUTOMATA_PATTERNS = """
# Cellular Automata Patterns

## Conway's Game of Life Rules
```python
# Count live neighbors (assuming grid positioning)
live_neighbors = 0
grid_x = ti.cast(pos.x / cell_size, ti.i32)
grid_y = ti.cast(pos.y / cell_size, ti.i32)

for dx in range(-1, 2):
    for dy in range(-1, 2):
        if dx != 0 or dy != 0:
            nx = grid_x + dx
            ny = grid_y + dy
            # Check if neighbor exists and is alive
            for j in range(tv.pn):
                other = tv.p.field[j]
                ox = ti.cast(other.pos.x / cell_size, ti.i32)
                oy = ti.cast(other.pos.y / cell_size, ti.i32)
                if ox == nx and oy == ny and other.active > 0.5:
                    live_neighbors += 1

# Apply rules
if tv.p.field[i].active > 0.5:  # Currently alive
    if live_neighbors < 2 or live_neighbors > 3:
        # Dies
        tv.p.field[i].active = 0.0
else:  # Currently dead
    if live_neighbors == 3:
        # Birth
        tv.p.field[i].active = 1.0
```

## Reaction-Diffusion
```python
# Simple activator-inhibitor system
activator = tv.s.llm_particle.field[particle_idx].activator
inhibitor = tv.s.llm_particle.field[particle_idx].inhibitor

# Diffusion (implemented as force towards concentration gradient)
gradient = ti.math.vec2(0.0, 0.0)
local_activator = 0.0
neighbor_count = 0

for j in range(tv.pn):
    if i != j:
        diff = tv.p.field[j].pos - pos
        dist = diff.norm()
        if dist < diffusion_radius:
            other_activator = tv.s.llm_particle.field[j].activator
            gradient += diff.normalized() * (other_activator - activator)
            local_activator += other_activator
            neighbor_count += 1

if neighbor_count > 0:
    local_activator /= neighbor_count
    force = gradient * diffusion_rate
```
"""

EMERGENT_PATTERNS = """
# Emergent Behavior Patterns

## Slime Mold (Physarum)
```python
# Sense pheromone ahead and turn towards highest concentration
look_ahead_dist = 20.0
sensor_angle = 0.5  # radians

# Current direction from velocity
current_dir = vel.normalized() if vel.norm() > 0.1 else ti.math.vec2(1.0, 0.0)

# Sample three directions
left_dir = ti.math.vec2(
    current_dir.x * ti.cos(-sensor_angle) - current_dir.y * ti.sin(-sensor_angle),
    current_dir.x * ti.sin(-sensor_angle) + current_dir.y * ti.cos(-sensor_angle)
)
right_dir = ti.math.vec2(
    current_dir.x * ti.cos(sensor_angle) - current_dir.y * ti.sin(sensor_angle),
    current_dir.x * ti.sin(sensor_angle) + current_dir.y * ti.cos(sensor_angle)
)

# Sample positions
front_pos = pos + current_dir * look_ahead_dist
left_pos = pos + left_dir * look_ahead_dist
right_pos = pos + right_dir * look_ahead_dist

# Get pheromone values (simplified - would need pixel sampling)
front_val = 0.5  # Placeholder
left_val = 0.6
right_val = 0.4

# Turn towards highest concentration
if left_val > front_val and left_val > right_val:
    force = left_dir * turn_speed
elif right_val > front_val and right_val > left_val:
    force = right_dir * turn_speed
else:
    force = current_dir * move_speed
```

## Ant Trails
```python
# Follow pheromone trails with some randomness
trail_following = ti.math.vec2(0.0, 0.0)
random_walk = ti.math.vec2(0.0, 0.0)

# Simplified trail following (would need pheromone field)
if has_trail_ahead:
    trail_following = trail_direction * trail_strength
else:
    # Random search
    angle = ti.random() * 2 * 3.14159
    random_walk = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * search_strength

# Combine behaviors
force = trail_following + random_walk

# Deposit pheromone (would need pixel writing)
# tv.px.point(pos.x, pos.y, pheromone_color)
```

## Firefly Synchronization
```python
# Kuramoto model for synchronization
phase = tv.s.llm_particle.field[particle_idx].phase
natural_freq = 1.0
coupling_strength = 0.1

# Calculate phase coupling with neighbors
phase_force = 0.0
neighbor_count = 0

for j in range(tv.pn):
    if i != j and tv.p.field[j].species == species:
        diff = tv.p.field[j].pos - pos
        if diff.norm() < sync_radius:
            other_phase = tv.s.llm_particle.field[j].phase
            phase_force += ti.sin(other_phase - phase)
            neighbor_count += 1

if neighbor_count > 0:
    phase_force = (phase_force / neighbor_count) * coupling_strength
    
# Update phase
new_phase = phase + (natural_freq + phase_force) * 0.016
tv.s.llm_particle.field[particle_idx].phase = new_phase % (2 * 3.14159)

# Flash when phase crosses threshold
if phase < 3.14159 and new_phase >= 3.14159:
    # Firefly flashes - could modify particle size or brightness
    tv.p.field[i].size = 10.0
else:
    tv.p.field[i].size = 5.0
```
"""