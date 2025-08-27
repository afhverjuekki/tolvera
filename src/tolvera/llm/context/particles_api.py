"""
Comprehensive Tölvera Particles API Documentation
==================================================
This module provides complete documentation for the Tölvera particle system API.
All signatures and types are verified against the actual implementation.
"""

PARTICLES_API = """
# Tölvera Particles API Reference

## Particle Data Structure (tv.p.field[i])

The particle system is accessed through `tv.p` (Particles instance) which contains a field of particles.

### Core Particle Properties
Every particle has these built-in properties - DO NOT create custom states for these:

```python
# Access particle properties (all are ALWAYS available)
tv.p.field[i].pos        # ti.math.vec2 - Position (x, y)
tv.p.field[i].vel        # ti.math.vec2 - Velocity (vx, vy)
tv.p.field[i].mass       # ti.f32 - Particle mass (default 1.0)
tv.p.field[i].size       # ti.f32 - Display/collision size
tv.p.field[i].speed      # ti.f32 - Speed multiplier
tv.p.field[i].species    # ti.i32 - Species ID (0 to tv.sn-1)
tv.p.field[i].active     # ti.f32 - Activity level (0.0=inactive, 1.0=active)
tv.p.field[i].ppos       # ti.math.vec2 - Previous position (for trails/collision)
tv.p.field[i].pvel       # ti.math.vec2 - Previous velocity (for acceleration)
```

### Particle Methods
Methods available on individual particle structs:

```python
# Distance calculations
p1.dist(p2)              # Returns ti.math.vec2 distance vector (p1.pos - p2.pos)
p1.dist_norm(p2)         # Returns scalar distance (Euclidean norm)
p1.dist_normalized(p2)   # Returns normalized direction vector
p1.dist_wrap(p2, tv.x, tv.y)  # Toroidal wrap-around distance

# Randomization (typically used in initialization)
p.randomise(tv.x, tv.y)  # Randomize both position and velocity
p.randomise_pos(tv.x, tv.y)  # Randomize position only
p.randomise_vel()        # Randomize velocity only
```

## Particles System Access (tv.p)

### Global Properties
```python
tv.p.n                   # Total number of particles (alias for tv.pn)
tv.p.field               # Particle field array
tv.p.field.shape[0]      # Number of particles
tv.p.active_count[None]  # Number of currently active particles
tv.p.active_indexes      # Array of active particle indices
```

### Common Iteration Patterns

#### Iterating Over All Particles
```python
@ti.kernel
def process_all_particles():
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            pos = tv.p.field[i].pos
            vel = tv.p.field[i].vel
            # Process particle...
```

#### Iterating Over Active Particles Only
```python
@ti.kernel
def process_active_particles():
    for idx in range(tv.p.active_count[None]):
        i = tv.p.active_indexes[idx]
        pos = tv.p.field[i].pos
        # Process active particle...
```

#### Particle-Particle Interactions
```python
@ti.func
def interaction_force(pos: ti.math.vec2, vel: ti.math.vec2, 
                      mass: ti.f32, species: ti.i32, 
                      particle_idx: ti.i32) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    
    for j in range(tv.pn):
        if particle_idx != j and tv.p.field[j].active > 0:
            other_pos = tv.p.field[j].pos
            other_species = tv.p.field[j].species
            
            # Calculate interaction...
            diff = other_pos - pos
            dist = diff.norm()
            
            # CRITICAL: Declare variables before conditionals
            direction = ti.math.vec2(0.0, 0.0)
            
            if dist > 0.001 and dist < interaction_radius:
                direction = diff / dist
                force += direction * strength
    
    return force
```

## Particle Update Pipeline

### Standard Update Order
1. Calculate forces (experts)
2. Update velocities
3. Update positions
4. Apply boundary conditions
5. Update previous state (ppos, pvel)

### Velocity and Position Updates
```python
# Apply force to velocity (F = ma)
acceleration = force / mass if mass > 0 else force
new_vel = vel + acceleration * dt  # dt typically 0.016 for 60 FPS

# Limit velocity
vel_norm = new_vel.norm()
if vel_norm > max_speed:
    new_vel = (new_vel / vel_norm) * max_speed

# Update position
new_pos = pos + new_vel * dt

# Store for next frame
tv.p.field[i].vel = new_vel
tv.p.field[i].pos = new_pos
```

## Species System Access

### Species Properties
```python
tv.sn                    # Total number of species
tv.s.species.field[i]    # Access species i properties
tv.s.species.field[i].rgba  # Species color (vec4)
```

### Species-Specific Behaviors
```python
@ti.func
def species_specific_force(...) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    
    # Different behavior per species
    if species == 0:  # Predator
        force = chase_force
    elif species == 1:  # Prey
        force = flee_force
    else:  # Neutral
        force = wander_force
    
    return force  # Single return at end
```

## Active/Inactive Particles

### Activity Management
```python
# Deactivate particle
tv.p.field[i].active = 0.0

# Activate particle
tv.p.field[i].active = 1.0

# Gradual decay
tv.p.field[i].active *= 0.99  # Fade out over time

# Check if active
if tv.p.field[i].active > 0:
    # Process particle
```

## Performance Patterns

### Efficient Distance Checks
```python
# Early exit for distant particles
dist_sq = diff.x * diff.x + diff.y * diff.y
if dist_sq > radius_sq:
    continue  # Skip expensive calculations

# Then calculate actual distance if needed
dist = ti.sqrt(dist_sq)
```

### Spatial Optimization
```python
# Use grid-based spatial hashing for large particle counts
grid_x = ti.cast(pos.x / cell_size, ti.i32)
grid_y = ti.cast(pos.y / cell_size, ti.i32)
```

## Common Gotchas and Solutions

### ❌ NEVER: Return inside conditionals
```python
# WRONG - Causes crash
if species == 0:
    return ti.math.vec2(0.0, 0.0)  # ERROR!
```

### ✅ ALWAYS: Declare variables before use
```python
# CORRECT
force = ti.math.vec2(0.0, 0.0)  # Default
if species == 0:
    force = calculated_force
return force  # Single return
```

### ❌ NEVER: Access out of bounds
```python
# WRONG
tv.p.field[tv.pn].pos  # ERROR: max index is tv.pn-1
```

### ✅ ALWAYS: Check bounds
```python
# CORRECT
if i < tv.pn:
    pos = tv.p.field[i].pos
```
"""

PARTICLE_PATTERNS = """
# Particle Force Calculation Patterns

## Expert Function Signature
All particle force experts MUST follow this exact signature:

```python
@ti.func
def expert_name(pos: ti.math.vec2, vel: ti.math.vec2, 
                mass: ti.f32, species: ti.i32, 
                particle_idx: ti.i32) -> ti.math.vec2:
    '''Docstring describing the force behavior.'''
    force = ti.math.vec2(0.0, 0.0)  # Always declare result first
    
    # Force calculation logic...
    
    return force  # Single return at end
```

## Force Magnitude Guidelines

### Gravity/Fall
- Range: 300-800 for visible effect
- Direction: Negative Y (downward)
```python
gravity_force = ti.math.vec2(0.0, -500.0)
```

### Chase/Flee
- Range: 200-600 for responsive movement
- Scale by distance for smooth behavior
```python
chase_strength = 400.0 * (1.0 - dist / max_dist)
```

### Wander/Random
- Range: 50-200 for gentle movement
```python
wander_force = random_dir * 100.0
```

### Flocking
- Separation: 300-500 (strong avoidance)
- Alignment: 100-300 (moderate matching)
- Cohesion: 100-200 (gentle attraction)

## Integration Formula
Forces are integrated using semi-implicit Euler:
```python
# Standard integration with dt = 0.016 (60 FPS)
acceleration = total_force / mass
velocity += acceleration * dt
position += velocity * dt
```
"""

# Export all documentation
__all__ = ['PARTICLES_API', 'PARTICLE_PATTERNS']