"""
Comprehensive Tölvera Core API Documentation
============================================
This module provides the complete core API reference for Tölvera,
including global properties, coordinate system, and state management.
"""

TOLVERA_CORE_API = """
# Tölvera Core API Reference

## Global Tölvera Instance (tv)

The `tv` object is your main interface to the Tölvera system, providing access to all subsystems.

### Core Properties
```python
tv.x          # Screen width in pixels (e.g., 1920)
tv.y          # Screen height in pixels (e.g., 1080)
tv.pn         # Total number of particles
tv.sn         # Total number of species
tv.ctx.i[None]  # Current frame number (increments each render)
```

### Subsystem Access
```python
tv.p          # Particle system (Particles instance)
tv.px         # Pixel/drawing system (Pixels instance)
tv.s          # State system (State instance)
```

## Coordinate System (CRITICAL - Standard Physics Convention)

Tölvera uses standard physics/mathematics coordinate conventions:

```
Origin (0,0) at TOP-LEFT
X-axis: Increases rightward (0 → tv.x)
Y-axis: Increases UPWARD (0 → tv.y) - Standard physics convention

Screen corners:
┌─────────────────┐
│(0,0)      (tv.x,0)│
│                  │
│                  │
│(0,tv.y)  (tv.x,tv.y)│
└─────────────────┘
```

### Physics Implications
```python
# Gravity pulls DOWNWARD (negative Y)
gravity = ti.math.vec2(0.0, -500.0)  # -Y for downward

# Jump/Rise is UPWARD (positive Y)
jump_force = ti.math.vec2(0.0, 300.0)  # +Y for upward

# Standard angle conventions apply
angle = 0        # Points right (+X)
angle = π/2      # Points up (+Y)
angle = π        # Points left (-X)
angle = 3π/2     # Points down (-Y)
```

## State System (tv.s)

The state system manages custom data beyond built-in particle properties.

### State Categories

#### Global States (System-wide parameters)
```python
# Access: tv.s.llm_global.field[0].state_name
# Examples: gravity, temperature, time_of_day, season

# Common global states
tv.s.llm_global.field[0].gravity_strength     # 0-1000
tv.s.llm_global.field[0].temperature          # 0-100
tv.s.llm_global.field[0].day_phase           # 0.0-1.0 (0=midnight, 0.5=noon)
tv.s.llm_global.field[0].time_of_day         # 0.0-24.0 hours
```

#### Particle States (Per-particle custom data)
```python
# Access: tv.s.llm_particle.field[particle_idx].state_name
# Examples: energy, home_position, memory_state

# IMPORTANT: Don't recreate built-in properties as states!
# ❌ BAD: Creating 'position' or 'velocity' states (already in tv.p.field[i])
# ✅ GOOD: Creating 'energy' or 'pheromone_strength' states
```

#### Species States (Per-species configuration)
```python
# Access: tv.s.llm_species.field[species_id].state_name
# Examples: aggression_level, reproduction_rate, diet_type

# Can also be 2D for species interactions
tv.s.llm_species.field[species1, species2].interaction_strength
```

### Creating States (Python Scope)
```python
# Define state container
tv.s.set('llm_global', {
    'state': {
        'gravity': (ti.f32, 0.0, 1000.0),      # (type, min, max)
        'temperature': (ti.f32, -50.0, 50.0),
        'day_phase': (ti.f32, 0.0, 1.0),
    },
    'shape': 1,  # Single global instance
    'osc': ('get', 'set'),  # OSC control
    'randomise': False
})

# Per-particle states
tv.s.set('llm_particle', {
    'state': {
        'energy': (ti.f32, 0.0, 100.0),
        'home_pos': (ti.math.vec2, 0.0, 1.0),  # Normalized position
    },
    'shape': tv.pn,  # One per particle
    'osc': ('get',),
    'randomise': True  # Initialize with random values
})
```

## Species System

### Species Properties
```python
tv.s.species.field[i].rgba      # Species color (vec4)
tv.s.species.field[i].size      # Default size
tv.s.species.field[i].mass      # Default mass
tv.s.species.field[i].speed     # Default speed
```

### Species Initialization Pattern
```python
@ti.kernel
def init_species_colors():
    # Predator - Red
    tv.s.species.field[0].rgba = ti.math.vec4(1.0, 0.2, 0.2, 1.0)
    # Prey - Green
    tv.s.species.field[1].rgba = ti.math.vec4(0.2, 1.0, 0.2, 1.0)
    # Neutral - Blue
    tv.s.species.field[2].rgba = ti.math.vec4(0.2, 0.2, 1.0, 1.0)
```

## Frame and Time

### Frame Counter
```python
# Get current frame number
frame = tv.ctx.i[None]

# Use for time-based effects
t = ti.cast(frame, ti.f32) * 0.016  # Convert to seconds (60 FPS)

# Cyclic behaviors
cycle = (frame % 300) / 300.0  # 5-second cycle at 60 FPS
```

### Common Time Patterns
```python
# Sine wave oscillation
wave = ti.sin(t * frequency) * amplitude

# Sawtooth wave
sawtooth = (frame % period) / period

# Square wave
square = 1.0 if (frame // period) % 2 == 0 else 0.0
```

## Force Magnitude Guidelines

Proper force scaling ensures visible, realistic motion:

```python
# Gravity
gravity_force = -500.0       # Strong downward pull
weak_gravity = -200.0        # Gentle drift

# Movement
walk_speed = 100.0           # Slow movement
run_speed = 300.0           # Fast movement
max_speed = 500.0           # Speed limit

# Interactions
chase_force = 400.0         # Strong pursuit
flee_force = 600.0         # Urgent escape
wander_force = 50.0        # Gentle drift

# Flocking
separation = 500.0         # Strong avoidance
alignment = 200.0         # Moderate matching
cohesion = 150.0         # Gentle attraction
```

## Integration Parameters

```python
# Standard time step for 60 FPS
dt = 0.016

# Semi-implicit Euler integration
acceleration = force / mass
velocity += acceleration * dt
position += velocity * dt

# With damping
velocity *= 0.98  # 2% energy loss per frame
```

## Boundary Conditions

### Toroidal Wrapping (Most Common)
```python
# Particles wrap around edges
new_pos.x = new_pos.x % tv.x
new_pos.y = new_pos.y % tv.y
```

### Bounce/Reflection
```python
if new_pos.x < 0 or new_pos.x > tv.x:
    vel.x *= -0.9  # Reverse with energy loss
    new_pos.x = ti.math.clamp(new_pos.x, 0, tv.x)
```

### Soft Boundaries (Force-based)
```python
margin = 50.0
if pos.x < margin:
    force.x += (1.0 - pos.x/margin) * repulsion_strength
```

## Critical Rules for Taichi

### ✅ ALWAYS: Declare variables before conditionals
```python
force = ti.math.vec2(0.0, 0.0)  # Declare with default
if condition:
    force = calculated_value     # Modify in branch
return force                     # Single return
```

### ❌ NEVER: Return inside conditionals
```python
# WRONG - Causes crash
if species == 0:
    return ti.math.vec2(0.0, 0.0)  # ERROR!
```

### ✅ ALWAYS: Check array bounds
```python
if 0 <= x < tv.x and 0 <= y < tv.y:
    tv.px.px.rgba[x, y] = color
```

### ✅ ALWAYS: Handle division by zero
```python
dist = diff.norm()
if dist > 0.001:  # Avoid div by zero
    direction = diff / dist
```
"""

# Export documentation
__all__ = ['TOLVERA_CORE_API']