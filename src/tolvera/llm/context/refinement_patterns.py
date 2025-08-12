"""
Refinement Patterns - Common patterns for iterative sketch refinement
"""

REFINEMENT_PATTERNS = """
# Common Refinement Patterns

## Force Magnitude Adjustments

### Making Forces Stronger
Request: "Make gravity stronger" / "Increase attraction"
Pattern: Multiply force values by scaling factor
```python
# BEFORE:
force = ti.math.vec2(0.0, -300.0)  # Gravity

# AFTER (2x stronger):
force = ti.math.vec2(0.0, -600.0)  # Gravity (doubled)
```

### Making Forces Weaker
Request: "Reduce repulsion" / "Less gravity"
Pattern: Reduce force multipliers
```python
# BEFORE:
force = direction * 100.0  # Repulsion strength

# AFTER (half strength):
force = direction * 50.0  # Repulsion strength (reduced)
```

## Speed and Movement Adjustments

### Increasing Speed
Request: "Make particles move faster"
Pattern: Increase velocity multipliers or reduce damping
```python
# BEFORE:
tv.p.field[i].vel *= 0.98  # Damping
tv.p.field[i].speed = 20.0

# AFTER (faster):
tv.p.field[i].vel *= 0.995  # Less damping (keeps more speed)
tv.p.field[i].speed = 40.0  # Higher base speed
```

### Decreasing Speed
Request: "Slow down the particles"
Pattern: Increase damping or reduce speed values
```python
# BEFORE:
tv.p.field[i].vel *= 0.98  # Damping

# AFTER (slower):
tv.p.field[i].vel *= 0.95  # More damping (loses speed faster)
```

## Visual Parameter Changes

### Size Adjustments
Request: "Make particles bigger" / "Smaller dots"
Pattern: Modify size values in initialization
```python
# BEFORE:
tv.p.field[i].size = 5.0

# AFTER (bigger):
tv.p.field[i].size = 10.0
```

### Color Changes
Request: "Change to blue" / "Make it red"
Pattern: Modify rgba values
```python
# BEFORE:
tv.s.species.field[0].rgba = [1.0, 0.6, 0.0, 1.0]  # Orange

# AFTER (blue):
tv.s.species.field[0].rgba = [0.0, 0.3, 1.0, 1.0]  # Blue
```

### Transparency
Request: "Add transparency" / "Make semi-transparent"
Pattern: Reduce alpha channel (4th component)
```python
# BEFORE:
color = ti.math.vec4(1.0, 0.0, 0.0, 1.0)  # Solid red

# AFTER (semi-transparent):
color = ti.math.vec4(1.0, 0.0, 0.0, 0.5)  # 50% transparent red
```

## Adding New Behaviors

### Adding Random Drift
Request: "Add some random movement" / "Make them wander"
Pattern: Add new expert function with random forces
```python
@ti.func
def expert_random_drift(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    angle = ti.random() * 2 * 3.14159
    force = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * 50.0
    return force

# Then add to integration kernel:
total_force += expert_random_drift(pos, vel, mass, species, i) * 0.5
```

### Adding Attraction/Repulsion
Request: "Make them attract to center" / "Repel from edges"
Pattern: Add expert with distance-based forces
```python
@ti.func
def expert_center_attraction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    center = ti.math.vec2(tv.x * 0.5, tv.y * 0.5)
    to_center = center - pos
    dist = to_center.norm()
    force = ti.math.vec2(0.0, 0.0)
    if dist > 0.001:
        direction = to_center / dist
        force = direction * 100.0
    return force
```

## Temporal/Drawing Adjustments

### Changing Drawing Frequency
Request: "Draw more often" / "Less frequent updates"
Pattern: Adjust time thresholds or frame counters
```python
# BEFORE:
time_threshold = 2.0  # Every 2 seconds

# AFTER (more frequent):
time_threshold = 0.5  # Every 0.5 seconds
```

### Modifying Oscillations
Request: "Faster pulsing" / "Slower waves"
Pattern: Adjust frequency values
```python
# BEFORE:
frequency = 1.0  # 1 Hz

# AFTER (faster):
frequency = 3.0  # 3 Hz
```

## Common Bug Fixes

### Particles Not Moving
Issue: Forces calculated but particles static
Fix: Ensure position is updated from velocity
```python
# MISSING LINE (add this):
tv.p.field[i].pos += tv.p.field[i].vel * tv.p.field[i].speed * dt
```

### Division by Zero
Issue: Crash when particles overlap
Fix: Add safety check before division
```python
# BEFORE (crashes):
direction = diff / dist

# AFTER (safe):
if dist > 0.001:
    direction = diff / dist
else:
    direction = ti.math.vec2(0.0, 0.0)
```

### Return in Conditional (CRITICAL)
Issue: "Return inside non-static if" crash
Fix: Move return outside conditional
```python
# BEFORE (CRASHES):
if species == 0:
    return chase_force
else:
    return flee_force

# AFTER (CORRECT):
force = ti.math.vec2(0.0, 0.0)
if species == 0:
    force = chase_force
else:
    force = flee_force
return force
```

### Drawing Not Visible
Issue: Drawing function exists but nothing appears
Fix: Ensure draw kernel is called in render loop
```python
# Add to render function:
draw()  # Execute drawing kernel
```

## Multi-Species Adjustments

### Species-Specific Changes
Request: "Make red species faster than blue"
Pattern: Conditional logic based on species ID
```python
# In expert function:
speed_multiplier = 1.0
if species == 0:  # Red species
    speed_multiplier = 2.0
elif species == 1:  # Blue species
    speed_multiplier = 0.5
force = base_force * speed_multiplier
```

### Adding Species Interactions
Request: "Make species 0 chase species 1"
Pattern: Add interaction expert
```python
@ti.func
def expert_chase_interaction(p1_idx: ti.i32, p2_idx: ti.i32) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    p1 = tv.p.field[p1_idx]
    p2 = tv.p.field[p2_idx]
    
    if p1.species == 0 and p2.species == 1:
        to_target = p2.pos - p1.pos
        dist = to_target.norm()
        if dist > 0.001 and dist < 200.0:
            direction = to_target / dist
            force = direction * 150.0
    
    return force
```

## Performance Optimizations

### Reducing Computation
Request: "Running too slow"
Pattern: Add distance cutoffs, reduce particle checks
```python
# Add distance check to skip far particles:
if dist > perception_radius:
    continue  # Skip this particle
```

### Adjusting Time Step
Request: "More stable simulation"
Pattern: Reduce dt value
```python
# BEFORE:
dt = 0.016  # 60 FPS

# AFTER (more stable):
dt = 0.008  # Smaller time step
```
"""

ERROR_FIX_PATTERNS = """
# Common Error Fixes

## Taichi-Specific Crashes

### "Return inside non-static if"
The most common Taichi crash - NEVER use return inside conditionals
```python
# WRONG (CRASHES):
@ti.func
def expert_function(...) -> ti.math.vec2:
    if condition:
        return some_force  # CRASH!
    return other_force

# CORRECT:
@ti.func
def expert_function(...) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)  # Declare result
    if condition:
        force = some_force  # Set result
    else:
        force = other_force
    return force  # Single return at end
```

### "Division by zero"
When normalizing vectors or dividing by distance
```python
# WRONG:
direction = diff / diff.norm()  # Crashes if diff is zero

# CORRECT:
dist = diff.norm()
if dist > 0.001:
    direction = diff / dist
else:
    direction = ti.math.vec2(0.0, 0.0)
```

### "Index out of bounds"
Accessing arrays or pixel field
```python
# WRONG:
tv.px.px.rgba[x, y] = color  # May be out of bounds

# CORRECT:
if 0 <= x < tv.x and 0 <= y < tv.y:
    tv.px.px.rgba[x, y] = color
```

## Physics Issues

### Gravity Wrong Direction
Remember: Y+ points UPWARD in standard physics
```python
# WRONG (particles fall up):
gravity = ti.math.vec2(0.0, 500.0)  # Positive Y

# CORRECT (particles fall down):
gravity = ti.math.vec2(0.0, -500.0)  # Negative Y
```

### Forces Too Weak
Tölvera needs larger force values for visible motion
```python
# Too weak (barely visible):
force = direction * 10.0

# Better (clear motion):
force = direction * 200.0  # Or 100-1000 range
```

### Particles Escape Screen
Add boundary checks or wrapping
```python
# Wrapping:
tv.p.field[i].pos[0] = tv.p.field[i].pos[0] % tv.x
tv.p.field[i].pos[1] = tv.p.field[i].pos[1] % tv.y

# Or bouncing:
if tv.p.field[i].pos[0] < 0 or tv.p.field[i].pos[0] > tv.x:
    tv.p.field[i].vel[0] *= -1
```

## Drawing Issues

### Drawing Doesn't Appear
Check that drawing kernel is called
```python
# In render function:
draw()  # Must call the drawing kernel!
```

### Wrong Coordinate System
Drawing uses integer pixel coordinates
```python
# Convert float positions to int:
x = ti.cast(pos[0], ti.i32)
y = ti.cast(pos[1], ti.i32)
```

### Colors Not Visible
Ensure RGBA values are 0.0-1.0 range
```python
# WRONG:
color = ti.math.vec4(255, 0, 0, 255)  # 0-255 range

# CORRECT:
color = ti.math.vec4(1.0, 0.0, 0.0, 1.0)  # 0.0-1.0 range
```
"""