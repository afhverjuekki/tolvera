DRAWING_PATTERNS = """
## Drawing Expert Patterns

Drawing experts create visual effects, trails, overlays, and visualizations.

### Basic Drawing Operations

1. **Pixel Operations**:
   - `px.set(x, y, color)` - Set a single pixel
   - Colors as `ti.Vector([r, g, b, a])` where values are 0.0-1.0

2. **Line Drawing**:
   - `px.line(x1, y1, x2, y2, color)` - Draw a line between two points
   - Coordinates must be integers (use `ti.cast(value, ti.i32)`)

3. **Circle Drawing**:
   - `px.circle(x, y, radius, color)` - Draw a filled circle
   - Center coordinates and radius must be integers

4. **Rectangle Drawing**:
   - `px.rect(x, y, width, height, color)` - Draw a filled rectangle
   - All parameters must be integers

### Common Drawing Patterns

#### Velocity Visualization
```python
@ti.func
def expert_velocity_lines(px: ti.template(), p: ti.template(), i: ti.i32):
    scale = 50.0
    start_x = ti.cast(p.pos[0], ti.i32)
    start_y = ti.cast(p.pos[1], ti.i32)
    end_x = ti.cast(p.pos[0] + p.vel[0] * scale, ti.i32)
    end_y = ti.cast(p.pos[1] + p.vel[1] * scale, ti.i32)
    color = ti.Vector([0.0, 0.5, 1.0, 0.8])
    px.line(start_x, start_y, end_x, end_y, color)
```

#### Speed-Based Effects
```python
@ti.func
def expert_speed_halos(px: ti.template(), p: ti.template(), i: ti.i32):
    speed = p.vel.norm()
    if speed > 0.01:
        radius = ti.cast(speed * 100.0, ti.i32)
        x = ti.cast(p.pos[0], ti.i32)
        y = ti.cast(p.pos[1], ti.i32)
        intensity = ti.min(speed * 10.0, 1.0)
        color = ti.Vector([intensity, intensity * 0.5, 0.0, 0.3])
        px.circle(x, y, radius, color)
```

#### Trail Effects
```python
@ti.func
def expert_motion_trails(px: ti.template(), p: ti.template(), i: ti.i32):
    # Draw a fading trail behind the particle
    trail_length = 5
    for j in range(trail_length):
        factor = 1.0 - (j / trail_length)
        x = ti.cast(p.pos[0] - p.vel[0] * j * 10, ti.i32)
        y = ti.cast(p.pos[1] - p.vel[1] * j * 10, ti.i32)
        if 0 <= x < px.shape[0] and 0 <= y < px.shape[1]:
            color = ti.Vector([1.0, 1.0, 1.0, factor * 0.5])
            px.set(x, y, color)
```

### Interaction Drawing Patterns

#### Connection Lines
```python
@ti.func
def expert_proximity_lines(px: ti.template(), p1: ti.template(), p2: ti.template()):
    dist = (p1.pos - p2.pos).norm()
    if dist < 0.1:  # Within proximity threshold
        x1 = ti.cast(p1.pos[0], ti.i32)
        y1 = ti.cast(p1.pos[1], ti.i32)
        x2 = ti.cast(p2.pos[0], ti.i32)
        y2 = ti.cast(p2.pos[1], ti.i32)
        intensity = 1.0 - (dist / 0.1)
        color = ti.Vector([intensity, intensity, intensity, 0.5])
        px.line(x1, y1, x2, y2, color)
```

#### Collision Effects
```python
@ti.func
def expert_collision_ripples(px: ti.template(), p1: ti.template(), p2: ti.template()):
    dist = (p1.pos - p2.pos).norm()
    if dist < p1.radius + p2.radius:
        center = (p1.pos + p2.pos) * 0.5
        x = ti.cast(center[0], ti.i32)
        y = ti.cast(center[1], ti.i32)
        radius = ti.cast(dist * 50.0, ti.i32)
        color = ti.Vector([1.0, 0.5, 0.0, 0.6])
        px.circle(x, y, radius, color)
```

### Color Mapping Patterns

#### State-to-Color Mapping
```python
@ti.func
def expert_energy_coloring(p: ti.template()) -> ti.math.vec4:
    energy = p.vel.norm() * p.mass
    normalized = ti.min(energy / 0.1, 1.0)
    return ti.Vector([
        normalized,           # Red for high energy
        0.0,                 # No green
        1.0 - normalized,    # Blue for low energy
        1.0                  # Full opacity
    ])
```

#### Species-Based Colors
```python
@ti.func
def expert_species_colors(p: ti.template()) -> ti.math.vec4:
    if p.species == 0:
        return ti.Vector([1.0, 0.2, 0.2, 1.0])  # Red
    elif p.species == 1:
        return ti.Vector([0.2, 0.2, 1.0, 1.0])  # Blue
    elif p.species == 2:
        return ti.Vector([0.2, 1.0, 0.2, 1.0])  # Green
    else:
        return ti.Vector([1.0, 1.0, 0.2, 1.0])  # Yellow
```

### Important Constraints

1. **Coordinate Conversion**: Always cast float coordinates to integers
2. **Bounds Checking**: Ensure drawing stays within pixel buffer bounds
3. **Alpha Blending**: Use alpha channel for transparency effects
4. **Performance**: Limit complex calculations in drawing functions
5. **Color Range**: Keep RGBA values between 0.0 and 1.0
"""

DRAWING_API_REFERENCE = """
## Pixel Buffer Drawing API

The pixel buffer `px` provides methods for drawing operations:

### Basic Methods
- `px.set(x: int, y: int, color: vec4)` - Set a single pixel
- `px.get(x: int, y: int) -> vec4` - Get pixel color
- `px.clear()` - Clear the entire buffer
- `px.diffuse(factor: float)` - Apply diffusion effect

### Drawing Primitives
- `px.line(x1: int, y1: int, x2: int, y2: int, color: vec4)` - Draw line
- `px.circle(x: int, y: int, radius: int, color: vec4)` - Draw filled circle
- `px.rect(x: int, y: int, width: int, height: int, color: vec4)` - Draw filled rectangle

### Blending Operations
- `px.blend_mix(other_px, factor: float)` - Blend with another pixel buffer
- Alpha blending is automatic when using colors with alpha < 1.0

### Coordinate System
- Origin (0, 0) at top-left corner
- X axis: left to right (0 to tv.x)
- Y axis: top to bottom (0 to tv.y)
- Use `tv.norm_x(x)` and `tv.norm_y(y)` for normalized coordinates
"""