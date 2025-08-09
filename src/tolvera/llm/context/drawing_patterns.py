DRAWING_PATTERNS = """
## Drawing Expert Patterns (Based on Tölvera Examples)

Drawing operations work with the pixel buffer (tv.px) to create visual effects.

### Basic Drawing Operations

1. **Pixel Operations**:
   - NOT directly setting pixels - use drawing primitives instead
   - Colors as `ti.Vector([r, g, b, a])` or `ti.math.vec4(r, g, b, a)` where values are 0.0-1.0

2. **Line Drawing**:
   - `tv.px.line(x1, y1, x2, y2, color)` - Draw a line between two points
   - Coordinates are automatically cast to integers internally
   - Example: `tv.px.line(p1.pos.x, p1.pos.y, p2.pos.x, p2.pos.y, ti.Vector([1., 0., 0., 1.]))`

3. **Circle Drawing**:
   - `tv.px.circle(x, y, radius, color)` - Draw a filled circle
   - Center coordinates and radius can be float, internally cast
   - Example: `tv.px.circle(p.pos.x, p.pos.y, 5, species_color)`

4. **Rectangle Drawing**:
   - `tv.px.rect(x, y, width, height, color)` - Draw a filled rectangle
   - Example: `tv.px.rect(tv.x/2-50, tv.y/2-50, 100, 100, ti.Vector([1., 0., 0., 1.]))`

### Common Drawing Patterns (From Examples)

#### Velocity/Force Visualization (from see_flock.py)
```python
@ti.kernel
def draw_forces():
    n = tv.p.field.shape[0]
    for i in range(n):
        if tv.p.field[i].active == 0:
            continue
        p1 = tv.p.field[i]
        fp = tv.s.flock_p[i]  # Access behavior-specific state
        
        # Color codes for different forces
        red = ti.Vector([1., 0., 0., 1.])
        green = ti.Vector([0., 1., 0., 1.])
        blue = ti.Vector([0., 0., 1., 1.])
        
        # Draw separation force (red)
        tv.px.line(
            p1.pos.x, p1.pos.y,
            p1.pos.x + fp.separate.x,
            p1.pos.y + fp.separate.y,
            red)
        
        # Draw alignment force (green)
        tv.px.line(
            p1.pos.x, p1.pos.y,
            p1.pos.x + fp.align.x * 100.,
            p1.pos.y + fp.align.y * 100.,
            green)
        
        # Draw cohesion force (blue)
        tv.px.line(
            p1.pos.x, p1.pos.y,
            p1.pos.x + fp.cohere.x,
            p1.pos.y + fp.cohere.y,
            blue)
        
        # Draw perception radius
        c = tv.s.species[p1.species].rgba
        tv.px.circle(p1.pos.x, p1.pos.y, fp.nearby/10, c)
```

#### Particle Drawing with Colors (from swarm.py)
```python
@ti.kernel
def draw_swarm_particles(particles: ti.template()):
    '''Draw particles with phase-based coloring'''
    for i in range(tv.p.n):
        p = particles.field[i]
        ps = tv.s.swarm_p[i]  # Access swarm-specific state
        if p.active == 0.0:
            continue
            
        px = ti.cast(p.pos[0], ti.i32)
        py = ti.cast(p.pos[1], ti.i32)
        
        # Use custom color mapping based on state
        rgba = color_map_1d(ps.color, .1, .3, .7) * tv.px.CONSTS.BRIGHTNESS
        tv.px.circle(px, py, 3, rgba)

@ti.func
def color_map_1d(val, r=0., g=0., b=0.):
    '''Weighted colormap for visualization'''
    val = 1 - val
    r = ti.max(.2, 1 - ti.abs(val - r))
    g = ti.max(.2, 1 - ti.abs(val - g))
    b = ti.max(.2, 1 - ti.abs(val - b))
    return ti.Vector([r, g, b, 1])
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

### Grid Visualization (from draw_species.py)

#### Drawing Species Interaction Matrix
```python
@ti.kernel
def draw_matrix(x: ti.i32, y: ti.i32, param: ti.template()):
    '''Visualize species interaction parameters as grid'''
    w, h = 75, 75
    _y = tv.y - y - h  # Flip y coordinate
    xgap = w // tv.sn
    ygap = h // tv.sn
    
    for ix in range(tv.sn):
        xoff = x + ix * xgap
        xc = tv.s.species[ix].rgba  # Species color
        r = xgap // 4
        
        # Draw species indicator at top
        tv.px.circle(xoff + xgap//2 - r/2, _y + h + r, r, xc)
        
        for iy in range(tv.sn):
            iyi = tv.sn - iy - 1  # Flip y
            yoff = _y + iy * ygap
            yc = tv.s.species[iyi].rgba
            
            # Get interaction parameter value
            param_val = tv.s.flock_s.field[ix, iyi][param]
            fc = ti.Vector([param_val, param_val, param_val, 1])
            
            # Draw cell
            tv.px.rect(xoff, yoff, xgap-5, ygap-5, fc)
            
            # Draw species indicator on left
            if ix == 0:
                tv.px.circle(xoff - r*2, yoff + ygap//2 - r/2, r, yc)
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

### Pixel Buffer Operations

#### Common Effects
```python
@tv.render
def _():
    # Diffusion effect (blur/spread)
    tv.px.diffuse(0.99)
    
    # Decay effect (fade)
    tv.px.decay(0.995)
    
    # Clear buffer
    tv.px.clear()
    
    # Apply behaviors
    tv.v.flock(tv.p)
    
    # Draw particles with species colors
    tv.px.particles(tv.p, tv.s.species())
    
    # Blend with another pixel buffer
    # tv.px.blend_mix(other_px, 0.5)
    
    return tv.px
```

#### Species Colors Access
```python
# Access species colors in kernels
@ti.kernel
def draw_with_species_colors():
    for i in range(tv.pn):
        p = tv.p.field[i]
        if p.active > 0:
            # Get species color
            color = tv.s.species[p.species].rgba
            tv.px.circle(p.pos.x, p.pos.y, 5, color)
```

### Important Constraints

1. **Coordinate Conversion**: Always cast float coordinates to integers
2. **Bounds Checking**: Ensure drawing stays within pixel buffer bounds
3. **Alpha Blending**: Use alpha channel for transparency effects
4. **Performance**: Limit complex calculations in drawing functions
5. **Color Range**: Keep RGBA values between 0.0 and 1.0
"""

DRAWING_API_REFERENCE = """
## Pixel Buffer Drawing API (Tölvera)

The pixel buffer `tv.px` provides methods for drawing operations:

### Basic Methods
- `tv.px.clear()` - Clear the entire buffer
- `tv.px.diffuse(factor)` - Apply diffusion/blur effect (0.99 typical)
- `tv.px.decay(factor)` - Apply decay/fade effect (0.995 typical)
- `tv.px.particles(particles, species)` - Draw all particles with species colors

### Drawing Primitives (in @ti.kernel functions)
- `tv.px.line(x1, y1, x2, y2, color)` - Draw line
- `tv.px.circle(x, y, radius, color)` - Draw filled circle  
- `tv.px.rect(x, y, width, height, color)` - Draw filled rectangle
- Coordinates can be float and are cast internally

### Blending Operations
- `tv.px.blend_mix(other_px, factor)` - Blend with another Pixels object
- `tv.px.set(other_px)` - Replace with another Pixels object
- `tv.px.from_img(path)` - Load image into pixels

### Coordinate System
- Origin (0, 0) at top-left corner
- X axis: left to right (0 to tv.x)
- Y axis: **top to bottom** (0 to tv.y) - screen coordinates
- Use modulo for wrapping: `px = ti.cast(pos.x, ti.i32) % tv.x`
"""