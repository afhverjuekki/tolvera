TOLVERA_CORE_API = """
# Tölvera Core API Reference

## Global Properties
- tv.x, tv.y: Screen dimensions (width, height)
- tv.pn: Total number of particles
- tv.sn: Total number of species
- tv.ctx.i[None]: Current frame number

## Particle System (tv.p) 

### IMPORTANT: Built-in Particle Properties
The following properties are ALREADY AVAILABLE on every particle. DO NOT create custom states for these:

**Core Properties (Always Available)**:
- `tv.p.field[i].pos`: ti.math.vec2 - Position (x, y)
- `tv.p.field[i].vel`: ti.math.vec2 - Velocity (vx, vy)
- `tv.p.field[i].mass`: ti.f32 - Particle mass (default 1.0)
- `tv.p.field[i].size`: ti.f32 - Display size
- `tv.p.field[i].speed`: ti.f32 - Speed magnitude
- `tv.p.field[i].species`: ti.i32 - Species ID (0 to tv.sn-1)
- `tv.p.field[i].active`: ti.f32 - Activity level (0.0-1.0)
- `tv.p.field[i].ppos`: ti.math.vec2 - Previous position
- `tv.p.field[i].pvel`: ti.math.vec2 - Previous velocity

These properties are accessed directly in expert functions via the passed parameters or particle struct.

## Particle Methods (available on Particle struct)
- p.dist(other): Distance vector between particles
- p.dist_norm(other): Euclidean distance (scalar)
- p.dist_normalized(other): Normalized direction vector
- p.dist_wrap(other, tv.x, tv.y): Wrap-around distance for toroidal topology

## State System (tv.s)

### When to Create Custom States
Only create custom states for properties that DON'T already exist:
- ✅ Create states for: energy, home_pos, pheromone_strength, day_phase, gravity_strength
- ❌ DON'T create states for: mass, pos, vel, species (already in particle struct)

### State Categories
- **Global States**: System-wide parameters INCLUDING time-based (gravity, temperature, day_phase, time_of_day)
  - Access: `tv.s.llm_global.field[0].state_name`
  - Example ranges: gravity (0-1000), temperature (0-100), day_phase (0.0-1.0), time_of_day (0.0-24.0)
- **Particle States**: Per-particle custom data (energy, home_position, memory)
  - Access: `tv.s.llm_particle.field[i].state_name`
- **Species States**: Per-species configuration (aggression, speed_modifier)
  - Access: `tv.s.llm_species.field[species_id].state_name`

States are organized into containers that can be accessed via:
- tv.s.state_name.field[index].property
- Use tv.s.set() to create new state containers

## Coordinate System (STANDARD PHYSICS/MATH)
- Origin (0,0) is at TOP-LEFT corner of screen
- X increases rightward (0 to tv.x) - standard
- Y increases UPWARD - standard physics/math convention
- Screen corners:
  - Top-left: (0, 0)
  - Top-right: (tv.x, 0)  
  - Bottom-left: (0, tv.y)
  - Bottom-right: (tv.x, tv.y)

## Physics Conventions
- Gravity: NEGATIVE Y force (downward) because Y+ points up
  - Example: ti.math.vec2(0.0, -500.0) for normal gravity
  - Use -300 to -800 for visible gravity effects
- Upward forces: Use POSITIVE Y values
- Force scaling: Use larger values (200-1000) for visible motion
- Velocity damping: Multiply by 0.95-0.98 each frame
- Time step (dt): Typically 0.016 for 60 FPS
"""

PIXELS_API = """
# Pixels API (tv.px)

## Drawing Functions (Taichi kernel context) - EXACT SIGNATURES
- tv.px.point(x: ti.i32, y: ti.i32, rgba: vec4): Draw single pixel
- tv.px.line(x0: ti.f32, y0: ti.f32, x1: ti.f32, y1: ti.f32, rgba: vec4): Anti-aliased line
- tv.px.circle(x: ti.i32, y: ti.i32, r: ti.i32, rgba: vec4, fill: ti.i32 = 1): Circle (filled by default)
- tv.px.rect(x: ti.i32, y: ti.i32, w: ti.i32, h: ti.i32, rgba: vec4, fill: ti.i32 = 1): Rectangle
- tv.px.triangle(a: vec2, b: vec2, c: vec2, rgba: vec4, fill: ti.i32 = 1): Triangle with vec2 points

CRITICAL: Colors must be ti.math.vec4(r, g, b, a) or ti.Vector([r, g, b, a]) with values 0.0-1.0

## Pixel Field Access
- tv.px.px.rgba[x, y]: Direct pixel access (ti.math.vec4)
- tv.px.px: Pixel field (shape: tv.x × tv.y)

## Pixel Effects (call from render function)
- tv.px.diffuse(rate): Blur/diffuse pixels (0.99 = slight blur)
- tv.px.decay(rate): Fade pixels over time (0.99 = slow fade)
- tv.px.clear(): Clear all pixels to black

## Particle Rendering
- tv.px.particles(tv.p, tv.s.species()): Draw particles with species colors
- tv.px.particles(tv.p, tv.s.species(), shape='circle'): Specify shape

## Color Format
- Colors are ti.math.vec4(r, g, b, a) with range 0.0-1.0
- Examples:
  - Red: ti.math.vec4(1.0, 0.0, 0.0, 1.0)
  - Green: ti.math.vec4(0.0, 1.0, 0.0, 1.0)
  - Blue: ti.math.vec4(0.0, 0.0, 1.0, 1.0)
  - White: ti.math.vec4(1.0, 1.0, 1.0, 1.0)
  - Transparent: ti.math.vec4(r, g, b, 0.5)

## Pheromone/Trail Pattern
@ti.kernel
def deposit_pheromone():
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            x = ti.cast(tv.p.field[i].pos[0], ti.i32)
            y = ti.cast(tv.p.field[i].pos[1], ti.i32)
            if 0 <= x < tv.x and 0 <= y < tv.y:
                # Deposit pheromone based on species
                color = tv.s.species().field[tv.p.field[i].species].color
                tv.px.px.rgba[x, y] += color * 0.1
"""

TAICHI_ESSENTIALS = """
# Taichi Language Essentials for Tölvera

## Types
- Scalars: ti.f32, ti.i32, ti.u32
- Vectors: ti.math.vec2, ti.math.vec3, ti.math.vec4
- Create vector: ti.math.vec2(x, y)

## Math Functions
- Trigonometry: ti.sin(), ti.cos(), ti.tan(), ti.atan2()
- Algebra: ti.sqrt(), ti.pow(), ti.exp(), ti.log()
- Utilities: ti.abs(), ti.sign(), ti.clamp(), ti.random()
- Constants: Use 3.14159 for π (NOT ti.pi or math.pi)

## Vector Operations
- Arithmetic: vec1 + vec2, vec1 - vec2, vec * scalar
- Methods for ti.math.vec2: vec.norm(), vec.dot(other)
- Access: vec[0] or vec.x, vec[1] or vec.y
- Normalization: Manual division or ti.math.normalize()

## Common Patterns
# Safe normalization (CORRECT)
dist = vec.norm()
if dist > 0.01:
    direction = vec / dist  # Manual normalization
    # OR: direction = ti.math.normalize(vec)
else:
    direction = ti.math.vec2(1.0, 0.0)

# Distance calculation
dist = (pos2 - pos1).norm()  # Works for ti.math.vec2

# Random direction
angle = ti.random() * 2 * 3.14159
direction = ti.math.vec2(ti.cos(angle), ti.sin(angle))

## Important Notes
- ❌ NEVER: ti.pi, math.sqrt(), vec.normalized() (doesn't exist!)
- ✅ ALWAYS: 3.14159, ti.sqrt(), vec / vec.norm() or ti.math.normalize(vec)
- Check division by zero when normalizing
- Taichi is statically typed - declare types explicitly

## CRITICAL: Variable Declaration Rules
Taichi requires ALL variables to be declared before use in conditional branches:
❌ WRONG:
if condition:
    x = 5.0
else:
    x = 10.0
use(x)  # ERROR: x not defined in all code paths

✅ CORRECT:
x = 10.0  # Declare with default
if condition:
    x = 5.0
use(x)  # OK: x is always defined

## CRITICAL: No Returns Inside Conditionals/Loops
Taichi does NOT support return statements inside if/for/while blocks:
❌ WRONG:
if species == 0:
    return ti.math.vec2(0.0, 0.0)  # ERROR!
return other_value

✅ CORRECT:
result = other_value  # Default
if species == 0:
    result = ti.math.vec2(0.0, 0.0)
return result  # Single return at end
"""

STATE_ACCESS_PATTERNS = """
# State Access Patterns

## IMPORTANT: Available States
When synthesizing experts, you will be told what custom states are available.
Only use states that have been explicitly created and are listed as available.

## Global States (shared across entire system, INCLUDING time-based)
Access: tv.s.llm_global.field[0].state_name
Example: gravity = tv.s.llm_global.field[0].gravity_strength
Example: day_phase = tv.s.llm_global.field[0].day_phase
Common time-based global states:
- day_phase: 0.0-1.0 (0=midnight, 0.25=dawn, 0.5=noon, 0.75=dusk)
- time_of_day: 0.0-24.0 (hours)
- season_cycle: 0.0-1.0 (0=winter, 0.25=spring, 0.5=summer, 0.75=fall)

## Particle States (per-particle data)
Access: tv.s.llm_particle.field[particle_idx].state_name
Example: energy = tv.s.llm_particle.field[i].energy

## Species States (per-species configuration)
Access: tv.s.llm_species.field[species].state_name  
Example: aggression = tv.s.llm_species.field[species].aggression_level

## Common State Types
- Scalars: ti.f32 for continuous values
- Vectors: ti.math.vec2 for positions, directions
- Integers: ti.i32 for discrete values (grid positions, counts)
- Bounded values: Always respect min/max ranges

## Temporal Patterns
- Use modulo for cyclic behaviors: phase = (frame / cycle_length) % 1.0
- Day/night cycles: Use sin(day_phase * 3.14159) for smooth transitions
- Seasonal changes: Map season_cycle to behavior parameters
"""

BOUNDARY_HANDLING = """
# Boundary Handling Patterns

## Bounce (elastic collision)
if new_pos.x < 0 or new_pos.x > tv.x:
    tv.p.field[i].vel.x *= -0.8  # Energy loss on bounce
    new_pos.x = ti.math.clamp(new_pos.x, 0, tv.x)

## Wrap (toroidal topology)
new_pos.x = new_pos.x % tv.x
new_pos.y = new_pos.y % tv.y

## Stop (sticky boundaries)
new_pos.x = ti.math.clamp(new_pos.x, 0, tv.x)
new_pos.y = ti.math.clamp(new_pos.y, 0, tv.y)
if new_pos.x == 0 or new_pos.x == tv.x:
    tv.p.field[i].vel.x = 0
    
## Reflect with damping
if new_pos.x < 0:
    new_pos.x = -new_pos.x
    tv.p.field[i].vel.x *= -0.9
elif new_pos.x > tv.x:
    new_pos.x = 2 * tv.x - new_pos.x
    tv.p.field[i].vel.x *= -0.9
"""