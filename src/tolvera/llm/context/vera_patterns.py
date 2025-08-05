SPECIES_INTERACTION_PATTERNS = """
## CRITICAL: Proper Species Interaction Implementation

### 1. Spatial Awareness Pattern (from Flock.py)
Both species in an interaction should actively scan their environment:
```python
# CORRECT: Both predator and prey scan for each other
if species == 0:  # Predator
    for j in range(tv.pn):
        if tv.p.field[j].species == 1:  # Look for prey
            # Calculate distance and apply chase force
            
elif species == 1:  # Prey  
    for j in range(tv.pn):
        if tv.p.field[j].species == 0:  # Look for predators
            # Calculate distance and apply flee force
```

### 2. Distance-Based Detection
Use different detection radii for different behaviors:
- Predator detection: 200-300 units (hunting range)
- Prey detection: 250-350 units (slightly larger for survival)
- Flocking perception: 50-100 units (close neighbors)
- Separation radius: 20-40 units (personal space)

### 3. Force Balancing Guidelines
Proper force magnitudes create emergent behavior:
- Chase forces: 400-600 (strong but catchable)
- Flee forces: 300-500 (slightly weaker for drama)
- Flocking forces: 50-200 (gentle influences)
- Separation forces: 100-300 (avoid collisions)
- Random movement: 20-100 (idle behavior)

### 4. Direction Calculation Patterns
```python
# Chase: Move TOWARD target
direction_to_target = target_pos - my_pos
chase_force = direction_to_target.normalized() * strength

# Flee: Move AWAY from threat  
direction_from_threat = my_pos - threat_pos
flee_force = direction_from_threat.normalized() * strength

# Add randomness for natural movement
random_offset = ti.math.vec2(ti.random() - 0.5, ti.random() - 0.5) * 0.3
escape_direction = (direction_from_threat.normalized() + random_offset).normalized()
```

### 5. Idle Behavior
Always provide default behavior when no interactions occur:
```python
if no_threats_nearby and no_targets_nearby:
    # Small random movement
    force = ti.math.vec2(ti.random() - 0.5, ti.random() - 0.5) * 50.0
```

### 6. Active Particle Checks
Always verify particles are active before interaction:
```python
if tv.p.field[j].active > 0.0:  # Only interact with active particles
    # Perform interaction
```

### 7. Species Matrix Pattern (from ParticleLife)
For complex multi-species interactions, use a rule matrix:
```python
tv.s.interaction_matrix = {
    "state": {
        "attract": (ti.f32, -1.0, 1.0),  # -1 = repel, 0 = neutral, 1 = attract
        "radius": (ti.f32, 50.0, 300.0),
    },
    "shape": (tv.sn, tv.sn),  # species x species matrix
}

# Usage in expert:
rule = tv.s.interaction_matrix[species1, species2]
if dist < rule.radius:
    force = direction * rule.attract * strength
```

### 8. Neighbor Count Pattern (from Flock)
Track nearby particles for density-dependent behaviors:
```python
nearby_count = 0
for j in range(tv.pn):
    if i != j and tv.p.field[j].active > 0:
        dist = (pos - tv.p.field[j].pos).norm()
        if dist < perception_radius:
            nearby_count += 1

# Scale behavior by density
if nearby_count > 0:
    force = base_force / ti.sqrt(nearby_count + 1.0)
```
"""

VERA_PATTERNS = """
# Patterns from Tölvera Vera Module

## Slime Mold (Physarum) Pattern
Based on vera/slime.py - sensing and movement with pheromone trails:

### State Structure
tv.s.slime_p = {  # Per-particle state
    "state": {
        "sense_angle": (ti.f32, 0.0, 10.0),
        "sense_left": (ti.math.vec4, 0.0, 10.0),
        "sense_centre": (ti.math.vec4, 0.0, 10.0),
        "sense_right": (ti.math.vec4, 0.0, 10.0),
    },
    "shape": tv.pn,
}

tv.s.slime_s = {  # Per-species parameters
    "state": {
        "sense_angle": (ti.f32, 0.0, 1.0),
        "sense_dist": (ti.f32, 0.0, 1.0),
        "move_angle": (ti.f32, 0.0, 1.0),
        "move_dist": (ti.f32, 0.0, 1.0),
        "evaporate": (ti.f32, 0.0, 1.0),
    },
    "shape": tv.sn,
}

### Sensing Pattern
@ti.func
def sense(pos: ti.math.vec2, ang: ti.f32, dist: ti.f32) -> ti.math.vec4:
    ang_cos = ti.cos(ang)
    ang_sin = ti.sin(ang)
    v = ti.Vector([ang_cos, ang_sin])
    p = pos + v * dist
    px = ti.cast(p[0], ti.i32) % tv.x
    py = ti.cast(p[1], ti.i32) % tv.y
    return tv.px.px.rgba[px, py]

### Movement Decision
# Sense three directions
c = sense(p.pos, ang, sense_dist).norm()
l = sense(p.pos, ang - sense_angle, sense_dist).norm()
r = sense(p.pos, ang + sense_angle, sense_dist).norm()

# Turn based on sensed values
if l < c < r:
    ang += move_angle
elif l > c > r:
    ang -= move_angle
elif r > c and c < l:
    ang += move_angle * (2 * (ti.random() < 0.5) - 1)

## Flocking (Boids) Pattern
From vera/flock.py - alignment, cohesion, separation:

### Constants Structure
CONSTS = {
    "SEPARATION_RADIUS": (ti.f32, 25.0),
    "ALIGNMENT_RADIUS": (ti.f32, 50.0),
    "COHESION_RADIUS": (ti.f32, 50.0),
    "MAX_SPEED": (ti.f32, 4.0),
    "MAX_FORCE": (ti.f32, 0.2),
}

### Species-Specific Parameters
tv.s.flock_s = {
    "state": {
        "separate": (ti.f32, 0.0, 10.0),
        "align": (ti.f32, 0.0, 10.0),
        "cohere": (ti.f32, 0.0, 10.0),
    },
    "shape": tv.sn,
}

### Efficient Neighbor Processing
@ti.func
def process_neighbors(i: ti.i32, pos: ti.math.vec2):
    sep_sum = ti.Vector([0.0, 0.0])
    ali_sum = ti.Vector([0.0, 0.0])
    coh_sum = ti.Vector([0.0, 0.0])
    sep_count = 0
    ali_count = 0
    coh_count = 0
    
    for j in range(tv.pn):
        if i != j and tv.p.field[j].active > 0:
            other = tv.p.field[j]
            diff = pos - other.pos
            dist = diff.norm()
            
            # Separation
            if dist < SEPARATION_RADIUS and dist > 0.01:
                sep_sum += diff.normalized() / dist
                sep_count += 1
            
            # Only process same species for alignment/cohesion
            if tv.p.field[i].species == other.species:
                if dist < ALIGNMENT_RADIUS:
                    ali_sum += other.vel
                    ali_count += 1
                if dist < COHESION_RADIUS:
                    coh_sum += other.pos
                    coh_count += 1

## Force Patterns
From vera/forces.py - common force calculations:

### Gravitational Attraction
@ti.func
def gravity_force(p1: ti.template(), p2: ti.template()) -> ti.math.vec2:
    diff = p2.pos - p1.pos
    dist_sq = diff.dot(diff)
    if dist_sq > 1.0:  # Avoid singularity
        force_mag = G * p1.mass * p2.mass / dist_sq
        return diff.normalized() * force_mag
    return ti.math.vec2(0.0, 0.0)

### Spring Force
@ti.func
def spring_force(p1_pos: ti.math.vec2, p2_pos: ti.math.vec2, 
                 rest_length: ti.f32, k: ti.f32) -> ti.math.vec2:
    diff = p2_pos - p1_pos
    dist = diff.norm()
    if dist > 0.01:
        displacement = dist - rest_length
        return diff.normalized() * (k * displacement)
    return ti.math.vec2(0.0, 0.0)

### Damping Force
@ti.func
def damping_force(vel: ti.math.vec2, damping_coeff: ti.f32) -> ti.math.vec2:
    return -vel * damping_coeff

## State Initialization Pattern
From iil-examples - flexible state initialization:

# Method 1: Direct state creation
tv.s.my_state = {
    "state": {
        "energy": (ti.f32, 0.0, 100.0),
        "home_pos": (ti.math.vec2, 0.0, 1.0),
    },
    "shape": tv.pn,
    "randomise": True
}

# Method 2: Using from_vec for OSC control
states = ['species', 'custom_state']
def states_from_vec(vec: list):
    tv.s.from_vec(states, vec)

## Render Loop Pattern
Standard render function structure:

@tv.render
def _():
    # 1. Apply pixel effects
    tv.px.diffuse(0.99)  # Optional blur
    tv.px.decay(0.995)   # Optional fade
    
    # 2. Update physics/behaviors
    tv.p()  # Update particle system
    apply_behaviors()  # Custom behavior kernel
    
    # 3. Draw additional elements
    draw_trails()  # Optional trail rendering
    
    # 4. Render particles
    tv.px.particles(tv.p, tv.s.species())
    
    return tv.px
"""

INTERACTION_PATTERNS_VERA = """
# Advanced Interaction Patterns from Vera

## Multi-Species Slime Mold
Different species with different sensing parameters:

@ti.kernel
def multi_species_slime():
    for i in range(tv.pn):
        p = tv.p.field[i]
        species_params = tv.s.slime_s.field[p.species]
        
        # Species-specific sensing
        sense_angle = species_params.sense_angle * SENSE_ANGLE_BASE
        sense_dist = species_params.sense_dist * SENSE_DIST_BASE
        
        # Different species might repel each other
        for j in range(tv.pn):
            if i != j:
                other = tv.p.field[j]
                if p.species != other.species:
                    diff = p.pos - other.pos
                    dist = diff.norm()
                    if dist < REPEL_RADIUS and dist > 0.01:
                        force += diff.normalized() * (REPEL_STRENGTH / dist)

## Swarmalators Pattern
Particles that synchronize while swarming:

tv.s.swarmalator = {
    "state": {
        "phase": (ti.f32, 0.0, 2*3.14159),  # Oscillator phase
        "frequency": (ti.f32, 0.1, 2.0),    # Natural frequency
    },
    "shape": tv.pn,
}

@ti.func
def kuramoto_coupling(phase1: ti.f32, phase2: ti.f32, K: ti.f32) -> ti.f32:
    return K * ti.sin(phase2 - phase1)

## Reaction-Diffusion on Particles
Particles as mobile reaction sites:

tv.s.reaction = {
    "state": {
        "u": (ti.f32, 0.0, 1.0),  # Chemical U
        "v": (ti.f32, 0.0, 1.0),  # Chemical V
    },
    "shape": tv.pn,
}

@ti.kernel
def reaction_step():
    # Gray-Scott reaction
    for i in range(tv.pn):
        u = tv.s.reaction.field[i].u
        v = tv.s.reaction.field[i].v
        
        # Reaction terms
        uvv = u * v * v
        du = -uvv + F * (1.0 - u)
        dv = uvv - (F + K) * v
        
        # Update concentrations
        tv.s.reaction.field[i].u = u + du * dt
        tv.s.reaction.field[i].v = v + dv * dt
"""