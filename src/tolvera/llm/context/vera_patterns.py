SPECIES_INTERACTION_PATTERNS = """
## CRITICAL: Proper Species Interaction Implementation

### FUNDAMENTAL PRINCIPLE: Force → Velocity → Position Chain
The Tölvera particle system follows this flow:
1. **Experts return FORCE vectors** (ti.math.vec2)
2. **Forces are applied to particle velocity**: `vel += force * dt`
3. **Position updates use velocity AND speed**: `pos += vel * speed * dt`

The `speed` attribute (lines 221-224 in particles.py) is CRITICAL:
- Each particle has individual `speed` attribute (randomized 0.1-1.0)
- Position updates: `vel * speed` determines actual movement
- `limit_speed()` constrains velocity magnitude based on species speed

### CORRECT Expert Function Pattern:
```python
@ti.func
def expert_name(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    # 1. Declare force variable FIRST
    force = ti.math.vec2(0.0, 0.0)
    
    # 2. Calculate desired force (NOT velocity!)
    if species == 0:  # Predator
        # Find target and calculate FORCE toward it
        target_force = calculate_chase_force(pos, vel, mass)
        force = target_force
    elif species == 1:  # Prey
        # Calculate FORCE away from threat
        escape_force = calculate_flee_force(pos, vel, mass)
        force = escape_force
    
    # 3. Return force - integration kernel handles vel and pos updates
    return force
```

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

### 3. Force Magnitude Guidelines (Working with Speed System)
Forces interact with particle `speed` attribute in position updates:

**Understanding the Chain**: 
- Expert returns force → `vel += force * dt` → `pos += vel * speed * dt`
- Higher `speed` = more movement per unit velocity
- Forces should be scaled appropriately for the speed range (0.1-1.0)

**Recommended Force Magnitudes**:
- **Chase forces**: 300-500 (predators catch prey but not instantly)
- **Flee forces**: 400-600 (prey escapes but creates chase dynamic)  
- **Flocking cohesion**: 50-150 (gentle group attraction)
- **Flocking separation**: 200-400 (avoid collisions effectively)
- **Random wander**: 30-100 (subtle idle movement)
- **Gravity**: 100-300 (constant downward force)
- **Attraction to point**: 150-300 (pulls particles toward target)

**Force-Speed Interaction Example**:
```python
# Slow particles (speed=0.2) with force=300: movement = vel * 0.2 * dt
# Fast particles (speed=0.8) with force=300: movement = vel * 0.8 * dt
# Same force, different actual speeds due to particle speed attribute
```

### 4. Direction Calculation Patterns
```python
# Chase: Move TOWARD target
direction_to_target = target_pos - my_pos
dist_to_target = direction_to_target.norm()
if dist_to_target > 0.001:
    chase_force = (direction_to_target / dist_to_target) * strength
else:
    chase_force = ti.math.vec2(0.0, 0.0)

# Flee: Move AWAY from threat  
direction_from_threat = my_pos - threat_pos
dist_from_threat = direction_from_threat.norm()
if dist_from_threat > 0.001:
    flee_force = (direction_from_threat / dist_from_threat) * strength
else:
    flee_force = ti.math.vec2(0.0, 0.0)

# Add randomness for natural movement
random_offset = ti.math.vec2(ti.random() - 0.5, ti.random() - 0.5) * 0.3
if dist_from_threat > 0.001:
    normalized_threat = direction_from_threat / dist_from_threat
    escape_direction = normalized_threat + random_offset
    escape_norm = escape_direction.norm()
    if escape_norm > 0.001:
        escape_direction = escape_direction / escape_norm
else:
    escape_direction = random_offset
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

### 9. VERA FORCE API COMPATIBILITY
Our experts must be compatible with Tölvera's underlying force system:

**From vera/forces.py - Key Patterns:**
```python
# attract_particle() pattern - returns velocity change
@ti.func
def attract_particle(p: Particle, pos: ti.math.vec2, mass: ti.f32, radius: ti.f32) -> ti.math.vec2:
    target_distance = (pos - p.pos).norm()
    vel = ti.Vector([0.0, 0.0])
    if target_distance < radius:
        factor = (radius - target_distance) / radius
        vel = (pos - p.pos).normalized() * mass * factor
    return vel
```

**Our Expert Functions Should Follow Similar Patterns:**
```python
@ti.func
def expert_chase(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    
    if species == 0:  # Predator chases
        nearest_prey_pos = ti.math.vec2(0.0, 0.0)
        min_dist = 1000.0
        found_prey = False
        
        # Scan for prey (similar to vera pattern)
        for j in range(tv.pn):
            if tv.p.field[j].species == 1 and tv.p.field[j].active > 0.0:
                dist = (tv.p.field[j].pos - pos).norm()
                if dist < 300.0 and dist < min_dist:  # Detection radius
                    min_dist = dist
                    nearest_prey_pos = tv.p.field[j].pos
                    found_prey = True
        
        # Apply chase force (similar to vera attract)
        if found_prey and min_dist > 0.001:
            direction = (nearest_prey_pos - pos) / min_dist  # normalized
            # Scale by distance like vera forces
            factor = 1.0 if min_dist < 50.0 else (300.0 - min_dist) / 250.0
            force = direction * 400.0 * factor
    
    return force
```

**Key Vera-Compatible Principles:**
- Always check `.active > 0.0` before processing particles
- Use `.norm()` for distances, manual normalization for directions
- Apply distance-based scaling factors
- Return force vectors that work with speed system
- Follow vera's radius-based interaction patterns
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

## Using IML (Interactive Machine Learning)
Pattern from states_from_vec.py:

```python
# Map particle positions to behavior parameters
states = ['species', 'flock_s']
def states_from_vec(vec: list):
    tv.s.from_vec(states, vec)

tv.iml.particles_pos2states = {
    'size': ((tv.pn, 2), tv.s.get_size(states)),
    'io': (tv.p.get_pos_all_2d, states_from_vec),
    'randomise': True,
}

# Or map flock parameters dynamically
tv.iml.flock_p2flock_s = {
    'type': 'fun2fun',
    'size': (tv.s.flock_p.size, tv.s.flock_s.size),
    'io': (tv.s.flock_p.to_vec, tv.s.flock_s.from_vec),
    'randomise': True,
    'update_rate': tv.ti.fps,
}
```

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