TAICHI_FUNDAMENTALS = """
# Taichi Fundamentals for Tölvera

## Physics Conventions
Tölvera uses STANDARD PHYSICS/MATH coordinates:
- Origin (0,0) is at the TOP-LEFT corner of the screen
- X increases to the RIGHT (positive X = rightward)
- Y increases UPWARD (positive Y = upward) - standard physics convention
  
Consequences for physics:
- Gravity pulls particles DOWN = NEGATIVE Y direction
- Example gravity: ti.math.vec2(0.0, -500.0) # Negative Y!
- Jumping/upward motion = POSITIVE Y velocity
- Use force values 300-800 for visible effects (with appropriate sign)
- Screen dimensions: tv.x (width), tv.y (height)

## Kernels vs Functions
- @ti.kernel: Entry points called from Python scope
  - MUST type hint arguments and return values
  - Can only return ONE value (scalar, vector, matrix, or struct)
  - Cannot be called from other kernels or functions
  - Example:
    @ti.kernel
    def apply_forces(dt: ti.f32) -> ti.f32:
        total_energy = 0.0
        for i in range(tv.pn):
            # Process particles
        return total_energy

- @ti.func: Building blocks called from Taichi scope
  - Type hints recommended but not required
  - Can return multiple values
  - Cannot be called from Python scope
  - Force-inlined at compile time
  - Example:
    @ti.func
    def compute_force(p1: ti.template(), p2: ti.template()) -> ti.math.vec2:
        diff = p2.pos - p1.pos
        dist = diff.norm()
        if dist > 0.001:
            return (diff / dist) * 100.0  # Manual normalization
        else:
            return ti.math.vec2(0.0, 0.0)

## Field Access Patterns
### Scalar Fields
f_0d = ti.field(ti.f32, shape=())     # 0D field - access with f_0d[None]
f_1d = ti.field(ti.f32, shape=(n,))   # 1D field - access with f_1d[i]
f_2d = ti.field(ti.f32, shape=(h,w))  # 2D field - access with f_2d[i,j]

### Vector Fields
# N-dimensional vectors in M-dimensional fields
vec_field = ti.Vector.field(n=2, dtype=ti.f32, shape=(100, 100))
vec_field[i, j][0]  # Access x component
vec_field[i, j].x   # Alternative for n<=4
vec_field[i, j].xy  # Access multiple components

### Struct Fields (Tölvera Particles)
# Access particle properties
tv.p.field[i].pos    # Position (ti.math.vec2)
tv.p.field[i].vel    # Velocity (ti.math.vec2)
tv.p.field[i].mass   # Mass (ti.f32)
tv.p.field[i].size   # Display size (ti.f32)
tv.p.field[i].species # Species ID (ti.i32)
tv.p.field[i].active  # Activity level (ti.f32)

# Or in kernels, use template parameter:
@ti.kernel
def process(particles: ti.template()):
    p = particles[i]
    pos = p.pos
    vel = p.vel

## Loop Patterns
### Basic Range Loop
@ti.kernel
def process():
    for i in range(n):
        # Process element i
        
### Field Loop (Automatic)
@ti.kernel
def process_field():
    for i, j in field_2d:  # Automatically loops over all indices
        field_2d[i, j] = ti.random()

### Nested Loops (Pair-wise)
@ti.kernel
def pair_wise():
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip self-interaction
                # Process pair (i, j)

### Parallel Loops
@ti.kernel
def parallel_process():
    # All iterations run in parallel on GPU
    for i in ti.ndrange(n):
        # Independent operations only
        
## Common Pitfalls and Solutions
### Variable Declaration in Conditionals
# WRONG - Variable defined inside conditional
if species == 0:
    strength = 150.0
else:
    strength = 50.0
force = strength * direction  # ERROR: strength not defined in all paths

# CORRECT - Declare variable first
strength = 50.0  # Default value
if species == 0:
    strength = 150.0
force = strength * direction  # OK: strength always defined

# ALSO CORRECT - Use conditional expression
strength = 150.0 if species == 0 else 50.0

### CRITICAL: No Returns Inside Conditionals/Loops
# WRONG - Return inside if statement
if species == 0:
    return ti.math.vec2(0.0, 0.0)  # ERROR: Return inside non-static if!
return normal_force

# CORRECT - Set result variable then return at end
result = normal_force  # Default
if species == 0:
    result = ti.math.vec2(0.0, 0.0)  # Set to zero for species 0
return result  # Single return at function end

# ALSO CORRECT - Use conditional expression
return ti.math.vec2(0.0, 0.0) if species == 0 else normal_force

### Division by Zero
# WRONG
direction = diff / diff.norm()

# CORRECT
dist = diff.norm()
if dist > 0.001:
    direction = diff / dist
else:
    direction = ti.math.vec2(0.0, 0.0)

### Boundary Conditions
# Wrap-around
pos.x = pos.x % tv.x
pos.y = pos.y % tv.y

# Clamp
pos.x = ti.math.clamp(pos.x, 0, tv.x)
pos.y = ti.math.clamp(pos.y, 0, tv.y)

# Bounce
if pos.x < 0 or pos.x > tv.x:
    vel.x *= -0.8  # Energy loss

### Type Casting
# Integer positions for pixel access
px = ti.cast(pos.x, ti.i32)
py = ti.cast(pos.y, ti.i32)
if 0 <= px < tv.x and 0 <= py < tv.y:
    tv.px.px.rgba[px, py] = color

### Random Numbers
# ti.random() returns [0, 1)
angle = ti.random() * 2 * 3.14159
random_vec = ti.math.vec2(ti.cos(angle), ti.sin(angle))

# Random in range
value = min_val + ti.random() * (max_val - min_val)
"""

ARTIFICIAL_LIFE_PATTERNS = """
# Artificial Life Patterns in Tölvera

## Standard Behavior Structure
Based on vera class template from examples:

@ti.data_oriented
class CustomBehavior:
    def __init__(self, tolvera, **kwargs):
        self.tv = tolvera
        self.kwargs = kwargs
        
        # Constants using CONSTS utility
        self.CONSTS = CONSTS({
            "PARAM1": (ti.f32, 100.0),
            "PARAM2": (ti.f32, 0.5),
        })
        
        # Species interaction matrix
        self.tv.s.behavior_s = {
            "state": {
                "attract": (ti.f32, -1.0, 1.0),
                "repel": (ti.f32, 0.0, 100.0),
                "radius": (ti.f32, 10.0, 200.0),
            },
            "shape": (self.tv.sn, self.tv.sn),
            "randomise": True,
        }
        
        # Per-particle state
        self.tv.s.behavior_p = {
            "state": {
                "energy": (ti.f32, 0.0, 100.0),
                "phase": (ti.f32, 0.0, 2*3.14159),
                "nearby": (ti.i32, 0, 100),
            },
            "shape": self.tv.pn,
            "randomise": False,
        }
    
    @ti.kernel
    def step(self, particles: ti.template(), weight: ti.f32):
        n = particles.shape[0]
        for i in range(n):
            # Skip inactive particles
            if particles[i].active == 0:
                continue
                
            p1 = particles[i]
            total_force = ti.math.vec2(0.0, 0.0)
            
            # Pair-wise interactions
            for j in range(n):
                # Skip self and inactive particles
                if i == j or particles[j].active == 0:
                    continue
                    
                p2 = particles[j]
                # Get species interaction rules
                rules = self.tv.s.behavior_s[p1.species, p2.species]
                
                diff = p2.pos - p1.pos
                dist = diff.norm()
                
                if dist > 0.01 and dist < rules.radius:
                    direction = diff / dist
                    # Apply attraction/repulsion
                    force = direction * (rules.attract - rules.repel / (dist + 1.0))
                    total_force += force
            
            # Update particle with weighted force
            particles[i].vel += total_force * weight * 0.01
            particles[i].pos += particles[i].vel
    
    @ti.func
    def compute_self_force(self, p: ti.template(), idx: ti.i32) -> ti.math.vec2:
        # Example: energy-based movement
        energy = self.tv.s.behavior_p[idx].energy
        angle = self.tv.s.behavior_p[idx].phase
        return ti.math.vec2(ti.cos(angle), ti.sin(angle)) * energy
    
    @ti.func
    def compute_pair_force(self, p1: ti.template(), p2: ti.template(), 
                          rules: ti.template()) -> ti.math.vec2:
        diff = p2.pos - p1.pos
        dist = diff.norm()
        if dist > 0.01 and dist < 100.0:
            direction = diff / dist
            force = direction * (rules.attraction - rules.repulsion / dist)
            return force
        return ti.math.vec2(0.0, 0.0)
    
    def __call__(self, particles, weight: ti.f32 = 1.0):
        self.step(particles.field, weight)

## Multi-Behavior Integration
Combining multiple behaviors in render loop (from examples):

@tv.render
def _():
    # Pixel effects
    tv.px.diffuse(0.99)  # Blur/spread pixels
    
    # Apply vera behaviors
    tv.v.flock(tv.p, weight=0.5)  # Flocking behavior
    tv.v.slime(tv.p)  # Slime mold behavior (returns pixels)
    
    # Alternative: chain behaviors
    # tv.v.plife(tv.p)  # Particle life
    # tv.v.swarm(tv.p, 11)  # Swarm with parameter
    
    # Draw particles with species colors
    tv.px.particles(tv.p, tv.s.species())
    
    return tv.px  # Return pixel buffer for display

## Common A-Life Behavior Components

### Energy Systems
# Per-particle energy that depletes and regenerates
@ti.func
def update_energy(idx: ti.i32, activity: ti.f32):
    energy = tv.s.llm_particle.field[idx].energy
    # Deplete based on activity
    energy -= activity * 0.1
    # Regenerate slowly
    energy += 0.05
    # Clamp to valid range
    energy = ti.math.clamp(energy, 0.0, 100.0)
    tv.s.llm_particle.field[idx].energy = energy

### Life Cycles
# Birth, growth, reproduction, death
@ti.func
def check_lifecycle(idx: ti.i32):
    age = tv.s.llm_particle.field[idx].age
    energy = tv.s.llm_particle.field[idx].energy
    
    # Death from old age or low energy
    if age > 1000 or energy < 1.0:
        tv.p.field[idx].active = 0.0
    
    # Reproduction when high energy
    if energy > 80.0 and age > 100:
        # Find inactive particle to spawn
        for j in range(tv.pn):
            if tv.p.field[j].active == 0.0:
                # Spawn new particle
                tv.p.field[j].active = 1.0
                tv.p.field[j].pos = tv.p.field[idx].pos + ti.random() * 10.0
                tv.p.field[j].species = tv.p.field[idx].species
                # Split energy
                tv.s.llm_particle.field[idx].energy *= 0.5
                tv.s.llm_particle.field[j].energy = energy * 0.5
                break

### Environmental Interaction
# Particles interact with pixel environment
@ti.kernel
def interact_with_environment():
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            # Sample environment
            px = ti.cast(tv.p.field[i].pos[0], ti.i32) % tv.x
            py = ti.cast(tv.p.field[i].pos[1], ti.i32) % tv.y
            env_value = tv.px.px.rgba[px, py].norm()
            
            # React to environment
            if env_value > 0.5:
                # Move away from bright areas
                angle = ti.random() * 2 * 3.14159
                tv.p.field[i].vel += ti.math.vec2(ti.cos(angle), ti.sin(angle)) * 10.0
            else:
                # Slow down in dark areas
                tv.p.field[i].vel *= 0.95

### Morphogenesis Patterns
# Growth and form generation
@ti.func
def growth_pattern(pos: ti.math.vec2, center: ti.math.vec2, t: ti.f32) -> ti.f32:
    dist = (pos - center).norm()
    # Radial growth with oscillation
    growth_radius = t * 50.0 + ti.sin(t * 0.1) * 10.0
    if dist < growth_radius:
        # Reaction-diffusion style patterning
        return ti.sin(dist * 0.3 - t) * 0.5 + 0.5
    return 0.0

## State Machine Patterns
# Discrete behavioral states
@ti.func
def update_state_machine(idx: ti.i32):
    state = ti.cast(tv.s.llm_particle.field[idx].state, ti.i32)
    energy = tv.s.llm_particle.field[idx].energy
    
    # State transitions
    if state == 0:  # Resting
        if energy > 50.0:
            state = 1  # Foraging
    elif state == 1:  # Foraging
        if energy < 20.0:
            state = 0  # Resting
        elif ti.random() < 0.01:
            state = 2  # Mating
    elif state == 2:  # Mating
        if energy < 30.0:
            state = 0  # Resting
    
    tv.s.llm_particle.field[idx].state = ti.cast(state, ti.f32)

## Collective Intelligence
# Stigmergy - indirect coordination through environment
@ti.kernel
def stigmergy_update():
    # Particles modify environment
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            px = ti.cast(tv.p.field[i].pos[0], ti.i32) % tv.x
            py = ti.cast(tv.p.field[i].pos[1], ti.i32) % tv.y
            
            # Deposit based on state
            state = tv.s.llm_particle.field[i].state
            if state > 0.5:  # Active state
                tv.px.px.rgba[px, py] += ti.math.vec4(0.1, 0.0, 0.0, 1.0)
            else:  # Passive state
                tv.px.px.rgba[px, py] += ti.math.vec4(0.0, 0.1, 0.0, 1.0)
"""

IML_PATTERNS = """
# Interactive Machine Learning Patterns

## Vector to Vector Mapping
Map particle states to species rules dynamically:

tv.iml.state_to_rules = {
    'size': (state_vector_size, rules_vector_size),
    'io': (list, list),
    'randomise': True,
    'config': {'interpolate': 'rbf'},
    'map_kw': {'k': 10}
}

def update_mapping():
    # Collect state vector
    state_vec = tv.s.particle_state.to_vec()
    
    # Map through IML
    tv.iml.i = {'state_to_rules': state_vec}
    rules_vec = tv.iml.o['state_to_rules']
    
    # Apply to species rules
    if rules_vec is not None:
        tv.s.species_rules.from_vec(rules_vec)

## Sensor-Actuator Coupling
Direct mapping from inputs to behaviors:

# Camera to particle movement
tv.iml.camera_to_particles = {
    'size': ((480, 640, 3), (tv.pn, 2)),
    'io': (lambda: camera.read(), tv.p.set_vel_all),
    'randomise': True
}

## Gesture to Parameter Control
Map hand positions to behavior parameters:

def hands_to_params(hands_data):
    if hands_data:
        # Map hand height to flocking strength
        flock_weight = hands_data[0].y / 480.0
        # Map hand spread to separation radius  
        hand_dist = abs(hands_data[0].x - hands_data[1].x)
        tv.v.flock.CONSTS.SEPARATION_RADIUS = hand_dist
    return flock_weight

## Feedback Loops
Create self-modifying systems:

# Particle positions influence their own rules
tv.iml.position_feedback = {
    'size': ((tv.pn, 2), tv.s.behavior_rules.size),
    'io': (tv.p.get_pos_all_2d, lambda x: tv.s.behavior_rules.from_vec(x)),
    'randomise': True,
    'config': {'interpolate': 'ripple'},
    'map_kw': {'ripple_depth': 5}
}
"""