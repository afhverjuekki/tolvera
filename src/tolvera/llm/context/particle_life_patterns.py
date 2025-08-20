"""
Particle Life Patterns - Examples for Multi-Species Interaction Systems

This module provides context and patterns for particle-life simulations where
multiple species interact through attraction/repulsion matrices.
"""

PARTICLE_LIFE_STATE_PATTERN = """
# PARTICLE LIFE STATE DEFINITION (2D Interaction Matrix)
# For particle-life, we need a 2D matrix to store species-pair interactions

# Species interaction matrix (2D for species pairs)
if 'llm_species' not in tv.s:
    tv.s.set('llm_species', {
        'state': {
            'attraction_force': (ti.f32, -200.0, 200.0),  # Can be negative (repulsion) or positive (attraction)
            'interaction_radius': (ti.f32, 20.0, 200.0),
            'repulsion_radius': (ti.f32, 5.0, 30.0),
        },
        'shape': (tv.sn, tv.sn),  # 2D matrix for species interactions - CRITICAL!
        'osc': ('get', 'set'),
        'randomise': True
    })

# Initialize species interaction matrix with interesting patterns
@ti.kernel
def init_interaction_matrix():
    for s1 in range(tv.sn):
        for s2 in range(tv.sn):
            # More controlled attraction/repulsion forces
            tv.s.llm_species.field[s1, s2].attraction_force = (ti.random() - 0.5) * 100.0
            
            # Interaction radius varies by species pair
            tv.s.llm_species.field[s1, s2].interaction_radius = 60.0 + ti.random() * 80.0
            
            # Close-range repulsion to prevent overlap
            tv.s.llm_species.field[s1, s2].repulsion_radius = 15.0 + ti.random() * 10.0

init_interaction_matrix()
"""

PARTICLE_LIFE_EXPERT_PATTERN = """
# PARTICLE LIFE INTERACTION EXPERT
# Shows correct way to access 2D interaction matrix in Taichi

@ti.func
def particle_life_interaction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Calculate attraction/repulsion forces based on species interaction matrix.'''
    force = ti.math.vec2(0.0, 0.0)
    
    for j in range(tv.pn):
        if particle_idx != j and tv.p.field[j].active > 0:
            other_pos = tv.p.field[j].pos
            other_species = tv.p.field[j].species
            
            # Get interaction parameters from the 2D matrix - CORRECT ACCESS PATTERN!
            # Use [species1, species2] indexing for 2D fields
            attraction = tv.s.llm_species.field[species, other_species].attraction_force
            interaction_radius = tv.s.llm_species.field[species, other_species].interaction_radius
            repulsion_radius = tv.s.llm_species.field[species, other_species].repulsion_radius
            
            # Calculate distance and direction
            diff = other_pos - pos
            dist = diff.norm()
            
            # Declare force components BEFORE conditionals (Taichi requirement)
            direction = ti.math.vec2(0.0, 0.0)
            force_magnitude = 0.0
            
            if dist > 0.001 and dist < interaction_radius:
                direction = diff / dist
                
                # Apply forces based on distance
                if dist < repulsion_radius:
                    # Strong repulsion at close range
                    force_magnitude = -300.0 * (1.0 - dist / repulsion_radius)
                else:
                    # Attraction/repulsion based on matrix
                    normalized_dist = (dist - repulsion_radius) / (interaction_radius - repulsion_radius)
                    # Smooth falloff with distance
                    force_magnitude = attraction * (1.0 - normalized_dist)
                
                force += direction * force_magnitude
    
    # Clamp total force to prevent instability
    max_force = 200.0  # Or get from global state
    force_norm = force.norm()
    
    # Declare clamped_force BEFORE conditional
    clamped_force = force
    
    if force_norm > max_force and force_norm > 0.001:
        clamped_force = (force / force_norm) * max_force
    
    return clamped_force
"""

PARTICLE_LIFE_WRONG_PATTERNS = """
# ⚠️ COMMON MISTAKES IN PARTICLE LIFE IMPLEMENTATION

❌ WRONG - Nested field access (WILL CRASH):
```python
# Cannot do nested field member access:
attraction = tv.s.llm_species.field[species].attraction_matrix[other_species]  # CRASH!
```

❌ WRONG - 1D species field for matrix data:
```python
# This creates a 1D array, not a 2D matrix:
tv.s.set('llm_species', {
    'state': {'attraction_matrix': (ti.f32, -200.0, 200.0)},
    'shape': tv.sn,  # WRONG - this is 1D!
})
```

❌ WRONG - Return in conditional:
```python
if dist < repulsion_radius:
    return -direction * repulsion_force  # CRASH - return inside if!
```

✅ CORRECT - 2D field access:
```python
# Access 2D field with two indices:
value = tv.s.llm_species.field[species1, species2].attraction_force
```

✅ CORRECT - 2D shape specification:
```python
'shape': (tv.sn, tv.sn),  # Correct 2D shape for matrix
```

✅ CORRECT - Single return at end:
```python
force = ti.math.vec2(0.0, 0.0)
if condition:
    force = calculated_force
return force  # Single return at function end
```
"""

PARTICLE_LIFE_COMPLETE_EXAMPLE = """
# COMPLETE PARTICLE LIFE EXAMPLE
# This shows the full pattern for multi-species particle-life simulation

# 1. State Definition (2D matrix for species interactions)
if 'llm_species' not in tv.s:
    tv.s.set('llm_species', {
        'state': {
            'attraction_force': (ti.f32, -200.0, 200.0),
            'interaction_radius': (ti.f32, 20.0, 200.0),
            'repulsion_radius': (ti.f32, 5.0, 30.0),
        },
        'shape': (tv.sn, tv.sn),  # CRITICAL: 2D shape for species pairs
        'osc': ('get', 'set'),
        'randomise': True
    })

# Global parameters
if 'llm_global' not in tv.s:
    tv.s.set('llm_global', {
        'state': {
            'max_force': (ti.f32, 10.0, 500.0),
            'damping': (ti.f32, 0.9, 0.999),
        },
        'shape': 1,
        'osc': ('get', 'set'),
        'randomise': False
    })

# 2. Initialize interaction matrix
@ti.kernel
def init_interaction_matrix():
    for s1 in range(tv.sn):
        for s2 in range(tv.sn):
            # Different patterns for different species pairs
            if s1 == s2:
                # Same species - mild attraction (clustering)
                tv.s.llm_species.field[s1, s2].attraction_force = ti.random() * 50.0
            else:
                # Different species - varied interaction
                tv.s.llm_species.field[s1, s2].attraction_force = (ti.random() - 0.5) * 150.0
            
            tv.s.llm_species.field[s1, s2].interaction_radius = 80.0 + ti.random() * 60.0
            tv.s.llm_species.field[s1, s2].repulsion_radius = 15.0 + ti.random() * 10.0

init_interaction_matrix()
tv.s.llm_global.field[0].max_force = 200.0
tv.s.llm_global.field[0].damping = 0.98

# 3. Particle life interaction expert
@ti.func
def particle_life_interactions(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    
    for j in range(tv.pn):
        if particle_idx != j and tv.p.field[j].active > 0:
            other_species = tv.p.field[j].species
            diff = tv.p.field[j].pos - pos
            dist = diff.norm()
            
            if dist > 0.001:
                # Get interaction parameters from 2D matrix
                attraction = tv.s.llm_species.field[species, other_species].attraction_force
                interaction_radius = tv.s.llm_species.field[species, other_species].interaction_radius
                repulsion_radius = tv.s.llm_species.field[species, other_species].repulsion_radius
                
                if dist < interaction_radius:
                    direction = diff / dist
                    
                    # Calculate force based on distance
                    force_magnitude = 0.0
                    if dist < repulsion_radius:
                        # Close-range repulsion
                        force_magnitude = -300.0 * (1.0 - dist / repulsion_radius)
                    else:
                        # Attraction/repulsion with smooth falloff
                        normalized_dist = (dist - repulsion_radius) / (interaction_radius - repulsion_radius)
                        force_magnitude = attraction * (1.0 - normalized_dist)
                    
                    force += direction * force_magnitude
    
    # Apply force clamping
    max_force = tv.s.llm_global.field[0].max_force
    force_norm = force.norm()
    if force_norm > max_force:
        force = (force / force_norm) * max_force
    
    return force

# 4. Integration kernel
@ti.kernel
def apply_all_experts():
    dt = 0.016
    damping = tv.s.llm_global.field[0].damping
    
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            pos = tv.p.field[i].pos
            vel = tv.p.field[i].vel
            mass = tv.p.field[i].mass
            species = tv.p.field[i].species
            
            # Calculate total force
            total_force = particle_life_interactions(pos, vel, mass, species, i)
            
            # Apply damping
            total_force += -vel * (1.0 - damping)
            
            # Update velocity (force/mass * dt)
            acc = total_force / mass
            tv.p.field[i].vel += acc * dt
            
            # Clamp velocity
            max_vel = 500.0
            vel_mag = tv.p.field[i].vel.norm()
            if vel_mag > max_vel:
                tv.p.field[i].vel = (tv.p.field[i].vel / vel_mag) * max_vel
"""

PARTICLE_LIFE_CONTEXT = """
# Particle Life Pattern Context

Particle Life is a fundamental artificial life pattern where multiple species interact through
pairwise attraction/repulsion forces, creating emergent clustering and movement patterns.

## Key Characteristics:
1. **2D Interaction Matrix**: Species pairs have unique interaction parameters
2. **Distance-Based Forces**: Smooth falloff with distance, close-range repulsion
3. **Emergent Behavior**: Simple rules create complex patterns
4. **Force Clamping**: Prevents instability from excessive forces

## When to Use This Pattern:
- Description mentions "different species interact"
- "attraction and repulsion between species"
- "particle life" explicitly mentioned
- Multiple species with varied relationships
- "species X attracts/repels species Y"

## Critical Implementation Points:
1. **State Shape**: Must be (tv.sn, tv.sn) for 2D matrix
2. **Field Access**: Use field[s1, s2] not field[s1].matrix[s2]
3. **Force Calculation**: Loop through all particles, lookup interaction parameters
4. **Initialization**: Use kernel to set random or patterned interactions
5. **No Returns in Conditionals**: Declare result variable first

## Parameters to Include:
- attraction_force: Can be negative (repulsion) or positive (attraction)
- interaction_radius: Maximum distance for interaction
- repulsion_radius: Distance below which particles repel regardless
- max_force: Clamp total forces to prevent instability
- damping: Velocity damping for stability
"""

def get_particle_life_context():
    """Return all particle life patterns for prompt building."""
    return {
        'state_pattern': PARTICLE_LIFE_STATE_PATTERN,
        'expert_pattern': PARTICLE_LIFE_EXPERT_PATTERN,
        'wrong_patterns': PARTICLE_LIFE_WRONG_PATTERNS,
        'complete_example': PARTICLE_LIFE_COMPLETE_EXAMPLE,
        'context': PARTICLE_LIFE_CONTEXT
    }