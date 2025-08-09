"""
Initialization patterns for particle systems.
These patterns provide context-aware initialization strategies.
"""

INITIALIZATION_PATTERNS = """
# Particle Initialization Patterns

## Random Initialization (Default)
```python
@ti.kernel
def init_particles():
    for i in range(tv.pn):
        tv.p.field[i].active = 1.0
        tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
        tv.p.field[i].vel = ti.Vector([
            (ti.random() - 0.5) * 100.0,
            (ti.random() - 0.5) * 100.0
        ])
        tv.p.field[i].mass = 0.5 + ti.random() * 0.5
        tv.p.field[i].size = 3.0 + ti.random() * 4.0  # Visible size 3-7
        tv.p.field[i].species = 0  # Single species
```

## Grid Initialization (Cellular Automata, Lattice Systems)
```python
@ti.kernel
def init_grid_particles(grid_size: ti.i32):
    cell_width = tv.x / ti.cast(grid_size, ti.f32)
    cell_height = tv.y / ti.cast(grid_size, ti.f32)
    
    for i in range(tv.pn):
        if i < grid_size * grid_size:
            grid_x = i % grid_size
            grid_y = i // grid_size
            
            # Center particles in grid cells
            tv.p.field[i].pos = ti.Vector([
                (grid_x + 0.5) * cell_width,
                (grid_y + 0.5) * cell_height
            ])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])  # Static initially
            tv.p.field[i].active = 1.0
            tv.p.field[i].size = cell_width * 0.8  # Fill most of cell
            
            # Store grid position if needed
            if 'llm_particle' in tv.s:
                tv.s.llm_particle.field[i].grid_x = grid_x
                tv.s.llm_particle.field[i].grid_y = grid_y
                tv.s.llm_particle.field[i].is_alive = ti.random() > 0.5
```

## Clustered Species Initialization (Ecosystems, Multi-Species)
```python
@ti.kernel
def init_species_clustered():
    particles_per_species = tv.pn // tv.sn
    
    for i in range(tv.pn):
        species_id = min(i // particles_per_species, tv.sn - 1)
        tv.p.field[i].species = species_id
        tv.p.field[i].active = 1.0
        
        # Cluster each species in different regions
        if species_id == 0:  # Predators - spread out more
            tv.p.field[i].pos = ti.Vector([
                ti.random() * tv.x,
                ti.random() * tv.y
            ])
            tv.p.field[i].size = 8.0  # Larger
        elif species_id == 1:  # Prey - clustered in schools
            # Create schools/clusters
            cluster_center = ti.Vector([
                0.2 + (ti.random() * 0.6) * tv.x,
                0.2 + (ti.random() * 0.6) * tv.y
            ])
            offset = ti.Vector([
                (ti.random() - 0.5) * 100.0,
                (ti.random() - 0.5) * 100.0
            ])
            tv.p.field[i].pos = cluster_center + offset
            tv.p.field[i].size = 5.0  # Smaller
        else:  # Other species
            # Random within bounds
            tv.p.field[i].pos = ti.Vector([
                0.1 * tv.x + ti.random() * 0.8 * tv.x,
                0.1 * tv.y + ti.random() * 0.8 * tv.y
            ])
            tv.p.field[i].size = 6.0
        
        # Species-specific velocities
        base_speed = 50.0 + species_id * 20.0
        tv.p.field[i].vel = ti.Vector([
            (ti.random() - 0.5) * base_speed,
            (ti.random() - 0.5) * base_speed
        ])
        tv.p.field[i].mass = 0.5 + species_id * 0.2
```

## Ring/Circle Initialization (Orbital Systems, Vortex)
```python
@ti.kernel
def init_ring_formation():
    center = ti.Vector([tv.x * 0.5, tv.y * 0.5])
    radius = min(tv.x, tv.y) * 0.3
    
    for i in range(tv.pn):
        angle = (ti.cast(i, ti.f32) / ti.cast(tv.pn, ti.f32)) * 2 * 3.14159
        
        # Position on ring
        tv.p.field[i].pos = center + ti.Vector([
            ti.cos(angle) * radius,
            ti.sin(angle) * radius
        ])
        
        # Tangential velocity for orbital motion
        tangent = ti.Vector([-ti.sin(angle), ti.cos(angle)])
        tv.p.field[i].vel = tangent * 100.0
        
        tv.p.field[i].active = 1.0
        tv.p.field[i].mass = 1.0
        tv.p.field[i].size = 5.0
```

## Wave Initialization (Wave Patterns, Oscillations)
```python
@ti.kernel
def init_wave_particles():
    particles_per_row = ti.cast(ti.sqrt(ti.cast(tv.pn, ti.f32)), ti.i32)
    
    for i in range(tv.pn):
        row = i // particles_per_row
        col = i % particles_per_row
        
        x_pos = (ti.cast(col, ti.f32) / ti.cast(particles_per_row, ti.f32)) * tv.x
        y_base = tv.y * 0.5
        
        # Create initial wave shape
        phase = (ti.cast(col, ti.f32) / ti.cast(particles_per_row, ti.f32)) * 2 * 3.14159
        y_offset = ti.sin(phase) * 50.0
        
        tv.p.field[i].pos = ti.Vector([x_pos, y_base + y_offset])
        tv.p.field[i].vel = ti.Vector([0.0, 0.0])
        tv.p.field[i].active = 1.0
        tv.p.field[i].size = 5.0
        
        # Store phase for wave propagation
        if 'llm_particle' in tv.s and hasattr(tv.s.llm_particle.field[i], 'phase'):
            tv.s.llm_particle.field[i].phase = phase
```

## State-Dependent Initialization
```python
@ti.kernel
def init_with_states():
    for i in range(tv.pn):
        # Basic particle setup
        tv.p.field[i].active = 1.0
        tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
        tv.p.field[i].vel = ti.Vector([
            (ti.random() - 0.5) * 50.0,
            (ti.random() - 0.5) * 50.0
        ])
        
        # Initialize custom states if they exist
        if 'llm_particle' in tv.s:
            particle_states = tv.s.llm_particle.field[i]
            
            # Energy states
            if hasattr(particle_states, 'energy'):
                # Species-dependent energy
                if tv.p.field[i].species == 0:  # Predator
                    particle_states.energy = 80.0
                else:  # Prey
                    particle_states.energy = 60.0
            
            # Home position states
            if hasattr(particle_states, 'home_pos'):
                particle_states.home_pos = tv.p.field[i].pos
            
            # Phase states for oscillators
            if hasattr(particle_states, 'phase'):
                particle_states.phase = ti.random()
            
            # Age for growth systems
            if hasattr(particle_states, 'age'):
                particle_states.age = 0.0
```

## IMPORTANT INITIALIZATION GUIDELINES:
1. Always set tv.p.field[i].active = 1.0 for active particles
2. Particle size should be 3.0-10.0 for visibility (not 1.0)
3. Species ID must be within range [0, tv.sn-1]
4. Initialize velocities appropriate to behavior (static for grids, random for swarms)
5. Consider screen boundaries when positioning
6. Initialize custom states after basic particle properties
7. Use ti.random() for randomness, not Python's random
8. Match initialization to behavior type (clustered for schools, grid for cellular automata)
"""

SPECIES_INITIALIZATION_PATTERNS = """
# Species-Aware Initialization

## Color Assignment by Species Role
```python
# Semantic color mapping based on species role
species_colors = {
    0: [1.0, 0.2, 0.2, 1.0],  # Red - Predators/Hunters
    1: [0.2, 0.8, 0.2, 1.0],  # Green - Prey/Plants/Food
    2: [0.2, 0.2, 1.0, 1.0],  # Blue - Neutral/Passive
    3: [1.0, 0.9, 0.2, 1.0],  # Yellow - Active/Energetic
    4: [0.8, 0.2, 0.8, 1.0],  # Purple - Special/Rare
    5: [0.2, 0.8, 0.8, 1.0],  # Cyan - Support/Helper
}

@ti.kernel
def init_species_colors():
    for i in range(tv.pn):
        species = tv.p.field[i].species
        # Color stored in species matrix
        for c in range(4):
            tv.s.species.field[species, c] = species_colors[species][c]
```

## Balanced Species Distribution
```python
@ti.kernel
def init_balanced_species():
    # Ensure roughly equal numbers of each species
    particles_per_species = tv.pn // tv.sn
    remainder = tv.pn % tv.sn
    
    for i in range(tv.pn):
        # Distribute particles among species
        cumulative = 0
        species_id = 0
        
        for s in range(tv.sn):
            cumulative += particles_per_species
            if s < remainder:
                cumulative += 1
            if i < cumulative:
                species_id = s
                break
        
        tv.p.field[i].species = species_id
        tv.p.field[i].active = 1.0
        
        # Species-specific properties
        tv.p.field[i].size = 4.0 + ti.cast(species_id, ti.f32) * 2.0
        tv.p.field[i].mass = 0.5 + ti.cast(species_id, ti.f32) * 0.3
```

## Ecosystem Initialization with Ratios
```python
@ti.kernel  
def init_ecosystem_ratios():
    # Typical ecosystem ratios (prey:predator = 5:1)
    total = tv.pn
    predator_count = total // 6
    prey_count = total - predator_count
    
    for i in range(tv.pn):
        if i < predator_count:
            tv.p.field[i].species = 0  # Predator
            tv.p.field[i].size = 8.0
            tv.p.field[i].mass = 1.5
            # Spread predators out
            tv.p.field[i].pos = ti.Vector([
                ti.random() * tv.x,
                ti.random() * tv.y
            ])
        else:
            tv.p.field[i].species = 1  # Prey
            tv.p.field[i].size = 5.0
            tv.p.field[i].mass = 0.8
            # Cluster prey in groups
            cluster_id = (i - predator_count) // 20
            cluster_x = 0.2 + (cluster_id % 3) * 0.3
            cluster_y = 0.2 + (cluster_id // 3) * 0.3
            tv.p.field[i].pos = ti.Vector([
                cluster_x * tv.x + (ti.random() - 0.5) * 50.0,
                cluster_y * tv.y + (ti.random() - 0.5) * 50.0
            ])
        
        tv.p.field[i].active = 1.0
```
"""