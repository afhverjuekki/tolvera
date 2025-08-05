EVOLUTION_PATTERNS = """
# Evolutionary and Genetic Patterns

## Genetic Encoding
Store genetic information in particle states:

tv.s.genetics = {
    "state": {
        "gene1": (ti.f32, 0.0, 1.0),  # Movement tendency
        "gene2": (ti.f32, 0.0, 1.0),  # Social behavior
        "gene3": (ti.f32, 0.0, 1.0),  # Environmental preference
        "fitness": (ti.f32, 0.0, 100.0),
    },
    "shape": tv.pn,
    "randomise": True
}

@ti.func
def express_genes(idx: ti.i32) -> ti.math.vec3:
    # Convert genes to behavioral parameters
    g = tv.s.genetics.field[idx]
    speed = g.gene1 * 5.0 + 1.0
    social_radius = g.gene2 * 100.0 + 10.0
    env_sensitivity = g.gene3
    return ti.math.vec3(speed, social_radius, env_sensitivity)

## Selection and Reproduction
@ti.kernel
def evolutionary_step():
    # Calculate fitness based on behavior
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            fitness = 0.0
            
            # Fitness from energy
            fitness += tv.s.llm_particle.field[i].energy * 0.5
            
            # Fitness from social connections
            neighbor_count = 0
            for j in range(tv.pn):
                if i != j and tv.p.field[j].active > 0:
                    dist = (tv.p.field[i].pos - tv.p.field[j].pos).norm()
                    if dist < 50.0:
                        neighbor_count += 1
            fitness += neighbor_count * 2.0
            
            tv.s.genetics.field[i].fitness = fitness
    
    # Selection and crossover
    for i in range(tv.pn):
        if tv.p.field[i].active == 0:  # Find dead particle
            # Tournament selection
            parent1 = ti.random_int(ti.i32) % tv.pn
            parent2 = ti.random_int(ti.i32) % tv.pn
            
            if tv.s.genetics.field[parent1].fitness > tv.s.genetics.field[parent2].fitness:
                winner = parent1
            else:
                winner = parent2
            
            # Inherit with mutation
            for g in range(3):  # For each gene
                gene_value = tv.s.genetics.field[winner][g]
                mutation = (ti.random() - 0.5) * 0.1
                tv.s.genetics.field[i][g] = ti.math.clamp(gene_value + mutation, 0.0, 1.0)
            
            # Revive particle
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = tv.p.field[winner].pos + ti.random() * 20.0

## Speciation Through Divergence
@ti.func
def genetic_distance(idx1: ti.i32, idx2: ti.i32) -> ti.f32:
    dist = 0.0
    for g in range(3):
        diff = tv.s.genetics.field[idx1][g] - tv.s.genetics.field[idx2][g]
        dist += diff * diff
    return ti.sqrt(dist)

@ti.func
def can_interbreed(idx1: ti.i32, idx2: ti.i32) -> ti.i32:
    # Species emerge from genetic distance
    return genetic_distance(idx1, idx2) < 0.3
"""

ECOSYSTEM_PATTERNS = """
# Ecosystem and Environmental Patterns

## Resource Grid System
Create environmental resources that particles consume:

# Resource field
resource_field = ti.field(dtype=ti.f32, shape=(tv.x//10, tv.y//10))

@ti.kernel
def update_resources():
    # Regenerate resources
    for i, j in resource_field:
        resource_field[i, j] += 0.01
        resource_field[i, j] = ti.min(resource_field[i, j], 1.0)
    
    # Particles consume resources
    for idx in range(tv.pn):
        if tv.p.field[idx].active > 0:
            # Grid position
            gx = ti.cast(tv.p.field[idx].pos[0] / 10, ti.i32)
            gy = ti.cast(tv.p.field[idx].pos[1] / 10, ti.i32)
            
            if 0 <= gx < resource_field.shape[0] and 0 <= gy < resource_field.shape[1]:
                # Consume based on species
                consumption = 0.1 if tv.p.field[idx].species == 0 else 0.05
                available = resource_field[gx, gy]
                consumed = ti.min(available, consumption)
                
                resource_field[gx, gy] -= consumed
                tv.s.llm_particle.field[idx].energy += consumed * 10.0

## Predator-Prey Dynamics
@ti.kernel
def predator_prey_interaction():
    for i in range(tv.pn):
        if tv.p.field[i].active == 0:
            continue
            
        p1 = tv.p.field[i]
        
        if p1.species == 0:  # Prey
            # Flee from predators
            flee_force = ti.math.vec2(0.0, 0.0)
            
            for j in range(tv.pn):
                if tv.p.field[j].species == 1:  # Predator
                    diff = p1.pos - tv.p.field[j].pos
                    dist = diff.norm()
                    if 0.01 < dist < 100.0:
                        flee_force += diff / (dist * dist) * 1000.0
            
            tv.p.field[i].vel += flee_force * 0.1
            
        elif p1.species == 1:  # Predator
            # Hunt prey
            closest_prey = -1
            min_dist = 1000000.0
            
            for j in range(tv.pn):
                if tv.p.field[j].species == 0 and tv.p.field[j].active > 0:
                    dist = (p1.pos - tv.p.field[j].pos).norm()
                    if dist < min_dist:
                        min_dist = dist
                        closest_prey = j
            
            if closest_prey >= 0 and min_dist < 200.0:
                # Chase
                direction = tv.p.field[closest_prey].pos - p1.pos
                tv.p.field[i].vel += direction.normalized() * 2.0
                
                # Catch and eat
                if min_dist < 5.0:
                    tv.p.field[closest_prey].active = 0.0
                    tv.s.llm_particle.field[i].energy += 50.0

## Symbiosis Patterns
@ti.func
def symbiotic_interaction(p1_idx: ti.i32, p2_idx: ti.i32) -> ti.math.vec2:
    p1 = tv.p.field[p1_idx]
    p2 = tv.p.field[p2_idx]
    force = ti.math.vec2(0.0, 0.0)
    
    # Mutualism: both benefit
    if p1.species == 0 and p2.species == 2:
        diff = p2.pos - p1.pos
        dist = diff.norm()
        if 10.0 < dist < 50.0:
            # Attract to optimal distance
            force = diff.normalized() * (30.0 - dist) * 0.5
            # Exchange resources
            tv.s.llm_particle.field[p1_idx].energy += 0.1
            tv.s.llm_particle.field[p2_idx].energy += 0.1
    
    # Parasitism: one benefits, one harmed
    elif p1.species == 3:  # Parasite
        diff = p2.pos - p1.pos
        dist = diff.norm()
        if dist < 20.0:
            # Steal energy
            steal_amount = ti.min(tv.s.llm_particle.field[p2_idx].energy * 0.01, 1.0)
            tv.s.llm_particle.field[p1_idx].energy += steal_amount
            tv.s.llm_particle.field[p2_idx].energy -= steal_amount
            # Stick close
            force = diff.normalized() * 10.0
    
    return force

## Food Web Dynamics
# Multi-level trophic interactions
trophic_level = {
    0: "producer",     # Creates energy from environment
    1: "herbivore",    # Eats producers
    2: "carnivore",    # Eats herbivores
    3: "decomposer"    # Recycles dead matter
}

@ti.kernel
def trophic_cascade():
    # Producers generate energy from light
    for i in range(tv.pn):
        if tv.p.field[i].species == 0:  # Producer
            # Sample light from environment
            px = ti.cast(tv.p.field[i].pos[0], ti.i32) % tv.x
            py = ti.cast(tv.p.field[i].pos[1], ti.i32) % tv.y
            light = tv.px.px.rgba[px, py].xyz.norm()
            tv.s.llm_particle.field[i].energy += light * 0.5
"""

MORPHOGENETIC_PATTERNS = """
# Morphogenesis and Pattern Formation

## Reaction-Diffusion on Particles
Turing patterns through chemical interactions:

tv.s.morphogen = {
    "state": {
        "u": (ti.f32, 0.0, 1.0),  # Activator
        "v": (ti.f32, 0.0, 1.0),  # Inhibitor
    },
    "shape": tv.pn,
    "randomise": True
}

@ti.kernel
def reaction_diffusion_step():
    # Parameters for spots/stripes
    F = 0.055  # Feed rate
    K = 0.062  # Kill rate
    Du = 0.16  # Diffusion rate U
    Dv = 0.08  # Diffusion rate V
    dt = 1.0
    
    for i in range(tv.pn):
        if tv.p.field[i].active == 0:
            continue
            
        u = tv.s.morphogen.field[i].u
        v = tv.s.morphogen.field[i].v
        
        # Reaction terms
        uvv = u * v * v
        du = -uvv + F * (1.0 - u)
        dv = uvv - (F + K) * v
        
        # Diffusion via neighbors
        laplacian_u = 0.0
        laplacian_v = 0.0
        neighbor_count = 0
        
        for j in range(tv.pn):
            if i != j and tv.p.field[j].active > 0:
                dist = (tv.p.field[i].pos - tv.p.field[j].pos).norm()
                if dist < 20.0:  # Diffusion radius
                    weight = 1.0 - dist / 20.0
                    laplacian_u += (tv.s.morphogen.field[j].u - u) * weight
                    laplacian_v += (tv.s.morphogen.field[j].v - v) * weight
                    neighbor_count += 1
        
        if neighbor_count > 0:
            laplacian_u /= neighbor_count
            laplacian_v /= neighbor_count
        
        # Update concentrations
        tv.s.morphogen.field[i].u = u + (Du * laplacian_u + du) * dt
        tv.s.morphogen.field[i].v = v + (Dv * laplacian_v + dv) * dt
        
        # Express pattern as color
        pattern_value = tv.s.morphogen.field[i].u
        tv.p.field[i].size = 1.0 + pattern_value * 3.0

## Cell Division and Growth
Particles that divide and differentiate:

@ti.kernel
def cell_division():
    for i in range(tv.pn):
        if tv.p.field[i].active == 0:
            continue
            
        # Check division conditions
        age = tv.s.llm_particle.field[i].age
        energy = tv.s.llm_particle.field[i].energy
        size = tv.p.field[i].size
        
        if energy > 80.0 and size > 2.0 and age > 50:
            # Find empty slot
            for j in range(tv.pn):
                if tv.p.field[j].active == 0:
                    # Create daughter cell
                    tv.p.field[j].active = 1.0
                    
                    # Position offset
                    angle = ti.random() * 2 * 3.14159
                    offset = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * size
                    tv.p.field[j].pos = tv.p.field[i].pos + offset
                    
                    # Inherit properties with variation
                    tv.p.field[j].species = tv.p.field[i].species
                    
                    # Differentiation based on morphogen
                    if tv.s.morphogen.field[i].u > 0.6:
                        tv.p.field[j].species = 1  # Differentiate
                    
                    # Split resources
                    tv.s.llm_particle.field[i].energy *= 0.5
                    tv.s.llm_particle.field[j].energy = energy * 0.5
                    tv.p.field[i].size *= 0.8
                    tv.p.field[j].size = size * 0.8
                    
                    # Reset age
                    tv.s.llm_particle.field[j].age = 0
                    break

## L-System Growth
Rule-based morphogenesis:

@ti.func
def growth_rules(cell_type: ti.i32, age: ti.f32) -> ti.i32:
    # Returns: 0=stay, 1=branch, 2=die, 3=flower
    if cell_type == 0:  # Stem
        if age < 10:
            return 0  # Keep growing
        elif age < 50:
            return 1  # Branch
        else:
            return 3  # Flower
    elif cell_type == 1:  # Branch
        if age < 30:
            return 0
        else:
            return 2  # Die
    return 0

@ti.kernel
def lsystem_growth():
    for i in range(tv.pn):
        if tv.p.field[i].active == 0:
            continue
            
        cell_type = tv.p.field[i].species
        age = tv.s.llm_particle.field[i].age
        
        action = growth_rules(cell_type, age)
        
        if action == 1:  # Branch
            # Grow in current direction
            growth_dir = tv.p.field[i].vel.normalized()
            
            # Create branches at angles
            for angle in [-0.5, 0.5]:  # Branch angles
                for j in range(tv.pn):
                    if tv.p.field[j].active == 0:
                        # New branch
                        tv.p.field[j].active = 1.0
                        tv.p.field[j].pos = tv.p.field[i].pos
                        
                        # Rotate growth direction
                        c = ti.cos(angle)
                        s = ti.sin(angle)
                        new_dir = ti.math.vec2(
                            growth_dir[0] * c - growth_dir[1] * s,
                            growth_dir[0] * s + growth_dir[1] * c
                        )
                        tv.p.field[j].vel = new_dir * 2.0
                        tv.p.field[j].species = 1  # Branch type
                        break
"""

SWARM_INTELLIGENCE = """
# Swarm Intelligence and Collective Behaviors

## Ant Colony Optimization
Pheromone-based pathfinding:

# Dual pheromone system
@ti.kernel
def ant_pheromone_system():
    for i in range(tv.pn):
        if tv.p.field[i].active == 0 or tv.p.field[i].species != 0:  # Only ants
            continue
            
        px = ti.cast(tv.p.field[i].pos[0], ti.i32) % tv.x
        py = ti.cast(tv.p.field[i].pos[1], ti.i32) % tv.y
        
        ant_state = tv.s.llm_particle.field[i].state
        
        if ant_state < 0.5:  # Searching for food
            # Deposit home pheromone (blue channel)
            tv.px.px.rgba[px, py][2] += 0.05
            
            # Follow food pheromone (red channel)
            max_food = 0.0
            best_dir = ti.math.vec2(0.0, 0.0)
            
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    sx = (px + dx) % tv.x
                    sy = (py + dy) % tv.y
                    food_strength = tv.px.px.rgba[sx, sy][0]
                    
                    if food_strength > max_food:
                        max_food = food_strength
                        best_dir = ti.math.vec2(dx, dy).normalized()
            
            if max_food > 0.1:
                tv.p.field[i].vel = best_dir * 3.0
            else:
                # Random search
                angle = tv.s.llm_particle.field[i].phase
                tv.p.field[i].vel = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * 2.0
                tv.s.llm_particle.field[i].phase += (ti.random() - 0.5) * 0.5
                
        else:  # Returning home
            # Deposit food pheromone
            tv.px.px.rgba[px, py][0] += 0.1
            
            # Follow home pheromone
            # (Similar gradient following for blue channel)

## Bee Dance Communication
Information transfer through movement:

@ti.kernel
def waggle_dance():
    for i in range(tv.pn):
        if tv.p.field[i].species != 1:  # Only bees
            continue
            
        bee_state = tv.s.llm_particle.field[i].state
        
        if bee_state > 0.8:  # Dancing
            # Encode food direction in dance
            food_angle = tv.s.llm_particle.field[i].target_angle
            dance_phase = tv.s.llm_particle.field[i].phase
            
            # Figure-8 pattern
            if dance_phase < 3.14159:
                # Waggle run - encodes direction
                tv.p.field[i].vel = ti.math.vec2(
                    ti.cos(food_angle) * 5.0,
                    ti.sin(food_angle) * 5.0
                ) + ti.math.vec2(ti.sin(dance_phase * 10), 0) * 2.0
            else:
                # Return loop
                tv.p.field[i].vel = ti.math.vec2(
                    ti.cos(dance_phase) * 3.0,
                    ti.sin(dance_phase) * 3.0
                )
            
            # Observers learn from dance
            for j in range(tv.pn):
                if i != j and tv.p.field[j].species == 1:
                    dist = (tv.p.field[i].pos - tv.p.field[j].pos).norm()
                    if dist < 30.0 and tv.s.llm_particle.field[j].state < 0.5:
                        # Learn food direction
                        tv.s.llm_particle.field[j].target_angle = food_angle
                        tv.s.llm_particle.field[j].state = 0.6  # Start foraging

## Quorum Sensing
Density-dependent collective decisions:

@ti.kernel
def quorum_sensing():
    # Count local density for each particle
    for i in range(tv.pn):
        if tv.p.field[i].active == 0:
            continue
            
        local_density = 0
        same_species_count = 0
        
        for j in range(tv.pn):
            if i != j and tv.p.field[j].active > 0:
                dist = (tv.p.field[i].pos - tv.p.field[j].pos).norm()
                if dist < 50.0:
                    local_density += 1
                    if tv.p.field[j].species == tv.p.field[i].species:
                        same_species_count += 1
        
        # Store density
        tv.s.llm_particle.field[i].local_density = ti.cast(local_density, ti.f32)
        
        # Collective behavior emerges at threshold
        if same_species_count > 10:  # Quorum reached
            if tv.p.field[i].species == 0:
                # Aggregation behavior
                tv.s.llm_particle.field[i].state = 1.0  # Aggregate mode
            elif tv.p.field[i].species == 1:
                # Dispersal behavior
                tv.s.llm_particle.field[i].state = 2.0  # Disperse mode

## Firefly Synchronization
Coupled oscillators achieving synchrony:

@ti.kernel
def firefly_sync():
    coupling_strength = 0.1
    flash_threshold = 0.9
    
    for i in range(tv.pn):
        if tv.p.field[i].species != 2:  # Only fireflies
            continue
            
        phase = tv.s.llm_particle.field[i].phase
        frequency = tv.s.llm_particle.field[i].frequency
        
        # Natural oscillation
        phase += frequency * 0.016
        
        # Coupling with neighbors
        phase_shift = 0.0
        neighbor_count = 0
        
        for j in range(tv.pn):
            if i != j and tv.p.field[j].species == 2:
                dist = (tv.p.field[i].pos - tv.p.field[j].pos).norm()
                if dist < 100.0:
                    neighbor_phase = tv.s.llm_particle.field[j].phase
                    
                    # Kuramoto coupling
                    phase_shift += ti.sin(neighbor_phase - phase)
                    neighbor_count += 1
        
        if neighbor_count > 0:
            phase += coupling_strength * phase_shift / neighbor_count
        
        # Wrap phase
        phase = phase % (2 * 3.14159)
        tv.s.llm_particle.field[i].phase = phase
        
        # Visual flash
        if ti.sin(phase) > flash_threshold:
            tv.p.field[i].size = 5.0
            # Emit light
            px = ti.cast(tv.p.field[i].pos[0], ti.i32) % tv.x
            py = ti.cast(tv.p.field[i].pos[1], ti.i32) % tv.y
            tv.px.circle(px, py, 10, ti.math.vec4(1.0, 1.0, 0.0, 1.0))
        else:
            tv.p.field[i].size = 2.0
"""