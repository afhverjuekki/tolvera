"""
Working examples of expert functions for common behaviors. These serve as references for the LLM when generating new behaviors.
"""

EXAMPLE_EXPERTS = {
    "gravity": {
        "description": "Simple downward gravity",
        "code": """@ti.func
def expert_gravity(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Apply downward gravitational force'''
    gravity_strength = -500.0  # Negative Y for downward force
    return ti.math.vec2(0.0, gravity_strength * mass)  # Y+ is up, so gravity is negative"""
    },
    
    "center_attraction": {
        "description": "Particles attracted to screen center",
        "code": """@ti.func
def expert_center_attraction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Attract particles toward center of screen'''
    center = ti.math.vec2(tv.x / 2, tv.y / 2)
    to_center = center - pos
    dist = to_center.norm()
    
    force = ti.math.vec2(0.0, 0.0)
    if dist > 1.0:  # Avoid singularity at center
        direction = to_center / dist
        strength = 200.0 * (dist / 100.0)  # Stronger when farther
        force = direction * strength
    
    return force"""
    },
    
    "random_drift": {
        "description": "Particles drift randomly",
        "code": """@ti.func
def expert_random_drift(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Apply random drift force'''
    # Use particle index for consistent randomness per particle
    angle = (ti.random() + particle_idx * 0.1) * 2 * 3.14159
    strength = 150.0
    return ti.math.vec2(ti.cos(angle), ti.sin(angle)) * strength"""
    },
    
    "species_chase": {
        "description": "Species 0 chases species 1",
        "code": """@ti.func
def expert_chase(p1: ti.template(), p2: ti.template()) -> ti.math.vec2:
    '''Species 0 chases species 1'''
    force = ti.math.vec2(0.0, 0.0)
    
    if p1.species == 0 and p2.species == 1:
        to_target = p2.pos - p1.pos
        dist = to_target.norm()
        
        if dist > 5.0 and dist < 200.0:  # Chase within range
            direction = to_target / dist
            # Stronger force when closer (more exciting chase)
            strength = 400.0 * (1.0 - dist / 200.0)
            force = direction * strength
    
    return force"""
    },
    
    "mutual_repulsion": {
        "description": "Particles repel each other",
        "code": """@ti.func
def expert_repel(p1: ti.template(), p2: ti.template()) -> ti.math.vec2:
    '''Mutual repulsion between particles'''
    force = ti.math.vec2(0.0, 0.0)
    
    diff = p1.pos - p2.pos
    dist = diff.norm()
    
    if dist > 0.01 and dist < 50.0:  # Repel within radius
        direction = diff / dist
        # Inverse square law with softening
        strength = 2000.0 / (dist + 10.0)
        force = direction * strength
    
    return force"""
    },
    
    "flocking_alignment": {
        "description": "Boids-style alignment behavior",
        "code": """@ti.func
def expert_align(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Align velocity with nearby particles of same species'''
    perception_radius = 50.0
    alignment_strength = 0.1
    
    avg_velocity = ti.math.vec2(0.0, 0.0)
    neighbor_count = 0
    
    for j in range(tv.pn):
        if j != particle_idx:
            other = tv.p.field[j]
            if other.species == species and other.active > 0:
                diff = other.pos - pos
                dist = diff.norm()
                
                if dist < perception_radius:
                    avg_velocity += other.vel
                    neighbor_count += 1
    
    force = ti.math.vec2(0.0, 0.0)
    if neighbor_count > 0:
        avg_velocity /= neighbor_count
        desired_velocity = avg_velocity.normalized() * vel.norm()
        force = (desired_velocity - vel) * alignment_strength
    
    return force"""
    },
    
    "day_night_activity": {
        "description": "Activity varies with day/night cycle",
        "code": """@ti.func
def expert_day_night(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Particles more active during day, rest at night'''
    # Access global day phase
    day_phase = tv.s.llm_global.field[0].day_phase
    
    # Calculate activity level (0 = midnight, 0.5 = noon)
    activity = 0.1  # Base activity at night
    if day_phase > 0.25 and day_phase < 0.75:  # Daytime
        # Smooth sine curve for activity
        activity = 0.1 + 0.9 * ti.sin((day_phase - 0.25) * 2 * 3.14159)
    
    # Random movement scaled by activity
    angle = ti.random() * 2 * 3.14159
    base_force = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * 200.0
    
    return base_force * activity"""
    },
    
    "energy_based_movement": {
        "description": "Movement based on energy levels",
        "code": """@ti.func
def expert_energy_movement(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Move based on current energy level'''
    # Access particle energy state
    energy = tv.s.llm_particle.field[particle_idx].energy
    
    force = ti.math.vec2(0.0, 0.0)
    
    if energy > 50.0:  # High energy - active movement
        angle = ti.random() * 2 * 3.14159
        force = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * energy * 2.0
    elif energy > 20.0:  # Medium energy - slower movement
        force = vel * 0.5  # Maintain current direction
    else:  # Low energy - rest
        force = vel * -2.0  # Brake to stop
    
    # Slowly regenerate energy
    if energy < 100.0:
        tv.s.llm_particle.field[particle_idx].energy = energy + 0.1
    
    return force"""
    },
    
    "orbital_motion": {
        "description": "Particles orbit around center",
        "code": """@ti.func
def expert_orbit(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Create orbital motion around screen center'''
    center = ti.math.vec2(tv.x / 2, tv.y / 2)
    to_center = center - pos
    dist = to_center.norm()
    
    force = ti.math.vec2(0.0, 0.0)
    
    if dist > 10.0:  # Avoid singularity
        # Perpendicular vector for tangential force
        tangent = ti.math.vec2(-to_center.y, to_center.x) / dist
        
        # Orbital velocity based on distance (Kepler's law approximation)
        orbital_speed = 300.0 / ti.sqrt(dist)
        tangential_force = tangent * orbital_speed
        
        # Small centripetal correction to maintain orbit
        centripetal = (to_center / dist) * 50.0
        
        force = tangential_force + centripetal
    
    return force"""
    },
    
    "species_flocking": {
        "description": "Species-specific flocking with all three rules",
        "code": """@ti.func
def expert_flock(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Complete flocking behavior: alignment, cohesion, separation'''
    perception_radius = 80.0
    separation_radius = 30.0
    
    # Initialize result variable (CRITICAL: declare before any conditionals)
    result_force = ti.math.vec2(0.0, 0.0)
    
    # Check if this species should flock (example: predators don't flock)
    should_flock = 1  # Default: yes
    if species == 0:  # Species 0 are predators
        should_flock = 0  # Don't flock
    
    if should_flock == 1:
        # Initialize components
        alignment = ti.math.vec2(0.0, 0.0)
        cohesion = ti.math.vec2(0.0, 0.0)
        separation = ti.math.vec2(0.0, 0.0)
        
        neighbor_count = 0
        
        for j in range(tv.pn):
            if j != particle_idx:
                other = tv.p.field[j]
                if other.species == species and other.active > 0:
                    diff = other.pos - pos
                    dist = diff.norm()
                    
                    if dist < perception_radius:
                        # Alignment - match velocities
                        alignment += other.vel
                        
                        # Cohesion - move to center of mass
                        cohesion += other.pos
                        
                        neighbor_count += 1
                        
                        # Separation - avoid crowding
                        if dist < separation_radius and dist > 0.01:
                            separation += (pos - other.pos) / dist
        
        if neighbor_count > 0:
            # Alignment force
            alignment /= neighbor_count
            alignment_force = (alignment - vel) * 0.1
            
            # Cohesion force
            cohesion /= neighbor_count
            cohesion_force = (cohesion - pos) * 0.01
            
            # Separation force
            separation_force = separation * 5.0
            
            # Combine all forces
            result_force = alignment_force + cohesion_force + separation_force
    
    return result_force  # Single return at end"""
    },
    
    "home_return": {
        "description": "Particles return to home position when tired",
        "code": """@ti.func
def expert_home_return(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Return to home position based on energy level'''
    # Access particle states
    home_pos = tv.s.llm_particle.field[particle_idx].home_pos
    energy = tv.s.llm_particle.field[particle_idx].energy
    
    force = ti.math.vec2(0.0, 0.0)
    
    # When energy is low, return home
    if energy < 30.0:
        to_home = home_pos - pos
        dist = to_home.norm()
        
        if dist > 5.0:  # Not already home
            direction = to_home / dist
            # Urgency increases as energy decreases
            urgency = (30.0 - energy) / 30.0
            force = direction * urgency * 300.0
        else:
            # At home, rest and regenerate
            force = vel * -5.0  # Strong damping
            tv.s.llm_particle.field[particle_idx].energy = energy + 0.5
    else:
        # High energy, explore
        angle = ti.random() * 2 * 3.14159
        force = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * 150.0
        # Slowly deplete energy
        tv.s.llm_particle.field[particle_idx].energy = energy - 0.2
    
    return force"""
    },
    
    "predator_prey_chase": {
        "description": "Predators hunt prey that actively flee",
        "code": """@ti.func
def chase(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Red predators hunt green prey that try to escape'''
    force = ti.math.vec2(0.0, 0.0)
    
    if species == 0:  # Predator
        # Find closest prey (species 1)
        closest_prey = -1
        min_dist = 100000.0
        for j in range(tv.pn):
            if tv.p.field[j].species == 1 and tv.p.field[j].active > 0.0:
                dist = (pos - tv.p.field[j].pos).norm()
                if dist < min_dist:
                    min_dist = dist
                    closest_prey = j

        if closest_prey >= 0 and min_dist < 200.0:
            # Chase prey
            direction_to_prey = tv.p.field[closest_prey].pos - pos
            force = direction_to_prey.normalized() * 500.0

    elif species == 1:  # Prey
        # Find closest predator (species 0) and flee from it
        closest_predator = -1
        min_dist = 100000.0
        for j in range(tv.pn):
            if tv.p.field[j].species == 0 and tv.p.field[j].active > 0.0:
                dist = (pos - tv.p.field[j].pos).norm()
                if dist < min_dist:
                    min_dist = dist
                    closest_predator = j

        if closest_predator >= 0 and min_dist < 250.0:  # Larger detection radius
            # Flee from predator
            direction_from_predator = pos - tv.p.field[closest_predator].pos
            # Add some randomness to make escape less predictable
            random_offset = ti.math.vec2(ti.random() - 0.5, ti.random() - 0.5) * 0.3
            escape_direction = direction_from_predator.normalized() + random_offset
            force = escape_direction.normalized() * 400.0
        else:
            # When no predators nearby, add small random movement
            random_force = ti.math.vec2(ti.random() - 0.5, ti.random() - 0.5) * 50.0
            force = random_force

    return force"""
    }
}