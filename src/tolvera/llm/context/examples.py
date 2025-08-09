"""
Working examples of expert functions for common behaviors. These serve as references for the LLM when generating new behaviors.
"""

EXAMPLE_EXPERTS = {
    "particle_idx_usage": {
        "description": "CRITICAL: Correct usage of particle_idx parameter",
        "code": """@ti.func
def expert_neighbor_interaction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Example showing CORRECT usage of particle_idx parameter'''
    # IMPORTANT: The 5th parameter 'particle_idx' is the index of the current particle
    # Always use this when you need to skip self in loops or access current particle's state
    
    force = ti.math.vec2(0.0, 0.0)
    neighbor_radius = 100.0
    
    # Loop through all particles to find neighbors
    for j in range(tv.pn):
        # CORRECT: Use particle_idx to skip self
        if j != particle_idx:  # NOT 'if j != i:' - 'i' is undefined!
            other = tv.p.field[j]
            if other.active > 0:
                diff = other.pos - pos
                dist = diff.norm()
                
                if dist < neighbor_radius:
                    # Interact with neighbor
                    force += (pos - other.pos) / (dist + 1.0) * 50.0
    
    # Can also use particle_idx for particle-specific behavior
    # Example: particle-specific random phase
    phase = particle_idx * 0.1
    force += ti.math.vec2(ti.sin(phase), ti.cos(phase)) * 10.0
    
    return force"""
    },
    
    "gravity": {
        "description": "Simple downward gravity",
        "code": """@ti.func
def expert_gravity(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Apply downward gravitational force'''
    gravity_strength = 500.0  # Magnitude of gravity
    return ti.math.vec2(0.0, -gravity_strength * mass)  # Negative Y for downward force"""
    },
    
    "center_attraction": {
        "description": "Particles attracted to screen center",
        "code": """@ti.func
def expert_center_attraction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Attract particles toward center of screen'''
    center = ti.math.vec2(tv.x / 2.0, tv.y / 2.0)
    to_center = center - pos
    dist = to_center.norm()
    
    force = ti.math.vec2(0.0, 0.0)
    if dist > 1.0:  # Avoid singularity at center
        direction = to_center / dist
        strength = 200.0 * ti.min(dist / 100.0, 2.0)  # Capped strength
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
        "description": "Species 0 chases species 1 (interaction expert)",
        "code": """@ti.func
def expert_chase(p1: ti.template(), p2: ti.template()) -> ti.math.vec2:
    '''Species 0 chases species 1 - for interaction kernels'''
    force = ti.math.vec2(0.0, 0.0)
    
    if p1.species == 0 and p2.species == 1 and p2.active > 0.0:
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
    perception_radius = 80.0
    alignment_strength = 100.0
    
    avg_velocity = ti.math.vec2(0.0, 0.0)
    neighbor_count = 0
    
    for j in range(tv.pn):
        if j != particle_idx:  # IMPORTANT: Use particle_idx parameter, not 'i'!
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
        # Calculate desired velocity change
        desired_velocity = avg_velocity - vel
        force = desired_velocity * alignment_strength
    
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
            if j != particle_idx:  # CRITICAL: Use particle_idx param, NOT 'i'!
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
        min_dist = 200.0  # Hunt radius
        
        for j in range(tv.pn):
            if tv.p.field[j].species == 1 and tv.p.field[j].active > 0.0:
                diff = tv.p.field[j].pos - pos
                dist = diff.norm()
                if dist < min_dist:
                    min_dist = dist
                    closest_prey = j

        if closest_prey >= 0:
            # Chase prey
            prey_pos = tv.p.field[closest_prey].pos
            direction_to_prey = prey_pos - pos
            dist_to_prey = direction_to_prey.norm()
            if dist_to_prey > 0.001:
                # Stronger force when closer
                strength = 500.0 * (1.0 - min_dist / 200.0)
                force = (direction_to_prey / dist_to_prey) * strength

    elif species == 1:  # Prey
        # Find closest predator (species 0) and flee
        closest_predator = -1
        min_dist = 250.0  # Larger awareness radius for survival
        
        for j in range(tv.pn):
            if tv.p.field[j].species == 0 and tv.p.field[j].active > 0.0:
                diff = pos - tv.p.field[j].pos
                dist = diff.norm()
                if dist < min_dist:
                    min_dist = dist
                    closest_predator = j

        if closest_predator >= 0:
            # Flee from predator
            predator_pos = tv.p.field[closest_predator].pos
            direction_from_predator = pos - predator_pos
            dist_from_predator = direction_from_predator.norm()
            
            if dist_from_predator > 0.001:
                # Add some randomness for unpredictable escape
                angle_offset = (ti.random() - 0.5) * 0.5
                cos_a = ti.cos(angle_offset)
                sin_a = ti.sin(angle_offset)
                
                # Rotate escape direction slightly
                norm_dir = direction_from_predator / dist_from_predator
                escape_dir = ti.math.vec2(
                    norm_dir.x * cos_a - norm_dir.y * sin_a,
                    norm_dir.x * sin_a + norm_dir.y * cos_a
                )
                
                # Stronger force when predator is closer
                strength = 400.0 * (1.0 - min_dist / 250.0)
                force = escape_dir * strength
        else:
            # Idle wandering when safe
            angle = ti.random() * 2.0 * 3.14159
            force = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * 50.0

    return force"""
    },
    
    "pixel_trail_deposition": {
        "description": "Deposit trails in pixel buffer (like slime mold)",
        "code": """@ti.func
def expert_trail_deposit(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Deposit pheromone trail and follow existing trails'''
    
    # Sample ahead in three directions
    look_ahead = 20.0
    sensor_angle = 0.5  # radians
    
    # Current direction from velocity
    vel_norm = vel.norm()
    current_dir = vel / vel_norm if vel_norm > 0.1 else ti.math.vec2(1.0, 0.0)
    
    # Calculate sensor positions
    front_pos = pos + current_dir * look_ahead
    
    # Left sensor
    cos_left = ti.cos(-sensor_angle)
    sin_left = ti.sin(-sensor_angle)
    left_dir = ti.math.vec2(
        current_dir.x * cos_left - current_dir.y * sin_left,
        current_dir.x * sin_left + current_dir.y * cos_left
    )
    left_pos = pos + left_dir * look_ahead
    
    # Right sensor
    cos_right = ti.cos(sensor_angle)
    sin_right = ti.sin(sensor_angle)
    right_dir = ti.math.vec2(
        current_dir.x * cos_right - current_dir.y * sin_right,
        current_dir.x * sin_right + current_dir.y * cos_right
    )
    right_pos = pos + right_dir * look_ahead
    
    # Sample pixel values (simplified - actual implementation would sample pixels)
    # In real usage, you'd sample tv.px.px.rgba at these positions
    front_val = 0.5 + ti.random() * 0.1
    left_val = 0.5 + ti.random() * 0.1
    right_val = 0.5 + ti.random() * 0.1
    
    # Turn toward highest concentration
    force = ti.math.vec2(0.0, 0.0)
    turn_speed = 200.0
    move_speed = 100.0
    
    if left_val > front_val and left_val > right_val:
        force = left_dir * turn_speed
    elif right_val > front_val and right_val > left_val:
        force = right_dir * turn_speed
    else:
        force = current_dir * move_speed
    
    # Deposit trail at current position (would write to pixels in actual use)
    # px = ti.cast(pos.x, ti.i32) % tv.x
    # py = ti.cast(pos.y, ti.i32) % tv.y
    # tv.px.px.rgba[px, py] += trail_color * deposit_amount
    
    return force"""
    },
    
    "vera_style_interaction": {
        "description": "Vera-style species matrix interaction pattern",
        "code": """@ti.func
def expert_species_rules(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Apply species-specific interaction rules from matrix'''
    total_force = ti.math.vec2(0.0, 0.0)
    
    for j in range(tv.pn):
        if j != particle_idx and tv.p.field[j].active > 0:
            other = tv.p.field[j]
            diff = other.pos - pos
            dist = diff.norm()
            
            # Get interaction rules for this species pair
            # Assuming tv.s.interaction_rules exists with attract/repel parameters
            # rules = tv.s.interaction_rules[species, other.species]
            
            # For this example, use simple species-based rules
            if dist > 0.01 and dist < 100.0:
                direction = diff / dist
                
                if species == other.species:
                    # Same species: mild cohesion
                    attraction = 50.0 * (1.0 - dist / 100.0)
                    total_force += direction * attraction
                elif species == 0 and other.species == 1:
                    # Species 0 attracted to species 1
                    total_force += direction * 100.0
                elif species == 1 and other.species == 0:
                    # Species 1 repelled by species 0
                    total_force -= direction * 150.0 / (dist + 10.0)
    
    return total_force"""
    },
    
    "food_consumption": {
        "description": "Consume food particles on contact",
        "code": """@ti.func
def consume_food(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Seek and consume food particles (species 2)'''
    force = ti.math.vec2(0.0, 0.0)
    
    # Only consumers (species 0 or 1) seek food
    if species == 0 or species == 1:
        detection_radius = 100.0
        consumption_radius = 5.0
        nearest_food = -1
        min_dist = detection_radius
        
        # Find nearest food particle
        for j in range(tv.pn):
            if tv.p.field[j].species == 2 and tv.p.field[j].active > 0:  # Food is species 2
                diff = tv.p.field[j].pos - pos
                dist = diff.norm()
                if dist < min_dist:
                    min_dist = dist
                    nearest_food = j
        
        # Apply force toward food or consume it
        if nearest_food >= 0:
            if min_dist < consumption_radius:
                # Consume the food
                tv.p.field[nearest_food].active = 0.0
                # Could add energy gain here: tv.s.llm_particle.field[particle_idx].energy += 20.0
            else:
                # Move toward food
                food_pos = tv.p.field[nearest_food].pos
                direction = (food_pos - pos) / min_dist
                force = direction * 200.0
    
    return force"""
    },
    
    "toroidal_wrapping": {
        "description": "Calculate wrapped distance for toroidal topology",
        "code": """@ti.func
def wrap_distance(p1: ti.math.vec2, p2: ti.math.vec2) -> ti.math.vec2:
    '''Calculate shortest vector from p1 to p2 in toroidal world'''
    diff = p2 - p1
    
    # Check if wrapping gives shorter distance on X axis
    if ti.abs(diff.x) > tv.x * 0.5:
        if diff.x > 0:
            diff.x -= tv.x
        else:
            diff.x += tv.x
    
    # Check if wrapping gives shorter distance on Y axis
    if ti.abs(diff.y) > tv.y * 0.5:
        if diff.y > 0:
            diff.y -= tv.y
        else:
            diff.y += tv.y
    
    return diff

@ti.func
def find_nearest_wrapped(my_pos: ti.math.vec2, target_species: ti.i32, radius: ti.f32) -> ti.i32:
    '''Find nearest particle considering toroidal wrapping'''
    nearest = -1
    min_dist = radius
    
    for j in range(tv.pn):
        if tv.p.field[j].species == target_species and tv.p.field[j].active > 0:
            # Use wrapped distance
            diff = wrap_distance(my_pos, tv.p.field[j].pos)
            dist = diff.norm()
            
            if dist < min_dist:
                min_dist = dist
                nearest = j
    
    return nearest"""
    },
    
    "respawn_food": {
        "description": "Respawn consumed food particles",
        "code": """@ti.kernel
def respawn_food():
    '''Periodically respawn food particles that have been consumed'''
    respawn_rate = 0.01  # Probability of respawn per frame
    
    for i in range(tv.pn):
        if tv.p.field[i].species == 2 and tv.p.field[i].active == 0:  # Inactive food
            if ti.random() < respawn_rate:
                # Respawn at random location
                tv.p.field[i].pos = ti.math.vec2(
                    ti.random() * tv.x,
                    ti.random() * tv.y
                )
                tv.p.field[i].vel = ti.math.vec2(0.0, 0.0)  # Food doesn't move
                tv.p.field[i].active = 1.0
                # Reset any consumed state
                if hasattr(tv.s, 'llm_particle'):
                    tv.s.llm_particle.field[i].consumed = 0"""
    }
}