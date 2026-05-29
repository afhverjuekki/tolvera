"""
Predator-Prey Ecosystem Simulation in Tölvera
Energy-based ecosystem with hunting, reproduction, and evolutionary dynamics
"""

import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    """Main function for Predator-Prey ecosystem simulation."""
    # === Configuration ===
    if 'species' not in kwargs:
        kwargs['species'] = 4  # Prey1, Prey2, Predator1, Predator2
    if 'particles' not in kwargs:
        kwargs['particles'] = 1500  # Total organisms
    if 'width' not in kwargs:
        kwargs['width'] = 1920
    if 'height' not in kwargs:
        kwargs['height'] = 1080
    
    # Create Tölvera instance
    tv = Tolvera(**kwargs)
    
    # === Food/Vegetation Field ===
    vegetation_field = ti.field(dtype=ti.f32, shape=(tv.x // 10, tv.y // 10))
    
    # === Particle Initialization ===
    @ti.kernel
    def init_particles_ecosystem():
        """Initialize organisms with prey and predators."""
        particles_per_species = tv.pn // tv.sn
        remaining = tv.pn % tv.sn
        
        particle_idx = 0
        
        for species_id in range(tv.sn):
            count = particles_per_species + (1 if species_id < remaining else 0)
            
            # Prey species start more dispersed, predators start in groups
            if species_id < 2:  # Prey species
                for i in range(count):
                    if particle_idx < tv.pn:
                        tv.p.field[particle_idx].active = 1.0
                        tv.p.field[particle_idx].pos = ti.Vector([
                            100.0 + ti.random() * (tv.x - 200.0),
                            100.0 + ti.random() * (tv.y - 200.0)
                        ])
                        
                        # Prey move in grazing patterns
                        angle = ti.random() * 2.0 * 3.14159
                        speed = 30.0 + ti.random() * 20.0
                        tv.p.field[particle_idx].vel = ti.Vector([
                            ti.cos(angle) * speed,
                            ti.sin(angle) * speed
                        ])
                        
                        tv.p.field[particle_idx].size = 3.0
                        tv.p.field[particle_idx].mass = 1.0
                        tv.p.field[particle_idx].speed = 1.0
                        tv.p.field[particle_idx].species = species_id
                        
                        particle_idx += 1
            else:  # Predator species
                # Start in hunting packs
                pack_center = ti.Vector([
                    200.0 + ti.random() * (tv.x - 400.0),
                    200.0 + ti.random() * (tv.y - 400.0)
                ])
                
                for i in range(count):
                    if particle_idx < tv.pn:
                        angle = ti.random() * 2.0 * 3.14159
                        radius = ti.random() * 50.0
                        
                        tv.p.field[particle_idx].active = 1.0
                        tv.p.field[particle_idx].pos = pack_center + ti.Vector([
                            ti.cos(angle) * radius,
                            ti.sin(angle) * radius
                        ])
                        
                        # Predators start with hunting movement
                        heading = ti.random() * 2.0 * 3.14159
                        speed = 40.0 + ti.random() * 30.0
                        tv.p.field[particle_idx].vel = ti.Vector([
                            ti.cos(heading) * speed,
                            ti.sin(heading) * speed
                        ])
                        
                        tv.p.field[particle_idx].size = 5.0
                        tv.p.field[particle_idx].mass = 1.5
                        tv.p.field[particle_idx].speed = 1.2
                        tv.p.field[particle_idx].species = species_id
                        
                        particle_idx += 1
    
    init_particles_ecosystem()
    
    # Set species colors - prey are green/blue, predators are red/orange
    colors = [
        [0.2, 0.9, 0.3, 1.0],  # Green prey (herbivore)
        [0.2, 0.7, 0.9, 1.0],  # Blue prey (herbivore)
        [0.9, 0.2, 0.2, 1.0],  # Red predator
        [1.0, 0.5, 0.2, 1.0],  # Orange predator
    ]
    
    for i in range(min(tv.sn, len(colors))):
        tv.s.species.field[i].rgba = colors[i]
    
    # === Initialize Vegetation ===
    @ti.kernel
    def init_vegetation():
        """Create patches of vegetation for prey to eat."""
        for x in range(tv.x // 10):
            for y in range(tv.y // 10):
                # Create patchy vegetation with Perlin-like noise
                noise_val = ti.sin(x * 0.1) * ti.cos(y * 0.1) + ti.random() * 0.5
                vegetation_field[x, y] = max(0.0, min(100.0, 50.0 + noise_val * 50.0))
    
    init_vegetation()
    
    # === State Initialization ===
    # Global ecosystem parameters
    if 'llm_global' not in tv.s:
        tv.s.set('llm_global', {
            'state': {
                'vegetation_growth_rate': (ti.f32, 0.01, 0.1),
                'energy_transfer_efficiency': (ti.f32, 0.1, 0.5),
                'reproduction_threshold': (ti.f32, 50.0, 150.0),
                'starvation_rate': (ti.f32, 0.1, 1.0),
                'vision_range': (ti.f32, 50.0, 200.0),
                'flee_distance': (ti.f32, 30.0, 100.0),
            },
            'shape': 1,
            'osc': ('get', 'set'),
            'randomise': False
        })
    
    # Species-specific parameters
    if 'llm_species' not in tv.s:
        tv.s.set('llm_species', {
            'state': {
                'is_predator': (ti.f32, 0.0, 1.0),  # 0 for prey, 1 for predator
                'max_speed': (ti.f32, 20.0, 100.0),
                'hunt_speed': (ti.f32, 50.0, 150.0),
                'energy_consumption': (ti.f32, 0.1, 2.0),
                'reproduction_cost': (ti.f32, 20.0, 80.0),
                'attack_range': (ti.f32, 10.0, 30.0),
            },
            'shape': tv.sn,
            'osc': ('get', 'set'),
            'randomise': False
        })
    
    # Per-particle state
    if 'llm_particles' not in tv.s:
        tv.s.set('llm_particles', {
            'state': {
                'energy': (ti.f32, 0.0, 200.0),
                'age': (ti.f32, 0.0, 1000.0),
                'hunting_target': (ti.i32, -1, 5000),  # Index of prey being hunted
                'fleeing_from': (ti.i32, -1, 5000),  # Index of predator to flee from
                'reproduction_cooldown': (ti.f32, 0.0, 100.0),
            },
            'shape': tv.pn,
            'osc': ('get',),
            'randomise': False
        })
    
    # Initialize global parameters
    tv.s.llm_global.field[0].vegetation_growth_rate = 0.05
    tv.s.llm_global.field[0].energy_transfer_efficiency = 0.3
    tv.s.llm_global.field[0].reproduction_threshold = 80.0
    tv.s.llm_global.field[0].starvation_rate = 0.5
    tv.s.llm_global.field[0].vision_range = 120.0
    tv.s.llm_global.field[0].flee_distance = 60.0
    
    # Initialize species parameters
    @ti.kernel
    def init_species_params():
        # Prey species 1 - fast but fragile
        tv.s.llm_species.field[0].is_predator = 0.0
        tv.s.llm_species.field[0].max_speed = 60.0
        tv.s.llm_species.field[0].hunt_speed = 80.0  # Flee speed
        tv.s.llm_species.field[0].energy_consumption = 0.8
        tv.s.llm_species.field[0].reproduction_cost = 30.0
        tv.s.llm_species.field[0].attack_range = 0.0
        
        # Prey species 2 - slower but hardier
        tv.s.llm_species.field[1].is_predator = 0.0
        tv.s.llm_species.field[1].max_speed = 40.0
        tv.s.llm_species.field[1].hunt_speed = 60.0
        tv.s.llm_species.field[1].energy_consumption = 0.5
        tv.s.llm_species.field[1].reproduction_cost = 40.0
        tv.s.llm_species.field[1].attack_range = 0.0
        
        # Predator species 1 - fast hunters
        tv.s.llm_species.field[2].is_predator = 1.0
        tv.s.llm_species.field[2].max_speed = 50.0
        tv.s.llm_species.field[2].hunt_speed = 100.0
        tv.s.llm_species.field[2].energy_consumption = 1.2
        tv.s.llm_species.field[2].reproduction_cost = 60.0
        tv.s.llm_species.field[2].attack_range = 20.0
        
        # Predator species 2 - pack hunters
        tv.s.llm_species.field[3].is_predator = 1.0
        tv.s.llm_species.field[3].max_speed = 45.0
        tv.s.llm_species.field[3].hunt_speed = 80.0
        tv.s.llm_species.field[3].energy_consumption = 1.0
        tv.s.llm_species.field[3].reproduction_cost = 50.0
        tv.s.llm_species.field[3].attack_range = 25.0
    
    init_species_params()
    
    # Initialize particle energy
    @ti.kernel
    def init_particle_energy():
        for i in range(tv.pn):
            tv.s.llm_particles.field[i].energy = 50.0 + ti.random() * 30.0
            tv.s.llm_particles.field[i].age = 0.0
            tv.s.llm_particles.field[i].hunting_target = -1
            tv.s.llm_particles.field[i].fleeing_from = -1
            tv.s.llm_particles.field[i].reproduction_cooldown = 0.0
    
    init_particle_energy()
    
    # === Helper Functions ===
    @ti.func
    def find_nearest_prey(predator_idx: ti.i32) -> ti.i32:
        """Find the nearest prey for a predator."""
        predator_pos = tv.p.field[predator_idx].pos
        vision = tv.s.llm_global.field[0].vision_range
        
        nearest_idx = -1
        nearest_dist = vision
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0 and i != predator_idx:
                species = tv.p.field[i].species
                if tv.s.llm_species.field[species].is_predator < 0.5:  # It's prey
                    dist = (tv.p.field[i].pos - predator_pos).norm()
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_idx = i
        
        return nearest_idx
    
    @ti.func
    def find_nearest_predator(prey_idx: ti.i32) -> ti.i32:
        """Find the nearest predator to flee from."""
        prey_pos = tv.p.field[prey_idx].pos
        flee_dist = tv.s.llm_global.field[0].flee_distance
        
        nearest_idx = -1
        nearest_dist = flee_dist
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0 and i != prey_idx:
                species = tv.p.field[i].species
                if tv.s.llm_species.field[species].is_predator > 0.5:  # It's a predator
                    dist = (tv.p.field[i].pos - prey_pos).norm()
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_idx = i
        
        return nearest_idx
    
    @ti.func
    def sample_vegetation(x: ti.f32, y: ti.f32) -> ti.f32:
        """Sample vegetation amount at a position."""
        xi = ti.cast(x / 10, ti.i32)
        yi = ti.cast(y / 10, ti.i32)
        
        result = 0.0
        if xi >= 0 and xi < tv.x // 10 and yi >= 0 and yi < tv.y // 10:
            result = vegetation_field[xi, yi]
        
        return result
    
    # === Behavior Force Experts ===
    @ti.func
    def hunting_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Predators hunt prey, prey flee from predators."""
        force = ti.math.vec2(0.0, 0.0)
        
        if tv.s.llm_species.field[species].is_predator > 0.5:
            # Predator hunting behavior
            target_idx = tv.s.llm_particles.field[particle_idx].hunting_target
            
            # Find new target if needed
            if target_idx < 0 or target_idx >= tv.pn or tv.p.field[target_idx].active < 0.5:
                target_idx = find_nearest_prey(particle_idx)
                tv.s.llm_particles.field[particle_idx].hunting_target = target_idx
            
            # Chase the target
            if target_idx >= 0 and tv.p.field[target_idx].active > 0:
                target_pos = tv.p.field[target_idx].pos
                direction = target_pos - pos
                dist = direction.norm()
                
                if dist > 0.001:
                    # Strong pursuit force
                    hunt_speed = tv.s.llm_species.field[species].hunt_speed
                    desired_vel = (direction / dist) * hunt_speed
                    force = (desired_vel - vel) * 5.0
        else:
            # Prey fleeing behavior
            predator_idx = tv.s.llm_particles.field[particle_idx].fleeing_from
            
            # Check for nearby predators
            if predator_idx < 0 or predator_idx >= tv.pn or tv.p.field[predator_idx].active < 0.5:
                predator_idx = find_nearest_predator(particle_idx)
                tv.s.llm_particles.field[particle_idx].fleeing_from = predator_idx
            
            # Flee from predator
            if predator_idx >= 0 and tv.p.field[predator_idx].active > 0:
                predator_pos = tv.p.field[predator_idx].pos
                escape_direction = pos - predator_pos
                dist = escape_direction.norm()
                
                if dist > 0.001 and dist < tv.s.llm_global.field[0].flee_distance:
                    # Panic flight
                    flee_speed = tv.s.llm_species.field[species].hunt_speed
                    desired_vel = (escape_direction / dist) * flee_speed
                    force = (desired_vel - vel) * 10.0
            else:
                # No predator nearby, clear fleeing state
                tv.s.llm_particles.field[particle_idx].fleeing_from = -1
        
        return force
    
    @ti.func
    def grazing_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Prey move towards vegetation."""
        force = ti.math.vec2(0.0, 0.0)
        
        if tv.s.llm_species.field[species].is_predator < 0.5:  # Only prey graze
            # Sample vegetation gradient
            sample_dist = 20.0
            
            veg_right = sample_vegetation(pos.x + sample_dist, pos.y)
            veg_left = sample_vegetation(pos.x - sample_dist, pos.y)
            veg_up = sample_vegetation(pos.x, pos.y + sample_dist)
            veg_down = sample_vegetation(pos.x, pos.y - sample_dist)
            
            # Move towards higher vegetation
            grad_x = (veg_right - veg_left) / (2.0 * sample_dist)
            grad_y = (veg_up - veg_down) / (2.0 * sample_dist)
            
            force.x = grad_x * 50.0
            force.y = grad_y * 50.0
        
        return force
    
    @ti.func
    def flocking_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Same species flock together for safety."""
        force = ti.math.vec2(0.0, 0.0)
        perception_radius = 80.0
        
        alignment = ti.math.vec2(0.0, 0.0)
        cohesion = ti.math.vec2(0.0, 0.0)
        separation = ti.math.vec2(0.0, 0.0)
        count = 0
        
        for i in range(tv.pn):
            if i != particle_idx and tv.p.field[i].active > 0:
                if tv.p.field[i].species == species:
                    other_pos = tv.p.field[i].pos
                    dist = (other_pos - pos).norm()
                    
                    if dist < perception_radius and dist > 0.001:
                        # Alignment
                        alignment += tv.p.field[i].vel
                        # Cohesion
                        cohesion += other_pos
                        # Separation
                        if dist < 30.0:
                            diff = pos - other_pos
                            separation += diff / dist
                        count += 1
        
        if count > 0:
            # Combine flocking forces
            alignment = alignment / ti.cast(count, ti.f32)
            cohesion = cohesion / ti.cast(count, ti.f32) - pos
            
            # Weight the forces
            force = alignment * 0.5 + cohesion * 0.3 + separation * 2.0
        
        return force
    
    @ti.func
    def wander_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Random wandering for exploration."""
        angle = (ti.random() - 0.5 + particle_idx * 0.0001) * 2.0 * 3.14159
        magnitude = 10.0
        force = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * magnitude
        return force
    
    @ti.func
    def boundary_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
        """Keep organisms in bounds."""
        force = ti.math.vec2(0.0, 0.0)
        margin = 50.0
        strength = 300.0
        
        if pos.x < margin:
            force.x += strength * (1.0 - pos.x / margin)
        elif pos.x > tv.x - margin:
            force.x -= strength * (1.0 - (tv.x - pos.x) / margin)
        
        if pos.y < margin:
            force.y += strength * (1.0 - pos.y / margin)
        elif pos.y > tv.y - margin:
            force.y -= strength * (1.0 - (tv.y - pos.y) / margin)
        
        return force
    
    # === Ecosystem Dynamics ===
    @ti.kernel
    def update_ecosystem():
        """Handle energy, feeding, death, and reproduction."""
        dt = 0.15  # Per-frame display step multiplier (NOT real seconds); 0.10-0.20 for visible motion
        energy_consumption = tv.s.llm_global.field[0].starvation_rate
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                species = tv.p.field[i].species
                energy = tv.s.llm_particles.field[i].energy
                
                # Energy consumption
                energy -= tv.s.llm_species.field[species].energy_consumption * dt * 60.0
                
                # Update age
                tv.s.llm_particles.field[i].age += dt
                
                # Update reproduction cooldown
                if tv.s.llm_particles.field[i].reproduction_cooldown > 0:
                    tv.s.llm_particles.field[i].reproduction_cooldown -= dt * 10.0
                
                if tv.s.llm_species.field[species].is_predator < 0.5:
                    # Prey grazing
                    xi = ti.cast(pos.x / 10, ti.i32)
                    yi = ti.cast(pos.y / 10, ti.i32)
                    
                    if xi >= 0 and xi < tv.x // 10 and yi >= 0 and yi < tv.y // 10:
                        available = vegetation_field[xi, yi]
                        if available > 0:
                            # Eat vegetation
                            eaten = min(available, 10.0 * dt * 60.0)
                            vegetation_field[xi, yi] -= eaten
                            energy += eaten * 2.0
                else:
                    # Predator hunting
                    target_idx = tv.s.llm_particles.field[i].hunting_target
                    attack_range = tv.s.llm_species.field[species].attack_range
                    
                    if target_idx >= 0 and target_idx < tv.pn and tv.p.field[target_idx].active > 0:
                        target_pos = tv.p.field[target_idx].pos
                        dist = (target_pos - pos).norm()
                        
                        if dist < attack_range:
                            # Successful hunt!
                            prey_energy = tv.s.llm_particles.field[target_idx].energy
                            transfer = tv.s.llm_global.field[0].energy_transfer_efficiency
                            energy += prey_energy * transfer
                            
                            # Kill prey
                            tv.p.field[target_idx].active = 0.0
                            tv.s.llm_particles.field[i].hunting_target = -1
                
                # Update energy
                tv.s.llm_particles.field[i].energy = energy
                
                # Death from starvation
                if energy <= 0:
                    tv.p.field[i].active = 0.0
                
                # Update size based on energy
                base_size = 3.0 if tv.s.llm_species.field[species].is_predator < 0.5 else 5.0
                tv.p.field[i].size = base_size * (0.5 + min(energy / 100.0, 1.5))
    
    @ti.kernel
    def handle_reproduction():
        """Organisms reproduce when they have enough energy."""
        threshold = tv.s.llm_global.field[0].reproduction_threshold
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                energy = tv.s.llm_particles.field[i].energy
                species = tv.p.field[i].species
                cooldown = tv.s.llm_particles.field[i].reproduction_cooldown
                
                if energy > threshold and cooldown <= 0:
                    # Find an inactive particle to use as offspring
                    for j in range(tv.pn):
                        if tv.p.field[j].active < 0.5:
                            # Create offspring
                            parent_pos = tv.p.field[i].pos
                            
                            # Offspring appears near parent
                            offset_angle = ti.random() * 2.0 * 3.14159
                            offset_dist = 20.0
                            
                            tv.p.field[j].active = 1.0
                            tv.p.field[j].pos = parent_pos + ti.Vector([
                                ti.cos(offset_angle) * offset_dist,
                                ti.sin(offset_angle) * offset_dist
                            ])
                            
                            # Random initial velocity
                            angle = ti.random() * 2.0 * 3.14159
                            speed = 20.0 + ti.random() * 20.0
                            tv.p.field[j].vel = ti.Vector([
                                ti.cos(angle) * speed,
                                ti.sin(angle) * speed
                            ])
                            
                            # Inherit species
                            tv.p.field[j].species = species
                            tv.p.field[j].size = 2.0
                            tv.p.field[j].mass = 1.0
                            tv.p.field[j].speed = 1.0
                            
                            # Split energy with parent
                            cost = tv.s.llm_species.field[species].reproduction_cost
                            tv.s.llm_particles.field[i].energy -= cost
                            tv.s.llm_particles.field[j].energy = cost * 0.8
                            tv.s.llm_particles.field[j].age = 0.0
                            tv.s.llm_particles.field[j].hunting_target = -1
                            tv.s.llm_particles.field[j].fleeing_from = -1
                            tv.s.llm_particles.field[j].reproduction_cooldown = 50.0
                            
                            # Parent cooldown
                            tv.s.llm_particles.field[i].reproduction_cooldown = 50.0
                            
                            break
    
    @ti.kernel
    def grow_vegetation():
        """Vegetation regrows over time."""
        growth_rate = tv.s.llm_global.field[0].vegetation_growth_rate
        
        for x in range(tv.x // 10):
            for y in range(tv.y // 10):
                current = vegetation_field[x, y]
                # Logistic growth
                max_capacity = 100.0
                growth = growth_rate * current * (1.0 - current / max_capacity)
                vegetation_field[x, y] = min(max_capacity, current + growth)
    
    # === Integration Kernel ===
    @ti.kernel
    def apply_all_experts():
        """Apply behavioral forces to organisms."""
        dt = 0.15  # Per-frame display step multiplier (NOT real seconds); 0.10-0.20 for visible motion
        
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                vel = tv.p.field[i].vel
                mass = tv.p.field[i].mass
                species = tv.p.field[i].species
                
                # Accumulate forces
                total_force = ti.math.vec2(0.0, 0.0)
                
                # Main behaviors
                total_force += hunting_force(pos, vel, mass, species, i) * 2.0
                total_force += grazing_force(pos, vel, mass, species, i)
                total_force += flocking_force(pos, vel, mass, species, i) * 0.5
                total_force += wander_force(pos, vel, mass, species, i) * 0.3
                total_force += boundary_force(pos, vel, mass, species, i)
                
                # Energy affects movement
                energy_factor = min(1.0, tv.s.llm_particles.field[i].energy / 50.0)
                total_force *= energy_factor
                
                # Limit force
                force_norm = total_force.norm()
                max_force = 500.0
                limited_force = total_force
                if force_norm > max_force:
                    limited_force = (total_force / force_norm) * max_force
                
                # Update velocity
                acceleration = limited_force / mass if mass > 0 else limited_force
                tv.p.field[i].vel += acceleration * dt
                
                # Apply damping
                tv.p.field[i].vel *= 0.98
                
                # Limit speed based on species and energy
                vel_norm = tv.p.field[i].vel.norm()
                max_speed = tv.s.llm_species.field[species].max_speed * energy_factor
                if vel_norm > max_speed:
                    tv.p.field[i].vel = (tv.p.field[i].vel / vel_norm) * max_speed
                
                # Update position
                tv.p.field[i].pos += tv.p.field[i].vel * tv.p.field[i].speed * dt
    
    # === Drawing Functions ===
    @ti.kernel
    def draw_vegetation():
        """Draw vegetation field."""
        for x in range(tv.x // 10):
            for y in range(tv.y // 10):
                veg = vegetation_field[x, y]
                if veg > 1.0:
                    intensity = min(veg / 100.0, 1.0)
                    color = ti.math.vec4(
                        0.1,
                        intensity * 0.4,
                        0.05,
                        intensity * 0.6
                    )
                    tv.px.rect(x * 10, y * 10, 10, 10, color)
    
    @ti.func
    def draw_energy_bars(i: ti.i32):
        """Draw energy bars above organisms."""
        if tv.p.field[i].active > 0:
            pos = tv.p.field[i].pos
            energy = tv.s.llm_particles.field[i].energy
            species = tv.p.field[i].species
            color = ti.math.vec4(1.0, 1.0, 1.0, 0.8)
            
            # Energy bar position
            bar_x = ti.cast(pos.x - 10, ti.i32)
            bar_y = ti.cast(pos.y - 15, ti.i32)
            
            # Bar width based on energy
            bar_width = ti.cast(min(20.0, energy / 5.0), ti.i32)
            
            # Color based on energy level
            if energy > 60:
                color = ti.math.vec4(0.2, 0.8, 0.2, 0.8)  # Green
            elif energy > 30:
                color = ti.math.vec4(0.8, 0.8, 0.2, 0.8)  # Yellow
            else:
                color = ti.math.vec4(0.8, 0.2, 0.2, 0.8)  # Red
            
            if bar_width > 0:
                tv.px.rect(bar_x, bar_y, bar_width, 3, color)
    
    @ti.func
    def draw_hunting_lines(i: ti.i32):
        """Draw lines between predators and their prey."""
        species = tv.p.field[i].species
        if tv.s.llm_species.field[species].is_predator > 0.5:
            target_idx = tv.s.llm_particles.field[i].hunting_target
            if target_idx >= 0 and target_idx < tv.pn and tv.p.field[target_idx].active > 0:
                pos = tv.p.field[i].pos
                target_pos = tv.p.field[target_idx].pos
                
                # Red hunting line
                color = ti.math.vec4(0.8, 0.2, 0.2, 0.3)
                tv.px.line(pos.x, pos.y, target_pos.x, target_pos.y, color)
    
    # === Drawing Kernel ===
    @ti.kernel
    def draw_visuals():
        """Draw visual effects."""
        # Draw energy bars
        for i in range(tv.pn):
            draw_energy_bars(i)
        
        # Draw hunting lines for first few predators
        for i in range(min(100, tv.pn)):
            if tv.p.field[i].active > 0:
                draw_hunting_lines(i)
    
    # Frame counter
    frame_count = ti.field(ti.i32, shape=())
    frame_count[None] = 0
    
    # === Render Loop ===
    @tv.render
    def _():
        tv.px.diffuse(0.98)
        
        # Draw environment
        draw_vegetation()
        
        # Update ecosystem
        frame_count[None] += 1
        if frame_count[None] % 3 == 0:
            update_ecosystem()
            handle_reproduction()
            grow_vegetation()
        
        # Draw effects
        draw_visuals()
        
        # Apply movement
        apply_all_experts()
        tv.p()
        
        # Render organisms
        tv.px.particles(tv.p, tv.s.species())
        
        return tv.px

if __name__ == "__main__":
    run(main)