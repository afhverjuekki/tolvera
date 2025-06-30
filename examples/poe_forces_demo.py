#!/usr/bin/env python3
"""
Demonstration of the PoE (Product of Experts) behavior system for Tölvera
focusing on force behaviors without mouse interaction.

This example shows how to:
1. Use natural language to generate force behaviors
2. Test attract, repel, gravitate, noise, and centripetal forces
3. Combine multiple force behaviors using the PoE system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import taichi as ti
from tolvera import Tolvera
from src.tolvera.llm.poe_integration import TolveraBehaviorAgent, AsyncTolveraBehaviorAgent
from src.tolvera.llm.poe_synthesis import SimpleSynthesizer
from src.tolvera.llm.poe_core import SimpleProgrammaticExpert


def demo_attract_repel(**kwargs):
    """Demo with attraction and repulsion forces to fixed points."""
    print("🧲 Attract/Repel Forces Demo")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(**kwargs)
    
    # Create behavior agent
    behavior_agent = TolveraBehaviorAgent(tv)
    
    # Define attraction points
    attract_point_1 = ti.Vector([tv.x * 0.25, tv.y * 0.5])
    attract_point_2 = ti.Vector([tv.x * 0.75, tv.y * 0.5])
    
    # Add custom experts for attraction to fixed points
    attract_expert_1 = f"""@ti.func
def expert_attract_point_1(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Attract to left point
    pos = tv.p.field[i].pos
    target = ti.Vector([{attract_point_1[0]}, {attract_point_1[1]}])
    diff = target - pos
    dist = diff.norm()
    
    force = ti.Vector([0.0, 0.0])
    if dist > 10.0 and dist < 200.0:  # Only attract within range
        force = diff.normalized() * (1.0 / (dist * 0.01))  # Inverse distance
    
    return force
"""
    
    attract_expert_2 = f"""@ti.func
def expert_attract_point_2(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Attract to right point
    pos = tv.p.field[i].pos
    target = ti.Vector([{attract_point_2[0]}, {attract_point_2[1]}])
    diff = target - pos
    dist = diff.norm()
    
    force = ti.Vector([0.0, 0.0])
    if dist > 10.0 and dist < 200.0:  # Only attract within range
        force = diff.normalized() * (1.0 / (dist * 0.01))
    
    return force
"""
    
    # Repel from center
    repel_expert = f"""@ti.func
def expert_repel_center(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Repel from center
    pos = tv.p.field[i].pos
    center = ti.Vector([{tv.x * 0.5}, {tv.y * 0.5}])
    diff = pos - center
    dist = diff.norm()
    
    force = ti.Vector([0.0, 0.0])
    if dist < 150.0 and dist > 1.0:  # Repel within radius
        force = diff.normalized() * (2.0 / (dist * 0.02))
    
    return force
"""
    
    # Add experts
    print("Adding attraction and repulsion experts...")
    behavior_agent.add_expert_from_code("attract_left", attract_expert_1, weight=0.5)
    behavior_agent.add_expert_from_code("attract_right", attract_expert_2, weight=0.5)
    behavior_agent.add_expert_from_code("repel_center", repel_expert, weight=0.3)
    
    # Also add boundary forces
    behavior_agent.add_builtin_expert("boundary", weight=0.2)
    
    # Initialize particles
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([
                ti.random() * tv.x,
                ti.random() * tv.y
            ])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = i % tv.sn
    
    init_particles()
    
    # Drawing kernel for markers
    @ti.kernel
    def draw_markers():
        # Draw attraction points (green circles)
        tv.px.circle(
            ti.cast(attract_point_1[0], ti.i32), 
            ti.cast(attract_point_1[1], ti.i32), 
            5, 
            ti.Vector([0.0, 1.0, 0.0, 1.0]),  # Green
            fill=1
        )
        tv.px.circle(
            ti.cast(attract_point_2[0], ti.i32), 
            ti.cast(attract_point_2[1], ti.i32), 
            5, 
            ti.Vector([0.0, 1.0, 0.0, 1.0]),  # Green
            fill=1
        )
        
        # Draw repulsion center (red circle)
        tv.px.circle(
            ti.cast(tv.x * 0.5, ti.i32), 
            ti.cast(tv.y * 0.5, ti.i32), 
            10, 
            ti.Vector([1.0, 0.0, 0.0, 1.0]),  # Red
            fill=1
        )
    
    # Main render loop
    @tv.render
    def _():
        # Clear background
        tv.px.clear()
        
        # Draw markers
        draw_markers()
        
        # Update behavior system
        behavior_agent.update(dt=0.016)
        
        # Draw particles
        tv.px.particles(tv.p, tv.s.species, "circle")
        
        return tv.px
    
    print("\nRunning attract/repel demo...")
    print("Green circles: Attraction points")
    print("Red circle: Repulsion center")
    print("Press ESC to exit")


def demo_gravitation(**kwargs):
    """Demo with gravitational forces between particles."""
    print("🌍 Gravitation Forces Demo")
    print("=" * 50)
    
    # Initialize Tölvera with fewer particles for better visualization
    kwargs['pn'] = kwargs.get('pn', 100)
    tv = Tolvera(**kwargs)
    
    # Create behavior agent
    behavior_agent = TolveraBehaviorAgent(tv)
    
    # Gravitational expert
    gravitation_expert = """@ti.func
def expert_gravitation(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # N-body gravitation
    pos_i = tv.p.field[i].pos
    vel_i = tv.p.field[i].vel
    mass_i = tv.p.field[i].mass
    
    force = ti.Vector([0.0, 0.0])
    G = 50.0  # Gravitational constant
    
    for j in range(tv.pn):
        if i != j and tv.p.field[j].active > 0:
            pos_j = tv.p.field[j].pos
            mass_j = tv.p.field[j].mass
            
            diff = pos_j - pos_i
            dist_sq = diff.dot(diff) + 100.0  # Add small value to avoid singularity
            dist = ti.sqrt(dist_sq)
            
            if dist < 300.0:  # Only within range
                # F = G * m1 * m2 / r^2
                force_mag = G * mass_j / dist_sq
                force += diff.normalized() * force_mag
    
    return force * 0.01  # Scale down
"""
    
    # Add gravitational expert
    print("Adding gravitational expert...")
    behavior_agent.add_expert_from_code("gravitation", gravitation_expert, weight=1.0)
    
    # Add small damping
    damping_expert = """@ti.func
def expert_damping(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Velocity damping
    vel = tv.p.field[i].vel
    return -vel * 0.02  # Small damping
"""
    behavior_agent.add_expert_from_code("damping", damping_expert, weight=0.1)
    
    # Initialize particles with varied masses
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            
            # Create a few heavy masses and many light ones
            if i < 3:  # Heavy masses
                tv.p.field[i].mass = 10.0
                tv.p.field[i].size = 5.0
                tv.p.field[i].pos = ti.Vector([
                    tv.x * (0.3 + i * 0.2),
                    tv.y * 0.5
                ])
                tv.p.field[i].vel = ti.Vector([0.0, (i - 1) * 20.0])
            else:  # Light masses
                tv.p.field[i].mass = 0.5
                tv.p.field[i].size = 1.0
                angle = (i / tv.pn) * 2.0 * 3.14159
                radius = 100.0 + ti.random() * 50.0
                tv.p.field[i].pos = ti.Vector([
                    tv.x * 0.5 + radius * ti.cos(angle),
                    tv.y * 0.5 + radius * ti.sin(angle)
                ])
                # Orbital velocity
                tv.p.field[i].vel = ti.Vector([
                    -radius * ti.sin(angle) * 0.2,
                    radius * ti.cos(angle) * 0.2
                ])
            
            tv.p.field[i].species = i % tv.sn
    
    init_particles()
    
    # Main render loop
    @tv.render
    def _():
        # Fade trails
        tv.px.decay(0.98)
        
        # Update behavior system
        behavior_agent.update(dt=0.016)
        
        # Draw particles with size based on mass
        tv.px.particles(tv.p, tv.s.species, "circle")
        
        return tv.px
    
    print("\nRunning gravitation demo...")
    print("Large particles: Heavy masses")
    print("Small particles: Light masses")
    print("Watch orbital dynamics emerge!")
    print("Press ESC to exit")


async def demo_natural_language(**kwargs):
    """Demo using natural language descriptions to generate force behaviors."""
    print("🗣️ Natural Language Forces Demo")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(**kwargs)
    
    # Create async behavior agent
    behavior_agent = AsyncTolveraBehaviorAgent(tv)
    
    # Create synthesizer
    synthesizer = SimpleSynthesizer()
    
    # Natural language descriptions for various forces
    force_descriptions = [
        "particles are attracted to the center of the screen with strength inversely proportional to distance",
        "particles repel each other when they get too close, maintaining personal space",
        "particles experience a clockwise swirling force around the center",
        "particles have random jittery movement like brownian motion",
        "particles are pushed away from the edges of the screen"
    ]
    
    print("Synthesizing force behaviors from natural language...")
    print()
    
    for desc in force_descriptions:
        print(f"Description: {desc}")
        success = await behavior_agent.add_expert_from_description(desc, synthesizer)
        if success:
            print("  ✓ Successfully generated expert")
        else:
            print("  ✗ Failed to generate expert")
        print()
    
    # Initialize particles
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            # Start in a circle
            angle = (i / tv.pn) * 2.0 * 3.14159
            radius = min(tv.x, tv.y) * 0.3
            tv.p.field[i].pos = ti.Vector([
                tv.x * 0.5 + radius * ti.cos(angle),
                tv.y * 0.5 + radius * ti.sin(angle)
            ])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = i % tv.sn
    
    init_particles()
    
    # Main render loop
    @tv.render
    def _():
        # Create trails
        tv.px.decay(0.95)
        
        # Update behavior system
        behavior_agent.update(dt=0.016)
        
        # Draw particles
        tv.px.particles(tv.p, tv.s.species, "circle")
        
        return tv.px
    
    print("\nRunning natural language forces demo...")
    print("Watch how the combined behaviors create complex dynamics!")
    print("Press ESC to exit")


def demo_flocking(**kwargs):
    """Demo with flocking behavior (separation, alignment, cohesion)."""
    print("🐦 Flocking Behavior Demo")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(**kwargs)
    
    # Create behavior agent
    behavior_agent = TolveraBehaviorAgent(tv)
    
    # Separation expert
    separation_expert = """@ti.func
def expert_separation(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Separation: steer to avoid crowding local flockmates
    pos_i = tv.p.field[i].pos
    species_i = tv.p.field[i].species
    
    steer = ti.Vector([0.0, 0.0])
    count = 0
    desired_separation = 25.0
    
    for j in range(tv.pn):
        if i != j and tv.p.field[j].active > 0:
            if tv.p.field[j].species == species_i:  # Same species
                diff = pos_i - tv.p.field[j].pos
                dist = diff.norm()
                
                if dist > 0 and dist < desired_separation:
                    # Weight by distance
                    diff_norm = diff.normalized() / dist
                    steer += diff_norm
                    count += 1
    
    if count > 0:
        steer /= float(count)
    
    return steer * 2.0
"""
    
    # Alignment expert
    alignment_expert = """@ti.func
def expert_alignment(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Alignment: steer towards average heading of local flockmates
    pos_i = tv.p.field[i].pos
    vel_i = tv.p.field[i].vel
    species_i = tv.p.field[i].species
    
    avg_vel = ti.Vector([0.0, 0.0])
    count = 0
    neighbor_dist = 50.0
    
    for j in range(tv.pn):
        if i != j and tv.p.field[j].active > 0:
            if tv.p.field[j].species == species_i:  # Same species
                dist = (tv.p.field[j].pos - pos_i).norm()
                
                if dist > 0 and dist < neighbor_dist:
                    avg_vel += tv.p.field[j].vel
                    count += 1
    
    if count > 0:
        avg_vel /= float(count)
        # Steering = desired - current
        steer = avg_vel - vel_i
        return steer * 0.1
    
    return ti.Vector([0.0, 0.0])
"""
    
    # Cohesion expert
    cohesion_expert = """@ti.func
def expert_cohesion(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Cohesion: steer to move toward average position of local flockmates
    pos_i = tv.p.field[i].pos
    species_i = tv.p.field[i].species
    
    center = ti.Vector([0.0, 0.0])
    count = 0
    neighbor_dist = 50.0
    
    for j in range(tv.pn):
        if i != j and tv.p.field[j].active > 0:
            if tv.p.field[j].species == species_i:  # Same species
                dist = (tv.p.field[j].pos - pos_i).norm()
                
                if dist > 0 and dist < neighbor_dist:
                    center += tv.p.field[j].pos
                    count += 1
    
    if count > 0:
        center /= float(count)
        # Seek center
        return (center - pos_i) * 0.01
    
    return ti.Vector([0.0, 0.0])
"""
    
    # Add flocking experts
    print("Adding flocking behavior experts...")
    behavior_agent.add_expert_from_code("separation", separation_expert, weight=1.5)
    behavior_agent.add_expert_from_code("alignment", alignment_expert, weight=1.0)
    behavior_agent.add_expert_from_code("cohesion", cohesion_expert, weight=1.0)
    
    # Add boundary and some noise
    behavior_agent.add_builtin_expert("boundary", weight=0.5)
    behavior_agent.add_builtin_expert("noise", weight=0.1)
    
    # Initialize particles in groups by species
    @ti.kernel
    def init_particles():
        particles_per_species = tv.pn // tv.sn
        
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            species = i // particles_per_species
            tv.p.field[i].species = species
            
            # Start each species in a different area
            center_x = tv.x * (0.2 + (species % 3) * 0.3)
            center_y = tv.y * (0.2 + (species // 3) * 0.3)
            
            # Random position around species center
            tv.p.field[i].pos = ti.Vector([
                center_x + (ti.random() - 0.5) * 50.0,
                center_y + (ti.random() - 0.5) * 50.0
            ])
            
            # Random initial velocity
            angle = ti.random() * 2.0 * 3.14159
            speed = 20.0
            tv.p.field[i].vel = ti.Vector([
                speed * ti.cos(angle),
                speed * ti.sin(angle)
            ])
    
    init_particles()
    
    # Main render loop
    @tv.render
    def _():
        # Slight fade for trails
        tv.px.decay(0.99)
        
        # Update behavior system
        behavior_agent.update(dt=0.016)
        
        # Draw particles
        tv.px.particles(tv.p, tv.s.species, "circle")
        
        return tv.px
    
    print("\nRunning flocking demo...")
    print("Each color represents a different species")
    print("Watch as they form flocks with emergent behavior!")
    print("Press ESC to exit")


def main(**kwargs):
    """Main entry point with demo selection."""
    import fire
    
    demos = {
        'attract_repel': demo_attract_repel,
        'gravitation': demo_gravitation,
        'natural_language': lambda **kw: asyncio.run(demo_natural_language(**kw)),
        'flocking': demo_flocking,
    }
    
    # Check if demo type specified
    demo_type = kwargs.pop('demo', 'attract_repel')
    
    if demo_type == 'all':
        print("Available force demos:")
        for name, desc in [
            ('attract_repel', 'Attraction to points and repulsion from center'),
            ('gravitation', 'N-body gravitational simulation'),
            ('natural_language', 'Forces generated from natural language'),
            ('flocking', 'Separation, alignment, and cohesion behaviors')
        ]:
            print(f"  {name}: {desc}")
        return
    
    if demo_type not in demos:
        print(f"Unknown demo type: {demo_type}")
        print(f"Available demos: {', '.join(demos.keys())}")
        print("Use 'all' to see descriptions")
        return
    
    # Set default parameters if not provided
    kwargs.setdefault('width', 800)
    kwargs.setdefault('height', 600)
    kwargs.setdefault('pn', 300)  # Number of particles
    kwargs.setdefault('sn', 5)    # Number of species
    
    # Run selected demo
    demos[demo_type](**kwargs)


if __name__ == "__main__":
    import fire
    fire.Fire(main)