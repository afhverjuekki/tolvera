#!/usr/bin/env python3
"""
Test script for the fixed PoE implementation.
This validates that experts are dynamically executed, not routed to templates.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
from tolvera import Tolvera
from tolvera.llm.poe_core import PoEBehaviorSystemV2, SimpleProgrammaticExpert
from tolvera.llm.poe_integration import TolveraBehaviorAgentFixed
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_dynamic_expert_execution():
    """Test that experts are dynamically executed, not template-routed."""
    print("🧪 Testing Dynamic Expert Execution")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(width=800, height=600, pn=100, px="pixels")
    
    # Create behavior agent
    agent = TolveraBehaviorAgentFixed(tv)
    
    # Create a custom expert with unique behavior that wouldn't match any template
    custom_expert_code = """@ti.func
def expert_spiral_wave(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # This creates a spiral wave pattern - no template would match this
    pos = tv.p.field[i].pos
    
    # Calculate angle from center
    center = ti.Vector([400.0, 300.0])
    diff = pos - center
    angle = ti.atan2(diff[1], diff[0])
    dist = diff.norm()
    
    # Create spiral force
    spiral_angle = angle + dist * 0.01
    force_magnitude = 2.0 * ti.sin(dist * 0.05)
    
    force = ti.Vector([
        ti.cos(spiral_angle) * force_magnitude,
        ti.sin(spiral_angle) * force_magnitude
    ])
    
    return force
"""
    
    print("Adding custom spiral wave expert...")
    agent.add_expert_from_code("spiral_wave", custom_expert_code, weight=1.0)
    
    # Add another unique expert
    zigzag_expert_code = """@ti.func
def expert_zigzag_motion(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Zigzag motion based on particle ID - definitely not in any template
    pos = tv.p.field[i].pos
    
    # Create zigzag pattern
    phase = float(i) * 0.1
    zigzag_x = ti.sin(pos[1] * 0.02 + phase) * 3.0
    zigzag_y = ti.cos(pos[0] * 0.02 + phase) * 1.5
    
    return ti.Vector([zigzag_x, zigzag_y])
"""
    
    print("Adding custom zigzag motion expert...")
    agent.add_expert_from_code("zigzag_motion", zigzag_expert_code, weight=0.5)
    
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
            tv.p.field[i].species = i % 3
    
    init_particles()
    
    # Set species colors
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]  # Red
    tv.s.species.field[1].rgba = [0.3, 1.0, 0.3, 1.0]  # Green
    tv.s.species.field[2].rgba = [0.3, 0.3, 1.0, 1.0]  # Blue
    
    print("\nExpert info:")
    for info in agent.get_expert_info():
        print(f"  - {info['name']}: weight={info['weight']}")
    
    frame_count = 0
    
    @tv.render
    def render():
        nonlocal frame_count
        agent.update(dt=0.016)
        
        frame_count += 1
        if frame_count % 60 == 0:
            print(f"Frame {frame_count}: Experts executing dynamically")
        
        return tv.px
    
    print("\n✅ If you see spiral and zigzag patterns, dynamic execution is working!")
    print("❌ If particles just drift randomly, template routing is still active.")
    print("\nPress 'Esc' to exit")
    tv.run()


def test_weight_combination():
    """Test that PoE weight combination works correctly."""
    print("\n🧪 Testing PoE Weight Combination")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(width=800, height=600, pn=200, px="pixels")
    
    # Create behavior agent  
    agent = TolveraBehaviorAgentFixed(tv)
    
    # Add opposing experts with different weights
    left_expert = """@ti.func
def expert_left_force(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    return ti.Vector([-5.0, 0.0])  # Constant left force
"""
    
    right_expert = """@ti.func
def expert_right_force(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    return ti.Vector([10.0, 0.0])  # Stronger right force
"""
    
    print("Adding left force expert (weight=0.3)...")
    agent.add_expert_from_code("left_force", left_expert, weight=0.3)
    
    print("Adding right force expert (weight=0.7)...")
    agent.add_expert_from_code("right_force", right_expert, weight=0.7)
    
    # Initialize particles in center
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([tv.x * 0.5, tv.y * 0.5])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = 0
    
    init_particles()
    
    tv.s.species.field[0].rgba = [0.8, 0.8, 0.2, 1.0]  # Yellow
    
    @tv.render
    def render():
        agent.update(dt=0.016)
        return tv.px
    
    print("\n✅ Particles should move RIGHT (stronger weight)")
    print("The PoE combination should favor the right force expert.")
    print("\nPress 'Esc' to exit")
    tv.run()


def test_runtime_expert_modification():
    """Test adding/modifying experts during runtime."""
    print("\n🧪 Testing Runtime Expert Modification")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(width=800, height=600, pn=150, px="pixels")
    
    # Create behavior agent
    agent = TolveraBehaviorAgentFixed(tv)
    
    # Start with a simple downward gravity
    gravity_expert = """@ti.func
def expert_gravity(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    return ti.Vector([0.0, 5.0])  # Downward gravity
"""
    
    print("Starting with gravity expert...")
    agent.add_expert_from_code("gravity", gravity_expert, weight=1.0)
    
    # Initialize particles
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([
                ti.random() * tv.x,
                ti.random() * tv.y * 0.3  # Start in upper third
            ])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = 0
    
    init_particles()
    
    tv.s.species.field[0].rgba = [0.5, 0.5, 1.0, 1.0]  # Light blue
    
    frame_count = 0
    stage = 0
    
    @tv.render
    def render():
        nonlocal frame_count, stage
        
        frame_count += 1
        
        # Add new behaviors at different times
        if frame_count == 120 and stage == 0:
            print("\n➕ Adding wind expert...")
            wind_expert = """@ti.func
def expert_wind(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Oscillating wind
    wind_strength = ti.sin(float(i) * 0.1) * 3.0
    return ti.Vector([wind_strength, 0.0])
"""
            agent.add_expert_from_code("wind", wind_expert, weight=0.5)
            stage = 1
            
        elif frame_count == 240 and stage == 1:
            print("\n🔄 Modifying gravity weight...")
            agent.set_expert_weight("gravity", 0.3)
            stage = 2
            
        elif frame_count == 360 and stage == 2:
            print("\n➕ Adding upward thermal expert...")
            thermal_expert = """@ti.func
def expert_thermal(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    pos = tv.p.field[i].pos
    # Upward thermal in center
    center_x = 400.0
    if ti.abs(pos[0] - center_x) < 100.0:
        return ti.Vector([0.0, -8.0])
    return ti.Vector([0.0, 0.0])
"""
            agent.add_expert_from_code("thermal", thermal_expert, weight=0.8)
            stage = 3
        
        agent.update(dt=0.016)
        return tv.px
    
    print("\n📊 Timeline:")
    print("  0s: Gravity only")
    print("  2s: Add wind")
    print("  4s: Reduce gravity")  
    print("  6s: Add thermal updraft")
    print("\nPress 'Esc' to exit")
    tv.run()


if __name__ == "__main__":
    print("\n🚀 PoE Fixed Implementation Test Suite")
    print("=" * 50)
    print("\nSelect a test:")
    print("1. Dynamic Expert Execution")
    print("2. Weight Combination")
    print("3. Runtime Modification")
    print("4. Run All Tests")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-4): ")
    
    if choice == "1":
        test_dynamic_expert_execution()
    elif choice == "2":
        test_weight_combination()
    elif choice == "3":
        test_runtime_expert_modification()
    elif choice == "4":
        test_dynamic_expert_execution()
        input("\nPress Enter to continue to next test...")
        test_weight_combination()
        input("\nPress Enter to continue to next test...")
        test_runtime_expert_modification()
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice.")