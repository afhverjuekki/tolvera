#!/usr/bin/env python3
"""
Fixed demonstration of the PoE (Product of Experts) behavior system for Tölvera
using LLM synthesis instead of hardcoded templates.

This example shows how to:
1. Use natural language to generate force behaviors via LLM
2. Test attract, repel, gravitate, noise, and centripetal forces
3. Combine multiple force behaviors using the PoE system

Key difference: This version uses actual LLM synthesis to generate experts
from natural language descriptions, following the true PoE methodology.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import taichi as ti
from tolvera import Tolvera
from tolvera.llm.poe_core import PoEBehaviorSystemV2
from src.tolvera.llm.poe_integration import TolveraBehaviorAgent
from src.tolvera.llm.poe_ollama import PoEExpertSynthesizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_attract_repel_llm(**kwargs):
    """Demo with attraction and repulsion forces using LLM synthesis."""
    print("🧲 Attract/Repel Forces Demo (LLM Synthesized)")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(**kwargs)
    
    # Create PoE behavior system
    poe_system = PoEBehaviorSystemV2(tv)
    
    # Initialize LLM synthesizer
    synthesizer = PoEExpertSynthesizer(model_name="llama3.2:3b")
    
    # Generate experts from natural language descriptions
    print("\n📝 Generating experts from natural language descriptions...\n")
    
    descriptions = [
        "particles are attracted to the left quarter of the screen with moderate strength",
        "particles are attracted to the right quarter of the screen with moderate strength",
        "particles are repelled from the center of the screen within a 150 pixel radius",
        "particles avoid the edges of the screen with a 50 pixel margin"
    ]
    
    for desc in descriptions:
        print(f"Generating expert for: '{desc}'")
        try:
            await poe_system.add_expert_from_description(desc, synthesizer, weight=0.5)
            print("✅ Success\n")
        except Exception as e:
            logger.error(f"Failed to generate expert: {e}")
    
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
            tv.p.field[i].species = ti.random(ti.i32) % 2
    
    init_particles()
    
    # Set species colors
    tv.s.species.field[0].rgba = [0.2, 0.8, 1.0, 1.0]  # Cyan
    tv.s.species.field[1].rgba = [1.0, 0.5, 0.2, 1.0]  # Orange
    
    # Render loop
    @tv.render
    def render():
        poe_system.compute_and_apply_forces(
            {"mouse_x": tv.mouse.x, "mouse_y": tv.mouse.y}, 
            dt=0.016
        )
        return tv.px
    
    print("\n🎮 Running simulation...")
    print("Press 'Esc' to exit")
    tv.run()


async def demo_gravitation_llm(**kwargs):
    """Demo with gravitational forces using LLM synthesis."""
    print("🌍 Gravitation Forces Demo (LLM Synthesized)")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(**kwargs)
    
    # Create PoE behavior system
    poe_system = PoEBehaviorSystemV2(tv)
    
    # Initialize LLM synthesizer
    synthesizer = PoEExpertSynthesizer(model_name="llama3.2:3b")
    
    # Generate gravity-related experts
    print("\n📝 Generating gravity experts from natural language...\n")
    
    descriptions = [
        "particles experience a constant downward gravitational force like falling objects",
        "particles are attracted to each other with force proportional to inverse square distance, like gravitational bodies",
        "particles slow down over time due to air resistance or friction",
        "particles bounce elastically when they hit the bottom edge of the screen"
    ]
    
    for desc in descriptions:
        print(f"Generating expert for: '{desc}'")
        try:
            await poe_system.add_expert_from_description(desc, synthesizer, weight=0.7)
            print("✅ Success\n")
        except Exception as e:
            logger.error(f"Failed to generate expert: {e}")
    
    # Initialize particles
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([
                ti.random() * tv.x,
                ti.random() * tv.y * 0.5  # Start in upper half
            ])
            tv.p.field[i].vel = ti.Vector([
                (ti.random() - 0.5) * 2.0,
                0.0
            ])
            tv.p.field[i].species = 0
            tv.p.field[i].size = 2.0 + ti.random() * 4.0
    
    init_particles()
    
    # Set species color
    tv.s.species.field[0].rgba = [0.8, 0.8, 0.2, 1.0]  # Yellow
    
    # Render loop
    @tv.render
    def render():
        poe_system.compute_and_apply_forces({}, dt=0.016)
        return tv.px
    
    print("\n🎮 Running simulation...")
    print("Press 'Esc' to exit")
    tv.run()


async def demo_complex_behaviors_llm(**kwargs):
    """Demo with complex emergent behaviors using LLM synthesis."""
    print("🌌 Complex Behaviors Demo (LLM Synthesized)")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(**kwargs)
    
    # Create PoE behavior system
    poe_system = PoEBehaviorSystemV2(tv)
    
    # Initialize LLM synthesizer
    synthesizer = PoEExpertSynthesizer(model_name="llama3.2:3b")
    
    # Generate complex behavior experts
    print("\n📝 Generating complex behavior experts...\n")
    
    descriptions = [
        "particles swirl in a clockwise vortex pattern around the center of the screen",
        # "particles form small flocking groups that move together like birds",
        # "particles create wave-like patterns by oscillating up and down based on their x position",
        # "particles are attracted to their nearest neighbors but repelled if they get too close",
        # "particles accelerate when moving in the same direction as their neighbors"
    ]
    
    for desc in descriptions:
        print(f"Generating expert for: '{desc}'")
        try:
            weight = 0.6 if "vortex" in desc else 0.4
            await poe_system.add_expert_from_description(desc, synthesizer, weight=weight)
            print("✅ Success\n")
        except Exception as e:
            logger.error(f"Failed to generate expert: {e}")
    
    # Initialize particles
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([
                ti.random() * tv.x,
                ti.random() * tv.y
            ])
            tv.p.field[i].vel = ti.Vector([
                (ti.random() - 0.5) * 0.5,
                (ti.random() - 0.5) * 0.5
            ])
            tv.p.field[i].species = i % 3
            tv.p.field[i].size = 1.5
    
    init_particles()
    
    # Set species colors
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]  # Red
    tv.s.species.field[1].rgba = [0.3, 1.0, 0.3, 1.0]  # Green
    tv.s.species.field[2].rgba = [0.3, 0.3, 1.0, 1.0]  # Blue
    
    # Render loop
    @tv.render
    def render():
        poe_system.compute_and_apply_forces(
            {"mouse_x": tv.x/2, "mouse_y": tv.y/2}, 
            dt=0.016
        )
        return tv.px
    
    print("\n🎮 Running simulation...")
    print("Press 'Esc' to exit")
    tv.run()


async def demo_interactive_synthesis(**kwargs):
    """Interactive demo where users can add behaviors in real-time."""
    print("🎨 Interactive Behavior Synthesis Demo")
    print("=" * 50)
    
    # Initialize Tölvera
    tv = Tolvera(**kwargs)
    
    # Create PoE behavior system
    poe_system = PoEBehaviorSystemV2(tv)
    
    # Initialize LLM synthesizer
    synthesizer = PoEExpertSynthesizer(model_name="llama3.2:3b")
    
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
            tv.p.field[i].species = ti.random(ti.i32) % 4
            tv.p.field[i].size = 2.0
    
    init_particles()
    
    # Set species colors
    tv.s.species.field[0].rgba = [1.0, 0.5, 0.5, 1.0]  # Light red
    tv.s.species.field[1].rgba = [0.5, 1.0, 0.5, 1.0]  # Light green
    tv.s.species.field[2].rgba = [0.5, 0.5, 1.0, 1.0]  # Light blue
    tv.s.species.field[3].rgba = [1.0, 1.0, 0.5, 1.0]  # Light yellow
    
    print("\n📝 Enter behavior descriptions (or 'quit' to exit):")
    print("Example: 'particles spiral outward from the center'")
    print("Example: 'red particles chase blue particles'")
    print("Example: 'particles form a heart shape'\n")
    
    # Start with a basic behavior
    await poe_system.add_expert_from_description(
        "particles slowly drift with random brownian motion", 
        synthesizer, 
        weight=0.3
    )
    
    # Run in background
    running = True
    
    async def input_loop():
        nonlocal running
        while running:
            try:
                description = await asyncio.get_event_loop().run_in_executor(
                    None, input, "Enter behavior: "
                )
                
                if description.lower() == 'quit':
                    running = False
                    break
                
                if description.strip():
                    print(f"\n🔄 Synthesizing: '{description}'")
                    try:
                        await poe_system.add_expert_from_description(
                            description, synthesizer, weight=0.5
                        )
                        print("✅ Added successfully!\n")
                    except Exception as e:
                        print(f"❌ Failed: {e}\n")
                        
            except Exception as e:
                logger.error(f"Input error: {e}")
                break
    
    # Start input task
    input_task = asyncio.create_task(input_loop())
    
    # Render loop
    @tv.render
    def render():
        if not running:
            tv.stop()
            return tv.px
            
        poe_system.compute_and_apply_forces(
            {"mouse_x": tv.mouse.x, "mouse_y": tv.mouse.y}, 
            dt=0.016
        )
        return tv.px
    
    print("\n🎮 Running simulation...")
    tv.run()
    
    # Clean up
    running = False
    input_task.cancel()
    try:
        await input_task
    except asyncio.CancelledError:
        pass


# Main menu
async def main():
    print("\n🚀 PoE Forces Demo - LLM Synthesized")
    print("=" * 50)
    print("\nSelect a demo:")
    print("1. Attract/Repel Forces")
    print("2. Gravitation Forces")
    print("3. Complex Emergent Behaviors")
    print("4. Interactive Synthesis")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-4): ")
    
    # Default parameters
    import sys
    params = {
        "width": 1280,
        "height": 720,
        "pn": 500,
        "px": "canvas",
        "gpu": "metal" if sys.platform == "darwin" else "cuda"
    }
    
    if choice == "1":
        await demo_attract_repel_llm(**params)
    elif choice == "2":
        await demo_gravitation_llm(**params)
    elif choice == "3":
        await demo_complex_behaviors_llm(**params)
    elif choice == "4":
        await demo_interactive_synthesis(**params)
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice. Please try again.")
        await main()


if __name__ == "__main__":
    asyncio.run(main())