#!/usr/bin/env python3
"""
Simple demonstration of the PoE (Product of Experts) behavior system for Tölvera.

This example shows how to:
1. Use natural language to generate force behaviors via LLM
2. See the raw LLM output (no templates or fallbacks)
3. Combine multiple behaviors using the PoE system
4. Understand the limitations of raw LLM generation

Behaviors demonstrated:
- Gravity (particles fall down)
- Center attraction (particles move toward center)
- Particle repulsion (particles push each other away)
- Spiral motion (particles move in spirals)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import taichi as ti
from tolvera import Tolvera
from src.tolvera.llm.poe_integration import TolveraBehaviorAgent
from src.tolvera.llm.poe_synthesis import PureLLMSynthesizer
import logging

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_simple_behaviors():
    """Demonstrate basic particle behaviors using pure LLM generation."""
    
    print("\n" + "="*80)
    print("🚀 SIMPLE PoE BEHAVIOR DEMO - Pure LLM Generation")
    print("="*80)
    print("\nThis demo shows raw LLM output - no templates or fallbacks!")
    print("Watch for successes AND failures to understand limitations.\n")
    
    # Initialize Tölvera
    print("📊 Initializing Tölvera...")
    tv = Tolvera(
        width=800,
        height=600,
        pn=500,  # 500 particles
        px="pixels",
        gpu="metal" if sys.platform == "darwin" else "cuda"
    )
    
    # Create behavior agent
    agent = TolveraBehaviorAgent(tv)
    
    # Create LLM synthesizer
    print("🤖 Initializing LLM synthesizer...")
    synthesizer = PureLLMSynthesizer(model_name="qwen2.5:3b")
    
    # Initialize particles
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([
                ti.random() * tv.x,
                ti.random() * tv.y * 0.5  # Start in upper half
            ])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = i % 3
            tv.p.field[i].size = 2.0
    
    init_particles()
    
    # Set species colors
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]  # Red
    tv.s.species.field[1].rgba = [0.3, 1.0, 0.3, 1.0]  # Green
    tv.s.species.field[2].rgba = [0.3, 0.3, 1.0, 1.0]  # Blue
    
    # Test different behavior descriptions
    behaviors = [
        ("particles fall downward with gravity", 1.0),
        ("particles are attracted to the center of the screen", 0.5),
        ("particles repel each other when they get too close", 0.3),
        ("particles move in a clockwise spiral pattern", 0.2),
    ]
    
    print(f"\n📝 Attempting to generate {len(behaviors)} behaviors:")
    for desc, weight in behaviors:
        print(f"   - {desc} (weight: {weight})")
    
    # Generate experts from descriptions
    successful_experts = []
    
    for description, weight in behaviors:
        try:
            print(f"\n{'='*60}")
            print(f"Generating: '{description}'")
            print(f"{'='*60}")
            
            expert = await agent.add_expert_from_description(
                description,
                synthesizer.synthesizer,  # Pass the inner synthesizer
                weight=weight
            )
            successful_experts.append((expert.name, description))
            
        except Exception as e:
            print(f"\n⚠️  Failed to generate expert: {e}")
            print("This is expected with raw LLM output - not all generations succeed!")
    
    print(f"\n📊 Successfully generated {len(successful_experts)} out of {len(behaviors)} experts:")
    for name, desc in successful_experts:
        print(f"   ✓ {name}: {desc}")
    
    if not successful_experts:
        print("\n❌ No experts were successfully generated. Please check:")
        print("   1. Ollama is running ('ollama serve')")
        print("   2. The model is available ('ollama pull qwen2.5:3b')")
        return
    
    # Show expert info
    print("\n📋 Active Experts:")
    for info in agent.get_expert_info():
        print(f"   - {info['name']}: weight={info['weight']:.2f}")
    
    # Render loop
    frame_count = 0
    show_info_interval = 120  # Show info every 2 seconds at 60fps
    
    @tv.render
    def render():
        nonlocal frame_count
        
        # Update behavior system
        agent.update(dt=0.016)
        
        # Show periodic info
        frame_count += 1
        if frame_count % show_info_interval == 0:
            avg_speed = 0.0
            active_count = 0
            for i in range(tv.pn):
                if tv.p.field[i].active > 0:
                    avg_speed += tv.p.field[i].vel.norm()
                    active_count += 1
            if active_count > 0:
                avg_speed /= active_count
                print(f"\n⏱️  Frame {frame_count}: Avg particle speed: {avg_speed:.2f}")
        
        return tv.px
    
    print("\n🎮 Running simulation...")
    print("Press 'Esc' to exit")
    print("\nObserve:")
    print("- How particles behave based on LLM-generated code")
    print("- The combination of multiple experts")
    print("- Any unexpected behaviors from raw LLM output")
    
    tv.run()


async def demo_custom_behavior():
    """Allow user to input custom behavior descriptions."""
    
    print("\n" + "="*80)
    print("🎨 CUSTOM BEHAVIOR DEMO - Your Ideas, LLM Generation")
    print("="*80)
    
    # Initialize Tölvera
    tv = Tolvera(
        width=800,
        height=600,
        pn=300,
        px="pixels"
    )
    
    # Create behavior agent
    agent = TolveraBehaviorAgent(tv)
    
    # Create LLM synthesizer
    synthesizer = PureLLMSynthesizer()
    
    # Initialize particles
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([
                tv.x * 0.5 + (ti.random() - 0.5) * 200,
                tv.y * 0.5 + (ti.random() - 0.5) * 200
            ])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = 0
            tv.p.field[i].size = 3.0
    
    init_particles()
    tv.s.species.field[0].rgba = [0.8, 0.5, 0.2, 1.0]  # Orange
    
    print("\n📝 Enter particle behavior descriptions (or 'done' to start):")
    print("Examples:")
    print("  - particles bounce off the edges of the screen")
    print("  - particles form a rotating ring pattern")
    print("  - particles accelerate towards the bottom")
    
    behaviors = []
    while True:
        description = input("\nBehavior description: ").strip()
        if description.lower() == 'done' or not description:
            break
        
        weight_str = input("Weight (0.1-2.0, default=1.0): ").strip()
        try:
            weight = float(weight_str) if weight_str else 1.0
            weight = max(0.1, min(2.0, weight))
        except:
            weight = 1.0
        
        behaviors.append((description, weight))
    
    if not behaviors:
        print("\nNo behaviors specified. Using default...")
        behaviors = [("particles drift randomly", 1.0)]
    
    # Generate experts
    for description, weight in behaviors:
        try:
            expert = await agent.add_expert_from_description(
                description,
                synthesizer.synthesizer,
                weight=weight
            )
            print(f"\n✅ Added: {expert.name}")
        except Exception as e:
            print(f"\n❌ Failed: {e}")
    
    @tv.render
    def render():
        agent.update(dt=0.016)
        return tv.px
    
    print("\n🎮 Running your custom behaviors...")
    print("Press 'Esc' to exit")
    tv.run()


async def main():
    """Main menu for demos."""
    
    print("\n" + "="*80)
    print("🌟 PoE SIMPLE DEMO - Raw LLM Behavior Generation")
    print("="*80)
    print("\nThis demo shows the PoE system using ONLY LLM-generated code.")
    print("No templates, no fallbacks - just raw AI output!")
    print("\nSelect a demo:")
    print("1. Basic behaviors (gravity, attraction, repulsion)")
    print("2. Custom behavior (enter your own descriptions)")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-2): ").strip()
    
    if choice == "1":
        await demo_simple_behaviors()
    elif choice == "2":
        await demo_custom_behavior()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    asyncio.run(main())