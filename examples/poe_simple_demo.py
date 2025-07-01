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
from tolvera import Tolvera, run
from src.tolvera.llm.poe_integration import TolveraBehaviorAgent
from src.tolvera.llm.poe_synthesis import PureLLMSynthesizer
import logging
import datetime
import os

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_generated_sketch_to_file(agent, tv_config, filename=None):
    """
    Assembles the generated expert and kernel code into a complete,
    runnable Tölvera sketch and saves it to a file.

    Args:
        agent (TolveraBehaviorAgent): The agent containing the PoE system.
        tv_config (dict): A dictionary of Tölvera's initialization parameters.
        filename (str, optional): The name of the file to save. Defaults to a timestamped name.
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_sketch_{timestamp}.py"

    # Preamble and Tölvera initialization
    header = f'''"""

Dynamically generated Tölvera sketch.
Timestamp: {datetime.datetime.now()}
"""
import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    tv = Tolvera(**{tv_config})

'''

    # Particle and Species Initialization
    init_code = f'''
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = i % {tv_config.get('species', 1)}
            tv.p.field[i].size = 5.0

    init_particles()
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]
    tv.s.species.field[1].rgba = [0.3, 1.0, 0.3, 1.0]
    tv.s.species.field[2].rgba = [0.3, 0.3, 1.0, 1.0]

'''

    # Combine all generated code
    expert_code = "\n\n".join(agent.poe_system.generated_expert_code.values())
    kernel_code = agent.poe_system.generated_kernel_code or ""

    # Main render loop
    render_loop = f'''
    @tv.render
    def render():
        # This calls the dynamically generated kernel
        if callable(apply_all_experts):
             apply_all_experts(tv, 0.016)

        tv.px.clear()
        tv.px.particles(tv.p, tv.s.species)
        return tv.px

    print("Running generated sketch...")
    run(render)

if __name__ == "__main__":
    main()
'''
    # Write everything to the file
    with open(filename, "w") as f:
        f.write(header)
        f.write(init_code)

        # Indent and write the expert and kernel code
        indented_experts = "\n".join(["    " + line for line in expert_code.splitlines()])
        f.write(f"    # --- Generated Expert Functions ---\n{indented_experts}\n\n")

        indented_kernel = "\n".join(["    " + line for line in kernel_code.splitlines()])
        f.write(f"    # --- Generated Integration Kernel ---\n{indented_kernel}\n\n")

        f.write(render_loop)

    print(f"✅ Sketch successfully written to: {os.path.abspath(filename)}")


async def demo_simple_behaviors():
    """Demonstrate basic particle behaviors using pure LLM generation."""
    
    # Define Tölvera configuration
    tv_config = {
        "particles": 100,
        "px": "pixels",
        "gpu": "metal" if sys.platform == "darwin" else "cuda"
    }

    # Initialize Tölvera
    print("📊 Initializing Tölvera...")
    tv = Tolvera(**tv_config)
    
    # Create behavior agent
    agent = TolveraBehaviorAgent(tv)
    
    # Create LLM synthesizer
    print("🤖 Initializing LLM synthesizer...")
    synthesizer = PureLLMSynthesizer(model_name="llama3.2:3b")
    
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
            tv.p.field[i].size = 5.0
    
    init_particles()
    
    # Set species colors
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]  # Red
    tv.s.species.field[1].rgba = [0.3, 1.0, 0.3, 1.0]  # Green
    tv.s.species.field[2].rgba = [0.3, 0.3, 1.0, 1.0]  # Blue
    
    # Test different behavior descriptions
    behaviors = [
        # ("particles fall downward with strong initial gravity that then fades to a gentle trickle", 2),
        # ("particles are attracted to the center of the screen", 0.5),
        ("particles move to the right", 5.0),
        # ("particles repel each other when they get too close", 0.3),
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
        print("   2. The model is available ('ollama pull llama3.2:3b')")
        return
    
    # After the expert generation loop:
    if successful_experts:
        # Save the generated sketch to a file
        save_generated_sketch_to_file(agent, tv_config)

    # Show expert info
    print("\n📋 Active Experts:")
    for info in agent.get_expert_info():
        print(f"   - {info['name']}: weight={info['weight']:.2f}")
    
    # Render loop
    frame_count = 0
    show_info_interval = 120  # Show info every 2 seconds at 60fps
    
    @tv.render
    def render():
        if agent.poe_system._integration_kernel is not None:
            agent.poe_system._integration_kernel(tv, 0.016)
        tv.px.clear()
        tv.px.particles(tv.p, tv.s.species)
        return tv.px
    
    print("\n🎮 Running simulation...")
    print("Press 'Esc' to exit")
    run(render)


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
        if agent.poe_system._integration_kernel is not None:
            agent.poe_system._integration_kernel(tv, 0.016)
        return tv.px
    
    print("\n🎮 Running your custom behaviors...")
    print("Press 'Esc' to exit")
    tv.run()


async def main():
    """Main menu for demos."""
    
    print("\n" + "="*80)
    print("PoE Demo")
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