#!/usr/bin/env python3
"""
This example illustrates:
1. Generating force behaviors from natural language descriptions via LLM.
2. Observing raw LLM output without templates or fallbacks.
3. Multiple behaviors using the PoE system.
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
import subprocess

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_generated_sketch_to_file(agent, tv_config, filename=None):
    sketch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_sketches")
    os.makedirs(sketch_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_sketch_{timestamp}.py"
    
    filepath = os.path.join(sketch_dir, filename)

    header = f'''"""

Dynamically generated Tölvera sketch.
Timestamp: {datetime.datetime.now()}
"""
import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    tv = Tolvera(**kwargs)

'''

    
    has_interactions = any(expert.metadata.get("is_interaction", False) for expert in agent.poe_system.experts)
    
    
    
    if has_interactions:
        
        init_code = f'''
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = i % 2
            tv.p.field[i].size = 5.0
            tv.p.field[i].mass = 1.0

    init_particles()
    
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]
    tv.s.species.field[1].rgba = [0.3, 0.3, 1.0, 1.0]

'''
    else:
        
        init_code = f'''
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = 0
            tv.p.field[i].size = 5.0
            tv.p.field[i].mass = 1.0

    init_particles()
    
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]

'''

    
    
    expert_code = "\n\n".join(agent.poe_system.generated_expert_code.values())
    kernel_code = agent.poe_system.generated_kernel_code or ""

    render_loop = f'''
    @tv.render
    def _():
        tv.px.diffuse(0.99)
        
        tv.p()
        
        apply_all_experts()
        
        tv.px.particles(tv.p, tv.s.species())
        
        return tv.px

if __name__ == "__main__":
    run(main)
'''
    with open(filepath, "w") as f:
        f.write(header)
        f.write(init_code)

        indented_experts = "\n".join(["    " + line for line in expert_code.splitlines()])
        f.write(f"    # ***** Generated Expert Functions *****\n{indented_experts}\n\n")

        indented_kernel = "\n".join(["    " + line for line in kernel_code.splitlines()])
        f.write(f"    # ***** Generated Integration Kernel *****\n{indented_kernel}\n\n")

        f.write(render_loop)

    print(f"Sketch successfully written to: {os.path.abspath(filepath)}")
    return filepath


async def demo_simple_behaviors():
    """Demonstrate basic particle behaviors using pure LLM generation."""
    
    tv_config = {
        "particles": 100,
        "px": "pixels",
        "gpu": "metal" if sys.platform == "darwin" else "cuda"
    }

    print("Initializing Tölvera...")
    tv = Tolvera(**tv_config)
    
    agent = TolveraBehaviorAgent(tv)
    
    print("Initializing LLM synthesizer...")
    synthesizer = PureLLMSynthesizer(model_name="qwen2.5:3b")
    
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([
                ti.random() * tv.x,
                ti.random() * tv.y
            ])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = 0
            tv.p.field[i].size = 5.0
            tv.p.field[i].mass = 1.0
    
    init_particles()
    
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]
    behaviors = [
        ("particles fall downward strongly", 3),
        ("particles are attracted to the center", 10),
        ("particles move to the right", 20),
        ("particles rapidly repel the center of the screen", 5),
    ]
    
    print("\nAvailable behaviors:")
    for i, (desc, weight) in enumerate(behaviors, 1):
        print(f"{i}. {desc} (weight: {weight})")
    
    choice = input("\nWhich behavior would you like to generate? (1-4): ").strip()
    
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(behaviors):
            selected_behavior = behaviors[choice_idx]
        else:
            print("Invalid choice, using default behavior")
            selected_behavior = behaviors[0]
    except ValueError:
        print("Invalid input, using default behavior")
        selected_behavior = behaviors[0]
    
    description, weight = selected_behavior
    
    successful_experts = []
    
    try:
        print(f"\n{'='*60}")
        print(f"Generating: '{description}'")
        print(f"{'='*60}")
        
        expert = await agent.add_expert_from_description(
            description,
            synthesizer.synthesizer,
            weight=weight
        )
        successful_experts.append((expert.name, description))
        
    except Exception as e:
        print(f"Failed to generate expert: {e}")
        print("This is expected with raw LLM output - not all generations succeed!")
    
    if successful_experts:
        print("Successfully generated expert:")
        for name, desc in successful_experts:
            print(f"   - {name}: {desc}")
    
    if not successful_experts:
        print("No experts were successfully generated.")
        return
    
    if successful_experts:
        filename = save_generated_sketch_to_file(agent, tv_config)
        
        print("Active Experts:")
        for info in agent.get_expert_info():
            print(f"   - {info['name']}: weight={info['weight']:.2f}")
        
        print("\n" + "="*60)
        print("Generated sketch saved successfully!")
        print("="*60)
        print("\nWhat would you like to do?")
        print("1. Run the generated sketch")
        print("2. Exit")
        
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            print(f"\nRunning generated sketch: {filename}")
            try:
                subprocess.run([sys.executable, filename], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running sketch: {e}")
            except KeyboardInterrupt:
                print("Sketch execution stopped by user")
        else:
            print("Exiting without running the sketch")


async def demo_species_interactions():    
    print("\n" + "="*80)
    print("SPECIES INTERACTIONS DEMO - Particle-to-Particle Behaviors")
    print("="*80)
    
    tv_config = {
        "particles": 150,
        "px": "pixels",
        "gpu": "metal" if sys.platform == "darwin" else "cuda"
    }
    
    print("Initializing Tölvera...")
    tv = Tolvera(**tv_config)
    
    agent = TolveraBehaviorAgent(tv)
    
    print("Initializing LLM synthesizer...")
    synthesizer = PureLLMSynthesizer(model_name="qwen2.5:3b")
    
    
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = i % 2
            tv.p.field[i].size = 5.0
            tv.p.field[i].mass = 1.0
    
    init_particles()
    
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]
    tv.s.species.field[1].rgba = [0.3, 0.3, 1.0, 1.0]
    
    
    interaction_behaviors = [
        ("species 0 chases species 1 quickly", 20),
        ("particles of the same species attract each other strongly", 30),
        ("species 0 hunts species 1, species 1 flees from species 0 rapidly", 8),
        ("species 0 and species 1 repel each other strongly", 5),
        ("both species flock together within their own groups", 10),
    ]
    
    
    single_behaviors = [
        ("particles fall downward gently", 10),
        ("particles drift slightly to the right", 20),
    ]
    
    print("\nAvailable INTERACTION behaviors:")
    for i, (desc, weight) in enumerate(interaction_behaviors, 1):
        print(f"{i}. {desc} (weight: {weight})")
    
    print("\nAvailable SINGLE-PARTICLE behaviors (can be combined):")
    for i, (desc, weight) in enumerate(single_behaviors, 1):
        print(f"{i+len(interaction_behaviors)}. {desc} (weight: {weight})")
    
    print("\nYou can select multiple behaviors by entering numbers separated by commas.")
    print("Example: 1,6 would select 'species 0 chases species 1' + 'particles fall downward gently'")
    
    choices = input("\nWhich behaviors would you like to generate? ").strip()
    
    selected_behaviors = []
    try:
        if choices:
            choice_indices = [int(c.strip()) - 1 for c in choices.split(",")]
            all_behaviors = interaction_behaviors + single_behaviors
            for idx in choice_indices:
                if 0 <= idx < len(all_behaviors):
                    selected_behaviors.append(all_behaviors[idx])
    except ValueError:
        pass
    
    if not selected_behaviors:
        print("Invalid selection, using default interaction")
        selected_behaviors = [interaction_behaviors[0]]
    
    successful_experts = []
    
    for description, weight in selected_behaviors:
        try:
            print(f"\n{'='*60}")
            print(f"Generating: '{description}'")
            print(f"{'='*60}")
            
            expert = await agent.add_expert_from_description(
                description,
                synthesizer.synthesizer,
                weight=weight
            )
            successful_experts.append((expert.name, description, expert.metadata.get("is_interaction", False)))
            
        except Exception as e:
            print(f"Failed to generate expert: {e}")
    

    has_interactions = any(is_interaction for _, _, is_interaction in successful_experts)
    
    # Re-initialize particles based on whether we have interactions
    if has_interactions:
        print("Re-initializing particles with 2 species for interactions...")
        
    else:
        print("Re-initializing particles with single species...")
        @ti.kernel
        def reinit_particles_single():
            for i in range(tv.pn):
                tv.p.field[i].active = 1.0
                tv.p.field[i].pos = ti.Vector([
                    ti.random() * tv.x,
                    ti.random() * tv.y
                ])
                tv.p.field[i].vel = ti.Vector([0.0, 0.0])
                tv.p.field[i].species = 0
                tv.p.field[i].size = 5.0
                tv.p.field[i].mass = 1.0
        
        reinit_particles_single()
        
        tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]
    
    if successful_experts:
        print(f"Successfully generated {len(successful_experts)} expert(s):")
        for name, desc, is_interaction in successful_experts:
            interaction_type = "interaction" if is_interaction else "single-particle"
            print(f"   - {name} ({interaction_type}): {desc}")
        
    
        has_interactions = any(is_interaction for _, _, is_interaction in successful_experts)
        if has_interactions:
            print("Interaction experts detected! The kernel will include nested loops for particle-particle forces.")
        
        filename = save_generated_sketch_to_file(agent, tv_config)
        
        print("Active Experts:")
        for info in agent.get_expert_info():
            print(f"   - {info['name']}: weight={info['weight']:.2f}")
        
        print("\n" + "="*60)
        print("Generated sketch saved successfully!")
        print("="*60)
        print("\nWhat would you like to do?")
        print("1. Run the generated sketch")
        print("2. Exit")
        
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            print(f"\nRunning generated sketch: {filename}")
            try:
                subprocess.run([sys.executable, filename], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running sketch: {e}")
            except KeyboardInterrupt:
                print("Sketch execution stopped by user")
        else:
            print("Exiting without running the sketch")
    else:
        print("No experts were successfully generated.")


async def demo_custom_behavior():
    """Allow user to input custom behavior descriptions."""
    
    print("\n" + "="*80)
    print("CUSTOM BEHAVIOR DEMO - Your Ideas, LLM Generation")
    print("="*80)
    
    tv = Tolvera(
        width=800,
        height=600,
        pn=300,
        px="pixels"
    )
    
    agent = TolveraBehaviorAgent(tv)
    
    synthesizer = PureLLMSynthesizer()
    
    
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
            tv.p.field[i].mass = 1.0
    
    init_particles()
    tv.s.species.field[0].rgba = [0.8, 0.5, 0.2, 1.0]
    
    print("\nEnter particle behavior descriptions (or 'done' to start):")
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
    successful_experts = []
    for description, weight in behaviors:
        try:
            expert = await agent.add_expert_from_description(
                description,
                synthesizer.synthesizer,
                weight=weight
            )
            print("Added: {expert.name}")
            successful_experts.append(expert)
        except Exception as e:
            print(f"Failed: {e}")
    
    if successful_experts:
        # Save the generated sketch to a file
        tv_config = {
            "width": 800,
            "height": 600,
            "particles": 300,
            "px": "pixels"
        }
        filename = save_generated_sketch_to_file(agent, tv_config)
        
        # Ask user what to do next
        print("\n" + "="*60)
        print("Generated sketch saved successfully!")
        print("="*60)
        print("\nWhat would you like to do?")
        print("1. Run the generated sketch")
        print("2. Exit")
        
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            print(f"\nRunning generated sketch: {filename}")
            try:
                subprocess.run([sys.executable, filename], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running sketch: {e}")
            except KeyboardInterrupt:
                print("Sketch execution stopped by user")
        else:
            print("Exiting without running the sketch")
    else:
        print("No experts were successfully generated.")


async def main():
    """Main menu for demos."""
    
    print("\n" + "="*80)
    print("PoE Demo - Natural Language Particle Behaviors")
    print("="*80)
    print("\nThis demo shows the PoE system using ONLY LLM-generated code.")
    print("No templates, no fallbacks - just raw AI output!")
    
    print("\nSelect a demo:")
    print("1. Basic behaviors (gravity, attraction, repulsion)")
    print("2. Species interactions (chase, flock, repel between species)")
    print("3. Custom behavior (enter your own descriptions)")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-3): ").strip()
    
    if choice == "1":
        await demo_simple_behaviors()
    elif choice == "2":
        await demo_species_interactions()
    elif choice == "3":
        await demo_custom_behavior()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    asyncio.run(main())