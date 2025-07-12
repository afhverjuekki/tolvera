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
from src.tolvera.llm.poe_synthesis import PureLLMSynthesizer, PoEExpertSynthesizer
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
from math import pi

def main(**kwargs):
    tv = Tolvera(**kwargs)

'''

    # Use the dynamic species system to determine species requirements
    species_ids, species_analysis = agent.get_species_requirements()
    
    num_species = len(species_ids)
    
    # Build initialization code with species mapping
    init_code = f"""# Species configuration: {species_ids}
species_map = ti.field(dtype=ti.i32, shape={num_species})
"""
    
    # Add species mapping initialization
    for idx, species_id in enumerate(species_ids):
        init_code += f"species_map[{idx}] = {species_id}\n"
    
    init_code += f"""

@ti.kernel
def init_particles():
    for i in range(tv.pn):
        tv.p.field[i].active = 1.0
        tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
        tv.p.field[i].vel = ti.Vector([0.0, 0.0])
        tv.p.field[i].size = 5.0
        tv.p.field[i].mass = 1.0
        # Assign species using the mapping
        species_index = i % {num_species}
        tv.p.field[i].species = species_map[species_index]

init_particles()
"""
    
    # this can be whatever, but for now just simple colors
    default_colors = [
        [1.0, 0.3, 0.3, 1.0],  # Red
        [0.3, 0.3, 1.0, 1.0],  # Blue
        [0.3, 1.0, 0.3, 1.0],  # Green
        [1.0, 1.0, 0.3, 1.0],  # Yellow
        [1.0, 0.3, 1.0, 1.0],  # Magenta
    ]
    
    for idx, species_id in enumerate(species_ids):
        if idx < len(default_colors):
            color = default_colors[idx]
        else:
            # Generate distinct colors for additional species
            color = [
                0.5 + 0.5 * (idx / max(1, num_species)),
                0.5 + 0.5 * ((idx + 1) % 3 / 3),
                0.5 + 0.5 * ((idx + 2) % 3 / 3),
                1.0
            ]
        init_code += f"tv.s.species.field[{species_id}].rgba = {color}\n"
    
    print(f"Configuring simulation with species {species_ids} based on behavior analysis")

    
    
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
        
        indented_init = "\n".join(["    " + line for line in init_code.splitlines() if line.strip()])
        f.write(indented_init + "\n\n")

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
    synthesizer = PureLLMSynthesizer(model_name="qwen3:4b", enable_decomposition=False)
    
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
        ("particles fall downward strongly", 20),
        ("particles are attracted to the center", 10),
        ("particles move to the right", 20),
        ("particles rapidly repel the center of the screen", 30),
        ("particles drift randomly", 75),
    ]
    
    print("\nAvailable behaviors:")
    for i, (desc, weight) in enumerate(behaviors, 1):
        print(f"{i}. {desc} (weight: {weight})")
    
    choice = input("\nWhich behavior would you like to generate? (1-5): ").strip()
    
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
            weight=weight,
            use_decomposition=False
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
        "gpu": "metal" if sys.platform == "darwin" else "cuda",
        "species": 5  # just a demo
    }
    
    print("Initializing Tölvera...")
    tv = Tolvera(**tv_config)
    
    agent = TolveraBehaviorAgent(tv)
    
    print("Initializing LLM synthesizer with decomposition support...")
    
    synthesizer_engine = PoEExpertSynthesizer(model_name="qwen3:4b", enable_decomposition=True)
    
    # Start with a simple single-species initialization
    @ti.kernel
    def init_particles_default():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = 0
            tv.p.field[i].size = 5.0
            tv.p.field[i].mass = 1.0
    
    init_particles_default()
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]
    
    
    interaction_behaviors = [
        ("species 0 chases species 1 quickly", 20),
        ("particles of the same species attract each other strongly", 20),
        ("species 0 hunts species 1, species 1 flees from species 0 rapidly", 30),
        ("species 0 and species 1 repel each other strongly", 20),
        ("both species flock together within their own groups", 20),
        ("species 2 forms a protective barrier around species 0", 20),
        ("species 0 chases species 1, species 1 chases species 2, species 2 runs from species 0", 30),
        ("species 3 and species 4 orbit around each other", 40),
    ]
    
    single_behaviors = [
        ("particles fall downward gently", 30),
        ("particles drift slightly to the right", 25),
        ("particles drift randomly", 75),
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
                synthesizer_engine,
                weight=weight,
                use_decomposition=True
            )
            successful_experts.append((expert.name, description, expert.metadata.get("is_interaction", False)))
            
        except Exception as e:
            print(f"Failed to generate expert: {e}")
    

    species_ids, species_analysis = agent.get_species_requirements()
    
    print(f"\nSpecies analysis complete:")
    print(f"  - Required species IDs: {species_ids}")
    print(f"  - Species mentioned: {species_analysis['species_mentioned']}")
    print(f"  - Has interactions: {species_analysis['requires_multiple']}")
    
    if len(species_ids) > 1:
        print(f"Re-initializing particles with species {species_ids}...")
        
        # Create a Taichi field to store the species mapping
        species_map = ti.field(dtype=ti.i32, shape=len(species_ids))
        for idx, species_id in enumerate(species_ids):
            species_map[idx] = species_id
        
        @ti.kernel
        def init_particles_with_species(n_species: ti.i32):
            for i in range(tv.pn):
                tv.p.field[i].active = 1.0
                tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
                tv.p.field[i].vel = ti.Vector([0.0, 0.0])
                tv.p.field[i].size = 5.0
                tv.p.field[i].mass = 1.0
                species_index = i % n_species
                tv.p.field[i].species = species_map[species_index]
        
        init_particles_with_species(len(species_ids))
        
        default_colors = [
            [1.0, 0.3, 0.3, 1.0],  # Red
            [0.3, 0.3, 1.0, 1.0],  # Blue
            [0.3, 1.0, 0.3, 1.0],  # Green
            [1.0, 1.0, 0.3, 1.0],  # Yellow
            [1.0, 0.3, 1.0, 1.0],  # Magenta
        ]
        
        for idx, species_id in enumerate(species_ids):
            if idx < len(default_colors):
                tv.s.species.field[species_id].rgba = default_colors[idx]
            else:
                # Generate random colors for additional species
                color = [
                    0.5 + 0.5 * (idx / max(1, len(species_ids))),
                    0.5 + 0.5 * ((idx + 1) % 3 / 3),
                    0.5 + 0.5 * ((idx + 2) % 3 / 3),
                    1.0
                ]
                tv.s.species.field[species_id].rgba = color
    else:
        print("Keeping single species configuration...")
    
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
    
    tv_config = {
        "width": 800,
        "height": 600,
        "particles": 300,
        "px": "pixels",
        "species": 5 
    }
    
    tv = Tolvera(**tv_config)
    
    agent = TolveraBehaviorAgent(tv)
    
    synthesizer_engine = PoEExpertSynthesizer(model_name="qwen3:4b", enable_decomposition=True)
    
    @ti.kernel
    def init_particles_default():
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
    
    init_particles_default()
    tv.s.species.field[0].rgba = [0.8, 0.5, 0.2, 1.0]
    
    print("\nEnter particle behavior descriptions (or 'done' to start):")
    print("Examples:")
    print("  - particles bounce off the edges of the screen")
    print("  - particles form a rotating ring pattern")
    print("  - particles accelerate towards the bottom")
    print("\nMulti-species examples:")
    print("  - species 0 chases species 1")
    print("  - species 2 protects species 0 from species 1")
    print("  - all species repel each other")
    
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
                synthesizer_engine,
                weight=weight,
                use_decomposition=True
            )
            print(f"Added: {expert.name}")
            successful_experts.append(expert)
        except Exception as e:
            print(f"Failed: {e}")
    
    if successful_experts:
        species_ids, species_analysis = agent.get_species_requirements()
        
        if len(species_ids) > 1:
            print(f"\nRe-initializing particles with species {species_ids}...")
            
            # Create a Taichi field to store the species mapping
            species_map_custom = ti.field(dtype=ti.i32, shape=len(species_ids))
            for idx, species_id in enumerate(species_ids):
                species_map_custom[idx] = species_id
            
            # Define a static kernel that uses the species mapping
            @ti.kernel
            def init_particles_custom_species(n_species: ti.i32):
                for i in range(tv.pn):
                    tv.p.field[i].active = 1.0
                    tv.p.field[i].pos = ti.Vector([
                        tv.x * 0.5 + (ti.random() - 0.5) * 200,
                        tv.y * 0.5 + (ti.random() - 0.5) * 200
                    ])
                    tv.p.field[i].vel = ti.Vector([0.0, 0.0])
                    tv.p.field[i].size = 3.0
                    tv.p.field[i].mass = 1.0
                    # Assign species using the mapping
                    species_index = i % n_species
                    tv.p.field[i].species = species_map_custom[species_index]
            
            init_particles_custom_species(len(species_ids))
            
            default_colors = [
                [1.0, 0.3, 0.3, 1.0],  # Red
                [0.3, 0.3, 1.0, 1.0],  # Blue
                [0.3, 1.0, 0.3, 1.0],  # Green
                [1.0, 1.0, 0.3, 1.0],  # Yellow
                [1.0, 0.3, 1.0, 1.0],  # Magenta
            ]
            
            for idx, species_id in enumerate(species_ids):
                if idx < len(default_colors):
                    tv.s.species.field[species_id].rgba = default_colors[idx]
                else:
                    # Generate random colors for additional species
                    color = [
                        0.5 + 0.5 * (idx / max(1, len(species_ids))),
                        0.5 + 0.5 * ((idx + 1) % 3 / 3),
                        0.5 + 0.5 * ((idx + 2) % 3 / 3),
                        1.0
                    ]
                    tv.s.species.field[species_id].rgba = color
        
        # Save the generated sketch to a file
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