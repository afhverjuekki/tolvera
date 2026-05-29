#!/usr/bin/env python3
import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from tolvera import Tolvera
from tolvera.llm import BehaviorOrchestrator
from tolvera.llm.debug.tracing import get_collector
from tolvera.llm.debug.console_tracer import enable_console_tracing
from tolvera.llm.debug.trace_html_report import generate_html_report


def configure_debug_mode(enable_debug: bool = False, enable_prompt_debug: bool = False):
    """Configure debug logging based on command line arguments."""
    if enable_debug or enable_prompt_debug:
        # Set up detailed logging
        log_level = logging.DEBUG if enable_prompt_debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S',
            force=True  # Override any existing configuration
        )
        
        if enable_prompt_debug:
            # Enable DEBUG level for prompt-related modules
            logging.getLogger('tolvera.llm.core.prompt_loader').setLevel(logging.DEBUG)
            logging.getLogger('tolvera.llm.core.synthesizer').setLevel(logging.DEBUG)
            logging.getLogger('tolvera.llm.core.decomposer').setLevel(logging.DEBUG)
            logging.getLogger('tolvera.llm.core.sketch_refiner').setLevel(logging.DEBUG)
            print("\n🔍 PROMPT DEBUG MODE ENABLED - Full prompts will be logged")
            print("=" * 60)
        
        if enable_debug:
            print("\n🐛 DEBUG MODE ENABLED - Detailed logging active")
            print("=" * 60)
    else:
        # Normal logging configuration
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )


def save_trace_with_report(collector, trace, name_prefix):
    if not collector.enabled or not trace:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON trace
    json_trace = collector.export_trace(trace.id, format="json")
    # Get project root (5 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent.parent
    json_path = project_root / f"examples/generated_sketches/traces/{name_prefix}_{timestamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        f.write(json_trace)
    print(f"\nJSON trace saved to: {json_path}")
    
    # Generate HTML report
    try:
        html_path = generate_html_report(json_path)
        print(f"📊 Interactive report: {html_path}")
    except Exception as e:
        print(f"⚠️  HTML report generation failed: {e}")
    
    # Save Mermaid diagram
    mermaid_trace = collector.export_trace(trace.id, format="mermaid")
    mermaid_path = project_root / f"examples/generated_sketches/traces/{name_prefix}_{timestamp}.md"
    with open(mermaid_path, "w") as f:
        f.write(mermaid_trace)
    print(f"Mermaid diagram saved to: {mermaid_path}")


async def demo_basic_behaviors():
    print("\n" + "="*60)
    print("DEMO 1: Basic Single-Particle Behaviors")
    print("="*60)
    
    enable_console_tracing(colored=True)
    collector = get_collector()
    collector.enabled = True
    collector.capture_llm_content = True
    main_trace = collector.start_trace("Basic Behaviors Demo", "demo")
    
    tv = Tolvera(width=1920, height=1080, pn=500, sn=2)
    orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    # Combine behaviors into a single description to avoid state conflicts
    combined_behavior = "Particles fall down with gravity."
    
    print(f"\nAdding behavior: {combined_behavior}")
    try:
        result = await orchestrator.add_behavior(combined_behavior, weight=1.0)
        print(f"✓ Behavior added successfully")
        print(f"  - Experts added: {result['experts_added']}")
        print(f"  - Pattern detected: {result.get('pattern_type', 'particle_system')}")
        print(f"  - States created: {result.get('states_created', 0)}")
    except Exception as e:
        print(f"✗ Failed to add behavior: {e}")
    
    experts = orchestrator.get_expert_info()
    print(f"\nTotal experts: {len(experts)}")
    for expert in experts:
        print(f"  - {expert['name']}: {expert['description']} (weight={expert['weight']})")
    
    _, sketch_path = await orchestrator.generate_sketch_async(
        description="Basic particle physics demo",
        filename="demo_basic_behaviors",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    
    save_trace_with_report(collector, main_trace, "demo_basic")
    
    return orchestrator, sketch_path


async def demo_complex_behaviors():
    print("\n" + "="*60)
    print("DEMO 2: Complex Behavior with Decomposition")
    print("="*60)
    
    enable_console_tracing(colored=True)
    collector = get_collector()
    collector.enabled = True
    collector.capture_llm_content = True
    main_trace = collector.start_trace("Complex Behaviors Demo", "demo")
    
    tv = Tolvera(width=1920, height=1080, pn=1000, sn=5)
    orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    complex_description = """
    Two species, one maroon and one teal, are competing for food (green particles).  The blue one is quicker than the teal and runs away with the food while
    the slower blue one always chases the teal one.  The food (green particles) disappears when consumed by either species.  Even if no food is present, the blue species will continue to move with the teal species chasing them.
    The food regenerates over time in random places.
    """
    # complex_description = """
    # Species one (green) moves left continuously.  Species two (orange) moves down continously.
    # Speices three (turquoise) moves to the upper-right continously.  Species four (pink) moves to the upper-left continuously.
    # I want there to be 3000 pixels overall.
    # """
    
    print(f"\nComplex behavior: {complex_description.strip()}")
    
    try:
        # Use regular add_behavior which will detect ecosystem pattern and handle appropriately
        result = await orchestrator.add_behavior(complex_description, weight=1.0)
        
        print(f"\n✓ Complex behavior synthesized successfully")
        print(f"  - Pattern detected: {result.get('pattern_type', 'unknown')}")
        print(f"  - Pattern confidence: {result.get('pattern_confidence', 0):.2f}")
        print(f"  - Experts added: {result.get('experts_added', 0)}")
        print(f"  - States created: {result.get('states_created', 0)}")
        
        if orchestrator.current_species_config:
            print("\n🐟 Detected Species Configuration:")
            if orchestrator.current_species_config.species_names:
                for sid, name in orchestrator.current_species_config.species_names.items():
                    print(f"  • Species {sid}: {name}")
        
    except Exception as e:
        print(f"✗ Failed to add complex behavior: {e}")
    
    _, sketch_path = await orchestrator.generate_sketch_async(
        description="Ecosystem simulation with predator-prey dynamics",
        filename="demo_complex_ecosystem",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    
    save_trace_with_report(collector, main_trace, "demo_complex")
    
    return orchestrator, sketch_path


async def demo_drawing_behaviors():
    print("\n" + "="*60)
    print("DEMO 3: Drawing and Visual Effects")
    print("="*60)
    
    enable_console_tracing(colored=True)
    collector = get_collector()
    collector.enabled = True
    collector.capture_llm_content = True
    main_trace = collector.start_trace("Drawing Behaviors Demo", "demo")
    
    tv = Tolvera(width=1920, height=1080, pn=200, sn=3)
    orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    # Combine movement behaviors to avoid state conflicts
    movement_behavior = "particles move in circular orbits and species 0 and species 1 attract each other"
    
    print(f"\nAdding movement behavior: {movement_behavior}")
    try:
        result = await orchestrator.add_behavior(movement_behavior, 1.0)
        print(f"✓ Movement behavior added: {result['experts_added']} experts")
    except Exception as e:
        print(f"✗ Failed to add movement behavior: {e}")
    
    drawing_behaviors = [
        ("draw trails behind fast-moving particles", 1.0, "pre"),
        ("particles glow brighter when near others", 0.8, "post"),
        ("draw connecting lines between nearby particles of the same species", 0.5, "post"),
    ]
    
    print("\nAdding drawing behaviors:")
    for description, weight, order in drawing_behaviors:
        try:
            result = await orchestrator.add_drawing_behavior(description, weight, order)
            print(f"✓ {description}: {result['experts_added']} drawing experts added")
        except Exception as e:
            print(f"✗ {description}: Failed - {e}")
    
    experts = orchestrator.get_expert_info()
    print(f"\nTotal experts: {len(experts)}")
    
    movement_experts = [e for e in experts if e['expert_type'] in ['single', 'interaction']]
    drawing_experts = [e for e in experts if 'drawing' in e['expert_type']]
    
    print(f"\nMovement experts: {len(movement_experts)}")
    for expert in movement_experts:
        print(f"  - {expert['name']}: {expert['description']}")
    
    print(f"\nDrawing experts: {len(drawing_experts)}")
    for expert in drawing_experts:
        print(f"  - {expert['name']}: {expert['description']}")
    
    _, sketch_path = await orchestrator.generate_sketch_async(
        description="Visual effects demo with trails and glows",
        filename="demo_drawing_effects",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_drawing")
    
    return orchestrator, sketch_path


async def demo_species_interactions():
    print("\n" + "="*60)
    print("DEMO 4: Multi-Species Interactions (Dynamic Species Detection)")
    print("="*60)
    
    enable_console_tracing(colored=True)
    collector = get_collector()
    collector.enabled = True
    collector.capture_llm_content = True
    main_trace = collector.start_trace("Species Interactions Demo", "demo")
    
    tv = Tolvera(width=1920, height=1080, pn=800, sn=6)
    orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    interactions = [
        "red predators hunt green prey that try to escape",
    ]
    
    print("\nAdding interaction behaviors:")
    for description in interactions:
        try:
            result = await orchestrator.add_behavior(description, 1.0)
            print(f"✓ {description}: {result['experts_added']} experts added")
            if 'species_count' in result:
                print(f"  └─ Detected species: {result['species_count']}")
        except Exception as e:
            print(f"✗ {description}: Failed - {e}")
    
    if orchestrator.current_species_config:
        print("\n📊 Dynamic Species Configuration:")
        print(f"  - Total species: {len(orchestrator.current_species_config.species_ids)}")
        if orchestrator.current_species_config.species_names:
            print("  - Named species:")
            for sid, name in orchestrator.current_species_config.species_names.items():
                print(f"    • Species {sid}: {name}")
        if orchestrator.current_species_config.interaction_pairs:
            print("  - Interaction pairs:")
            for s1, s2 in orchestrator.current_species_config.interaction_pairs:
                n1 = orchestrator.current_species_config.species_names.get(s1, f"Species {s1}")
                n2 = orchestrator.current_species_config.species_names.get(s2, f"Species {s2}")
                print(f"    • {n1} ↔ {n2}")
    
    _, sketch_path = await orchestrator.generate_sketch_async(
        description="Multi-species ecosystem with dynamic species detection",
        filename="demo_species_interactions",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_species")
    
    return orchestrator, sketch_path


async def demo_species_detection():
    print("\n" + "="*60)
    print("DEMO 5: Automatic Species Detection")
    print("="*60)
    
    enable_console_tracing(colored=True)
    collector = get_collector()
    collector.enabled = True
    collector.capture_llm_content = True
    main_trace = collector.start_trace("Species Detection Demo", "demo")
    
    tv = Tolvera(width=1920, height=1080, pn=600, sn=8)
    orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    test_behaviors = [
        ("particles move randomly", "Single species behavior"),
        ("red particles chase blue particles", "Two species by color"),
        ("predators hunt prey while plants grow slowly", "Three species by role"),
        ("create 5 different colored particle groups that interact", "Five species by description"),
    ]
    
    print("\nTesting species detection:")
    for description, expected in test_behaviors:
        print(f"\n📝 Description: '{description}'")
        print(f"   Expected: {expected}")
        
        test_agent = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
        
        try:
            result = await test_agent.add_behavior(description, 1.0)
            
            if test_orchestrator.current_species_config:
                detected = len(test_orchestrator.current_species_config.species_ids)
                print(f"   ✓ Detected: {detected} species")
                
                if test_orchestrator.current_species_config.species_names:
                    print("     Species names:", list(test_orchestrator.current_species_config.species_names.values()))
                
                if test_orchestrator.current_species_config.colors:
                    print(f"     Colors assigned: {len(test_orchestrator.current_species_config.colors)}")
            else:
                print("   ⚠️  No species configuration detected")
                
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\n" + "-"*60)
    print("Adding complex multi-species behavior to main agent:")
    
    complex = """
    Create an ocean ecosystem with orange clownfish hiding in anemones, 
    purple sharks hunting smaller fish, yellow tangs grazing on algae, 
    and green sea turtles swimming peacefully
    """
    
    result = await orchestrator.add_behavior(complex, 1.0)
    print(f"\n✓ Added complex behavior with {result['species_count']} species")
    
    _, sketch_path = await orchestrator.generate_sketch_async(
        description="Species detection demonstration",
        filename="demo_species_detection",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_species_detection")
    
    return orchestrator, sketch_path


async def demo_state_generation():
    print("\n" + "="*60)
    print("DEMO 6: Automatic State Generation")
    print("="*60)
    
    enable_console_tracing(colored=True)
    collector = get_collector()
    collector.enabled = True
    collector.capture_llm_content = True
    main_trace = collector.start_trace("State Generation Demo", "demo")
    
    tv = Tolvera(width=1920, height=1080, pn=1000, sn=4)
    orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    state_behaviors = [
        {
            "description": "particles form a cellular automaton where each cell lives or dies based on neighbor count",
            "expected_type": "Game of Life",
            "expected_states": ["is_alive", "neighbor_count", "next_state", "grid_x", "grid_y"]
        },
        {
            "description": "particles follow day/night cycles with different behaviors during each phase",
            "expected_type": "Temporal Cycles",
            "expected_states": ["day_phase", "frame_count"]
        }
    ]
    
    print("\n🔬 Testing state generation for complex behaviors that require custom states:")
    print("\nEach behavior will be analyzed to determine what states it needs,")
    print("then those states will be automatically generated.\n")
    
    for test_idx, test in enumerate(state_behaviors):
        print(f"\n{test_idx + 1}. {test['expected_type']}")
        print("-" * 50)
        print(f"📝 Behavior: {test['description']}")
        print(f"🎯 Expected states: {', '.join(test['expected_states'])}")
        
        try:
            test_agent = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
            
            result = await test_agent.add_behavior(test['description'], weight=1.0)
            
            available_states = test_agent.state_manager.get_available_states()
            all_states = []
            state_details = {}
            
            for category in ['global', 'particle', 'species']:
                if category in available_states and available_states[category]:
                    all_states.extend(available_states[category])
                    state_details[category] = available_states[category]
            
            print(f"\n✅ Behavior synthesized successfully!")
            print(f"   Experts added: {result['experts_added']}")
            print(f"   States created: {result.get('states_created', 0)}")
            
            if state_details:
                print("\n📊 Generated States:")
                for category, states in state_details.items():
                    if states:
                        print(f"   {category.capitalize()}: {', '.join(states)}")
            else:
                print("\n⚠️  No states were generated (might not need any)")
            
            missing_states = []
            for expected in test['expected_states']:
                found = False
                for category in ['global', 'particle', 'species']:
                    if expected in available_states.get(category, []):
                        found = True
                        break
                if not found:
                    missing_states.append(expected)
            
            if missing_states:
                print(f"\n⚠️  Missing expected states: {', '.join(missing_states)}")
            else:
                print("\n✨ All expected states were generated!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n\n" + "="*50)
    print("Comparison: Simple behavior without state requirements")
    print("-" * 50)
    
    simple_agent = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    simple_behavior = "particles drift randomly"
    print(f"📝 Behavior: {simple_behavior}")
    
    try:
        result = await simple_agent.add_behavior(simple_behavior, weight=1.0)
        available_states = simple_agent.state_manager.get_available_states()
        
        has_states = any(available_states.get(cat, []) for cat in ['global', 'particle', 'species'])
        
        print(f"✅ Behavior added successfully!")
        print(f"   States generated: {'Yes' if has_states else 'None (as expected for simple behaviors)'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n\n" + "="*50)
    print("Generating example sketch with Game of Life states...")
    
    gol_orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    await gol_orchestrator.add_behavior(
        "particles form a cellular automaton where each cell lives or dies based on neighbor count",
        weight=1.0
    )
    
    _, sketch_path = gol_orchestrator.generate_sketch(
        description="Game of Life with automatic state generation",
        filename="demo_state_generation_gol",
        use_timestamp=True
    )
    print(f"\n📄 Sketch saved to: {sketch_path}")
    print("   Check the sketch to see how states are initialized and used!")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_state_generation")
    
    return gol_orchestrator, sketch_path


async def demo_artificial_life_patterns():
    print("\n" + "="*60)
    print("DEMO 7: Artificial Life Patterns")
    print("="*60)
    
    enable_console_tracing(colored=True)
    collector = get_collector()
    collector.enabled = True
    collector.capture_llm_content = True
    main_trace = collector.start_trace("Artificial Life Patterns Demo", "demo")
    
    tv = Tolvera(width=1920, height=1080, pn=1000, sn=4)
    orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    alife_patterns = [
        {
            "description": "particles form groups where each one lives or dies based on how many neighbors it has",
            "expected_pattern": "Cellular Automaton (like Game of Life)",
            "expected_components": ["grid_state_update", "neighbor counting", "synchronous updates"]
        }
    ]
    
    print("\n🧬 Testing artificial life pattern recognition and synthesis:")
    print("\nEach description represents a classic a-life pattern without naming it.")
    print("The system should recognize and synthesize these patterns correctly.\n")
    
    for idx, pattern in enumerate(alife_patterns):
        print(f"\n{idx + 1}. Testing: \"{pattern['description']}\"")
        print("-" * 60)
        print(f"   Expected pattern: {pattern['expected_pattern']}")
        
        try:
            test_agent = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
            
            print("   🔍 Using add_behavior to detect and synthesize pattern...")
            result = await test_agent.add_behavior(
                pattern['description'],
                weight=1.0
            )
            
            print("\n   ✅ Successfully synthesized!")
            print(f"   Pattern detected: {result.get('pattern_type', 'unknown')}")
            print(f"   Pattern confidence: {result.get('pattern_confidence', 0):.2f}")
            print(f"   Total experts created: {result.get('experts_added', 0)}")
            
            if result.get('states_created', 0) > 0:
                print(f"\n   📊 States automatically generated: {result['states_created']}")
                available_states = test_agent.state_manager.get_available_states()
                for category in ['global', 'particle', 'species']:
                    if category in available_states and available_states[category]:
                        print(f"      {category.capitalize()}: {', '.join(available_states[category])}")
            
            # Check for helper functions
            if test_agent.synthesized_helpers:
                print(f"\n   🔧 Helper functions generated: {list(test_agent.synthesized_helpers.keys())}")
            
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
    
    print("\n\n" + "="*60)
    print("Generating combined a-life sketch...")
    print("-" * 60)
    
    combined_orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    # Single combined behavior to avoid state conflicts
    combined_pattern = """
    particles form groups where each one lives or dies based on how many neighbors it has,
    while some particles leave trails that others follow to find food sources,
    and predator particles hunt prey particles in coordinated groups
    """
    
    print("\nAdding combined a-life behavior:")
    try:
        result = await combined_orchestrator.add_behavior(combined_pattern, weight=1.0)
        print(f"  ✓ Added combined behavior")
        print(f"    - Pattern: {result.get('pattern_type', 'unknown')}")
        print(f"    - Experts: {result.get('experts_added', 0)}")
        print(f"    - States: {result.get('states_created', 0)}")
    except Exception as e:
        print(f"  ✗ Failed to add combined behavior: {e}")
    
    _, sketch_path = combined_orchestrator.generate_sketch(
        description="Artificial life patterns demonstration",
        filename="demo_alife_patterns",
        use_timestamp=True
    )
    print(f"\n📄 Combined sketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_alife")
    
    return combined_orchestrator, sketch_path


async def demo_custom_behavior():
    print("\n" + "="*60)
    print("DEMO 8: Custom Behavior Input")
    print("="*60)
    
    enable_console_tracing(colored=True)
    collector = get_collector()
    collector.enabled = True
    collector.capture_llm_content = True
    main_trace = collector.start_trace("Custom Behavior Demo", "demo")
    
    tv = Tolvera(width=1920, height=1080, pn=500, sn=3)
    orchestrator = BehaviorOrchestrator(tv, model_name=os.getenv("DEFAULT_MODEL", "us.anthropic.claude-opus-4-8"))
    
    print("\nEnter custom particle behaviors (or 'done' to finish):")
    print("\nExamples:")
    print("  - particles bounce off walls")
    print("  - species 0 forms a rotating ring")
    print("  - particles leave rainbow trails")
    print("  - create a fireworks effect when particles collide")
    
    behaviors = []
    while True:
        description = await asyncio.to_thread(input, "\nBehavior: ")
        description = description.strip()
        if description.lower() == 'done' or not description:
            break
        behaviors.append(description)
    
    if not behaviors:
        behaviors = ["particles drift randomly", "particles sparkle occasionally"]
        print("\nUsing default behaviors:", behaviors)
    
    print("\nAdding behaviors:")
    for description in behaviors:
        try:
            if any(kw in description.lower() for kw in ['draw', 'trail', 'sparkle', 'glow', 'line']):
                await orchestrator.add_drawing_behavior(description, 1.0)
            else:
                await orchestrator.add_behavior(description, 1.0)
            print(f"✓ {description}: Success")
        except Exception as e:
            print(f"✗ {description}: Failed - {e}")
    
    _, sketch_path = await orchestrator.generate_sketch_async(
        description="Custom behavior demonstration",
        filename="demo_custom_behaviors",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_custom")
    
    return orchestrator, sketch_path


async def main():
    print("\n" + "="*80)
    print("TÖLVERA LLM UNIFIED DEMO")
    print("Natural Language Particle Behavior Synthesis")
    print("="*80)
    
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  No GEMINI_API_KEY found!")
        print("\nTo use this demo:")
        print("1. Copy .env.example to .env")
        print("2. Add your Gemini API key")
        print("3. Get a key at: https://makersuite.google.com/app/apikey")
        print("\nAlternatively, set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        return
    
    # Get project root (5 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent.parent
    (project_root / "examples/generated_sketches").mkdir(parents=True, exist_ok=True)
    (project_root / "examples/generated_sketches/traces").mkdir(parents=True, exist_ok=True)
    
    while True:
        print("\n" + "="*80)
        print("SELECT A DEMO:")
        print("="*80)
        print("1. Basic Behaviors - Simple particle physics")
        print("2. Complex Behaviors - Decomposition and embellishment")
        print("3. Drawing Effects - Visual trails and glows")
        print("4. Species Interactions - Multi-species ecosystems")
        print("5. Species Detection - Automatic species analysis")
        print("6. State Generation - Automatic state creation for complex behaviors")
        print("7. Artificial Life Patterns - Classic a-life behaviors without naming them")
        print("8. Custom Input - Enter your own behaviors")
        print("9. Run All Demos - Execute all demonstrations")
        print("0. Exit")
        
        choice = await asyncio.to_thread(input, "\nChoice (0-9): ")
        choice = choice.strip()
        
        if choice == "0":
            print("\nExiting demo. Goodbye!")
            break
        elif choice == "1":
            await demo_basic_behaviors()
        elif choice == "2":
            await demo_complex_behaviors()
        elif choice == "3":
            await demo_drawing_behaviors()
        elif choice == "4":
            await demo_species_interactions()
        elif choice == "5":
            await demo_species_detection()
        elif choice == "6":
            await demo_state_generation()
        elif choice == "7":
            await demo_artificial_life_patterns()
        elif choice == "8":
            await demo_custom_behavior()
        elif choice == "9":
            print("\nRunning all demos...")
            await demo_basic_behaviors()
            await demo_complex_behaviors()
            await demo_drawing_behaviors()
            await demo_species_interactions()
            await demo_species_detection()
            await demo_state_generation()
            await demo_artificial_life_patterns()
            print("\n✅ All demos completed!")
            print(f"\nGenerated sketches are in: {project_root}/examples/generated_sketches/")
            print(f"Trace files are in: {project_root}/examples/generated_sketches/traces/")
        else:
            print("\nInvalid choice. Please try again.")
        
        if choice in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            print("\n" + "-"*60)
            print("Options:")
            print("1. Run another demo")
            print("2. Exit")
            
            next_action = await asyncio.to_thread(input, "\nChoice (1-2): ")
            next_action = next_action.strip()
            if next_action == "2":
                print("\nExiting demo. Goodbye!")
                break


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tölvera LLM Demo')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--debug-prompts', action='store_true', help='Enable detailed prompt debugging')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode - log prompts without LLM calls')
    args = parser.parse_args()
    
    # Configure debug mode
    configure_debug_mode(enable_debug=args.debug, enable_prompt_debug=args.debug_prompts)
    
    # Set dry run environment variable if requested
    if args.dry_run:
        os.environ['LLM_DRY_RUN'] = '1'
        print("\n🏃 DRY RUN MODE - Prompts will be logged but no LLM calls will be made")
        print("=" * 60)
    
    # Load environment variables
    # Get project root (5 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from: {env_path}")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()