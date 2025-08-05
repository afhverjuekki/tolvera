#!/usr/bin/env python3
import asyncio
import os
from datetime import datetime
from pathlib import Path
from tolvera import Tolvera
from tolvera.llm import BehaviorAgent
from tolvera.llm.debug.tracing import get_collector
from tolvera.llm.debug.console_tracer import enable_console_tracing
from tolvera.llm.debug.trace_html_report import generate_html_report


def save_trace_with_report(collector, trace, name_prefix):
    if not collector.enabled or not trace:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON trace
    json_trace = collector.export_trace(trace.id, format="json")
    json_path = f"examples/generated_sketches/traces/{name_prefix}_{timestamp}.json"
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
    mermaid_path = f"examples/generated_sketches/traces/{name_prefix}_{timestamp}.md"
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
    agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
    behaviors = [
        ("particles fall downward with gravity", 1.0),
        ("particles drift randomly", 0.5),
    ]
    
    print("\nAdding behaviors:")
    for description, weight in behaviors:
        try:
            result = await agent.add_behavior(description, weight)
            print(f"✓ {description}: {result['experts_added']} experts added")
        except Exception as e:
            print(f"✗ {description}: Failed - {e}")
    
    experts = agent.get_expert_info()
    print(f"\nTotal experts: {len(experts)}")
    for expert in experts:
        print(f"  - {expert['name']}: {expert['description']} (weight={expert['weight']})")
    
    _, sketch_path = agent.generate_sketch(
        description="Basic particle physics demo",
        filename="demo_basic_behaviors",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    
    save_trace_with_report(collector, main_trace, "demo_basic")
    
    return agent, sketch_path


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
    agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
    complex_description = """
    Create an ecosystem where small green fish school together and swim in 
    coordinated patterns, while larger red predator fish hunt them. The small 
    fish should avoid predators and move as a group. Blue scavenger fish 
    clean up after the predators.
    """
    
    print(f"\nComplex behavior: {complex_description.strip()}")
    
    try:
        result = await agent.add_complex_behavior(
            complex_description,
            weight=1.0,
            decompose=True,
            auto_embellish=True
        )
        
        print(f"\nInterpretation: {result['interpretation']}")
        print(f"Components: {len(result['components'])}")
        for i, comp in enumerate(result['components']):
            print(f"  {i+1}. {comp['description']}")
        
        if result['embellishments']:
            print(f"\nEmbellishments: {len(result['embellishments'])}")
            for emb in result['embellishments']:
                print(f"  + {emb['description']}")
        
        print(f"\nTotal experts created: {result['total_experts']}")
        
        if agent.current_species_config:
            print("\n🐟 Detected Species Configuration:")
            if agent.current_species_config.species_names:
                for sid, name in agent.current_species_config.species_names.items():
                    print(f"  • Species {sid}: {name}")
        
    except Exception as e:
        print(f"✗ Failed to add complex behavior: {e}")
    
    _, sketch_path = agent.generate_sketch(
        description="Ecosystem simulation with predator-prey dynamics",
        filename="demo_complex_ecosystem",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    
    save_trace_with_report(collector, main_trace, "demo_complex")
    
    return agent, sketch_path


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
    agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
    await agent.add_behavior("particles move in circular orbits", 1.0)
    await agent.add_behavior("species 0 and species 1 attract each other", 0.5)
    
    drawing_behaviors = [
        ("draw trails behind fast-moving particles", 1.0, "pre"),
        ("particles glow brighter when near others", 0.8, "post"),
        ("draw connecting lines between nearby particles of the same species", 0.5, "post"),
    ]
    
    print("\nAdding drawing behaviors:")
    for description, weight, order in drawing_behaviors:
        try:
            result = await agent.add_drawing_behavior(description, weight, order)
            print(f"✓ {description}: {result['experts_added']} drawing experts added")
        except Exception as e:
            print(f"✗ {description}: Failed - {e}")
    
    experts = agent.get_expert_info()
    print(f"\nTotal experts: {len(experts)}")
    
    movement_experts = [e for e in experts if e['expert_type'] in ['single', 'interaction']]
    drawing_experts = [e for e in experts if 'drawing' in e['expert_type']]
    
    print(f"\nMovement experts: {len(movement_experts)}")
    for expert in movement_experts:
        print(f"  - {expert['name']}: {expert['description']}")
    
    print(f"\nDrawing experts: {len(drawing_experts)}")
    for expert in drawing_experts:
        print(f"  - {expert['name']}: {expert['description']}")
    
    _, sketch_path = agent.generate_sketch(
        description="Visual effects demo with trails and glows",
        filename="demo_drawing_effects",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_drawing")
    
    return agent, sketch_path


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
    agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
    interactions = [
        "red predators hunt green prey that try to escape",
    ]
    
    print("\nAdding interaction behaviors:")
    for description in interactions:
        try:
            result = await agent.add_behavior(description, 1.0)
            print(f"✓ {description}: {result['experts_added']} experts added")
            if 'species_count' in result:
                print(f"  └─ Detected species: {result['species_count']}")
        except Exception as e:
            print(f"✗ {description}: Failed - {e}")
    
    if agent.current_species_config:
        print("\n📊 Dynamic Species Configuration:")
        print(f"  - Total species: {len(agent.current_species_config.species_ids)}")
        if agent.current_species_config.species_names:
            print("  - Named species:")
            for sid, name in agent.current_species_config.species_names.items():
                print(f"    • Species {sid}: {name}")
        if agent.current_species_config.interaction_pairs:
            print("  - Interaction pairs:")
            for s1, s2 in agent.current_species_config.interaction_pairs:
                n1 = agent.current_species_config.species_names.get(s1, f"Species {s1}")
                n2 = agent.current_species_config.species_names.get(s2, f"Species {s2}")
                print(f"    • {n1} ↔ {n2}")
    
    _, sketch_path = agent.generate_sketch(
        description="Multi-species ecosystem with dynamic species detection",
        filename="demo_species_interactions",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_species")
    
    return agent, sketch_path


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
    agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
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
        
        test_agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
        
        try:
            result = await test_agent.add_behavior(description, 1.0)
            
            if test_agent.current_species_config:
                detected = len(test_agent.current_species_config.species_ids)
                print(f"   ✓ Detected: {detected} species")
                
                if test_agent.current_species_config.species_names:
                    print("     Species names:", list(test_agent.current_species_config.species_names.values()))
                
                if test_agent.current_species_config.colors:
                    print(f"     Colors assigned: {len(test_agent.current_species_config.colors)}")
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
    
    result = await agent.add_behavior(complex, 1.0)
    print(f"\n✓ Added complex behavior with {result['species_count']} species")
    
    _, sketch_path = agent.generate_sketch(
        description="Species detection demonstration",
        filename="demo_species_detection",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_species_detection")
    
    return agent, sketch_path


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
    agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
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
            test_agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
            
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
    
    simple_agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
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
    
    gol_agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
    await gol_agent.add_behavior(
        "particles form a cellular automaton where each cell lives or dies based on neighbor count",
        weight=1.0
    )
    
    _, sketch_path = gol_agent.generate_sketch(
        description="Game of Life with automatic state generation",
        filename="demo_state_generation_gol",
        use_timestamp=True
    )
    print(f"\n📄 Sketch saved to: {sketch_path}")
    print("   Check the sketch to see how states are initialized and used!")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_state_generation")
    
    return gol_agent, sketch_path


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
    agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
    alife_patterns = [
        {
            "description": "particles form groups where each one lives or dies based on how many neighbors it has",
            "expected_pattern": "Cellular Automaton (like Game of Life)",
            "expected_components": ["grid_state_update", "neighbor counting", "synchronous updates"]
        }
    ]
    
    print("\n🧬 Testing artificial life pattern recognition and synthesis:")
    print("\nEach description represents a classic a-life pattern without naming it.")
    print("The system should recognize and decompose these into appropriate components.\n")
    
    for idx, pattern in enumerate(alife_patterns):
        print(f"\n{idx + 1}. Testing: \"{pattern['description']}\"")
        print("-" * 60)
        print(f"   Expected pattern: {pattern['expected_pattern']}")
        
        try:
            test_agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
            
            print("   🔍 Calling add_complex_behavior with decompose=True...")
            result = await test_agent.add_complex_behavior(
                pattern['description'],
                weight=1.0,
                decompose=True,
                auto_embellish=False
            )
            
            print("\n   ✅ Successfully decomposed and synthesized!")
            print(f"   Interpretation: {result.get('interpretation', 'N/A')}")
            print(f"   Total experts created: {result.get('total_experts', 0)}")
            
            if 'components' in result:
                print(f"\n   Components ({len(result['components'])}):")
                for j, comp in enumerate(result['components']):
                    comp_result = comp.get('result', {})
                    experts = comp_result.get('experts_added', 0)
                    states = comp_result.get('states_created', 0)
                    print(f"     {j+1}. {comp['description']}")
                    print(f"        Type: {comp.get('type', 'unknown')}")
                    if experts > 0:
                        print(f"        ✓ {experts} experts, {states} states")
                    else:
                        print("        ✗ Failed to synthesize")
            
            if result.get('states_created', 0) > 0:
                print(f"\n   📊 States automatically generated: {result['states_created']}")
                available_states = agent.state_manager.get_available_states()
                for category in ['global', 'particle', 'species']:
                    if category in available_states and available_states[category]:
                        print(f"      {category.capitalize()}: {', '.join(available_states[category])}")
            
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
    
    print("\n\n" + "="*60)
    print("Generating combined a-life sketch...")
    print("-" * 60)
    
    combined_agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
    combined_patterns = [
        "particles form groups where each one lives or dies based on how many neighbors it has",
        "some particles leave trails that others follow to find food sources",
        "predator particles hunt prey particles in coordinated groups"
    ]
    
    print("\nAdding behaviors for combined sketch:")
    for pattern in combined_patterns:
        try:
            if "trail" in pattern:
                await combined_agent.add_complex_behavior(pattern, weight=0.8, decompose=True)
            else:
                await combined_agent.add_behavior(pattern, weight=1.0)
            print(f"  ✓ Added: {pattern[:50]}...")
        except Exception as e:
            print(f"  ✗ Failed: {pattern[:50]}... - {e}")
    
    _, sketch_path = combined_agent.generate_sketch(
        description="Artificial life patterns demonstration",
        filename="demo_alife_patterns",
        use_timestamp=True
    )
    print(f"\n📄 Combined sketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_alife")
    
    return combined_agent, sketch_path


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
    agent = BehaviorAgent(tv, model_name="gemini-2.0-flash")
    
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
                await agent.add_drawing_behavior(description, 1.0)
            else:
                await agent.add_behavior(description, 1.0)
            print(f"✓ {description}: Success")
        except Exception as e:
            print(f"✗ {description}: Failed - {e}")
    
    _, sketch_path = agent.generate_sketch(
        description="Custom behavior demonstration",
        filename="demo_custom_behaviors",
        use_timestamp=True
    )
    print(f"\nSketch saved to: {sketch_path}")
    
    main_trace.complete("success")
    save_trace_with_report(collector, main_trace, "demo_custom")
    
    return agent, sketch_path


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
    
    Path("examples/generated_sketches").mkdir(parents=True, exist_ok=True)
    Path("examples/generated_sketches/traces").mkdir(parents=True, exist_ok=True)
    
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
            print("\nGenerated sketches are in: examples/generated_sketches/")
            print("Trace files are in: examples/generated_sketches/traces/")
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
    from dotenv import load_dotenv
    env_path = Path.cwd() / ".env"
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