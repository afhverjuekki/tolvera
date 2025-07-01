#!/usr/bin/env python3
"""
Test PoE system with comprehensive CSV logging.

This script demonstrates the PoE behavior system while capturing
all LLM interactions in a CSV file for analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import taichi as ti
from tolvera import Tolvera
from src.tolvera.llm.poe_integration import TolveraBehaviorAgent
from src.tolvera.llm.poe_synthesis import PureLLMSynthesizer
from src.tolvera.llm.poe_logger import get_logger
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_with_logging():
    """Run tests and generate CSV log file."""
    
    print("\n" + "="*80)
    print("🧪 PoE SYSTEM TEST WITH CSV LOGGING")
    print("="*80)
    print("\nThis test will:")
    print("1. Generate experts from various behavior descriptions")
    print("2. Log all LLM interactions to 'poe_llm_interactions.csv'")
    print("3. Show success/failure patterns")
    print("4. Display summary statistics")
    print("="*80)
    
    # Initialize CSV logger with custom filename
    csv_logger = get_logger("poe_test_results.csv")
    
    # Initialize Tölvera
    print("\n📊 Initializing Tölvera...")
    tv = Tolvera(
        width=800,
        height=600,
        pn=200,
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
                ti.random() * tv.y
            ])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = i % 2
            tv.p.field[i].size = 3.0
    
    init_particles()
    
    # Set species colors
    tv.s.species.field[0].rgba = [1.0, 0.5, 0.2, 1.0]  # Orange
    tv.s.species.field[1].rgba = [0.2, 0.5, 1.0, 1.0]  # Blue
    
    # Test various behavior descriptions
    test_behaviors = [
        # Simple behaviors (likely to succeed)
        ("particles fall downward with gravity", 1.0),
        ("particles drift slowly to the right", 0.8),
        ("particles move upward against gravity", 0.8),
        
        # Medium complexity (mixed results)
        ("particles are attracted to the center of the screen", 0.5),
        ("particles repel each other when they get close", 0.5),
        ("particles slow down over time due to friction", 0.5),
        
        # Complex behaviors (likely to fail or produce unexpected results)
        ("particles form a spiral pattern", 0.3),
        ("particles orbit around the center", 0.3),
        ("particles flock together like birds", 0.3),
        
        # Edge cases
        ("particles bounce off walls", 0.4),
        ("particles accelerate randomly", 0.4),
        ("particles follow sine wave patterns", 0.3),
    ]
    
    print(f"\n📝 Testing {len(test_behaviors)} different behaviors...")
    successful_experts = []
    failed_attempts = []
    
    for idx, (description, weight) in enumerate(test_behaviors):
        try:
            print(f"\n[{idx+1}/{len(test_behaviors)}] Testing: '{description}'")
            
            expert = await agent.add_expert_from_description(
                description,
                synthesizer.synthesizer,
                weight=weight
            )
            successful_experts.append((expert.name, description))
            print(f"✅ Success: {expert.name}")
            
        except Exception as e:
            failed_attempts.append((description, str(e)))
            print(f"❌ Failed: {e}")
    
    # Show results summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Total attempts: {len(test_behaviors)}")
    print(f"Successful: {len(successful_experts)} ({len(successful_experts)/len(test_behaviors)*100:.1f}%)")
    print(f"Failed: {len(failed_attempts)} ({len(failed_attempts)/len(test_behaviors)*100:.1f}%)")
    
    print("\n✅ Successful experts:")
    for name, desc in successful_experts:
        print(f"   - {name}: {desc}")
    
    print("\n❌ Failed attempts:")
    for desc, error in failed_attempts:
        print(f"   - {desc}: {error}")
    
    # Get and display CSV logger statistics
    print("\n" + "="*80)
    print("📈 CSV LOGGER STATISTICS")
    print("="*80)
    
    stats = csv_logger.get_summary_stats()
    if "error" not in stats:
        print(f"Total logged attempts: {stats['total_attempts']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Models used: {', '.join(stats['models_used'])}")
        
        if stats['common_errors']:
            print("\nMost common errors:")
            for error, count in stats['common_errors'].items():
                print(f"   - {error}: {count} occurrences")
    else:
        print(f"Error reading stats: {stats['error']}")
    
    print(f"\n📄 Full results saved to: poe_test_results.csv")
    print("You can open this file in Excel or any CSV viewer for detailed analysis.")
    
    # If we have successful experts, run a quick demo
    if successful_experts:
        print("\n" + "="*80)
        print("🎮 Running quick demo with successful experts...")
        print("Press 'Esc' to exit")
        print("="*80)
        
        frame_count = 0
        
        @tv.render
        def render():
            nonlocal frame_count
            
            # Update behavior system
            agent.update(dt=0.016)
            
            # Show info every 2 seconds
            frame_count += 1
            if frame_count % 120 == 0:
                avg_speed = 0.0
                for i in range(min(10, tv.pn)):
                    vel = tv.p.field[i].vel
                    avg_speed += (vel[0]**2 + vel[1]**2)**0.5
                avg_speed /= min(10, tv.pn)
                print(f"Frame {frame_count}: Avg speed: {avg_speed:.2f}")
            
            return tv.px
        
        tv.run()
    else:
        print("\n⚠️  No successful experts generated. Check:")
        print("   1. Ollama is running ('ollama serve')")
        print("   2. Model is available ('ollama pull qwen2.5:3b')")
        print("   3. Review the CSV file for error patterns")


async def analyze_csv_only():
    """Just analyze existing CSV file without running new tests."""
    
    print("\n" + "="*80)
    print("📊 ANALYZING EXISTING CSV LOG")
    print("="*80)
    
    csv_logger = get_logger("poe_test_results.csv")
    stats = csv_logger.get_summary_stats()
    
    if "error" not in stats:
        print(f"Total logged attempts: {stats['total_attempts']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Models used: {', '.join(stats['models_used'])}")
        
        if stats['common_errors']:
            print("\nMost common errors:")
            for error, count in stats['common_errors'].items():
                print(f"   - {error}: {count} occurrences")
    else:
        print(f"Error: {stats['error']}")


async def main():
    """Main menu."""
    
    print("\n" + "="*80)
    print("🧪 PoE SYSTEM TESTING WITH CSV LOGGING")
    print("="*80)
    print("\nOptions:")
    print("1. Run full test suite (generates new CSV)")
    print("2. Analyze existing CSV file")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-2): ").strip()
    
    if choice == "1":
        await test_with_logging()
    elif choice == "2":
        await analyze_csv_only()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    asyncio.run(main())