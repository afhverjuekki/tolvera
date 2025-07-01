#!/usr/bin/env python3
"""
Minimal example of PoE system with raw LLM generation.

This is the simplest possible demo showing:
1. How to generate a single behavior from natural language
2. The raw LLM output
3. The resulting particle behavior
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import taichi as ti
from tolvera import Tolvera, run
from src.tolvera.llm.poe_integration import TolveraBehaviorAgent
from src.tolvera.llm.poe_ollama import PoEExpertSynthesizer


async def setup_expert_system():
    """Setup the expert system with LLM-generated behavior."""
    
    # Initialize Tölvera
    tv = Tolvera(width=800, height=600, pn=200)
    
    # Create behavior agent
    agent = TolveraBehaviorAgent(tv)
    
    # Create LLM synthesizer
    synthesizer = PoEExpertSynthesizer()
    
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
            tv.p.field[i].species = 0
    
    init_particles()
    tv.s.species.field[0].rgba = [0.5, 0.8, 1.0, 1.0]  # Light blue
    
    # Generate expert from description
    print("\n🎯 Generating expert from description...")
    description = "particles fall downward like rain"
    
    try:
        expert = await agent.add_expert_from_description(description, synthesizer)
        print(f"\n✅ Successfully added expert: {expert.name}")
        print(f"📝 Expert code preview:\n{expert.code[:200]}...")
        
        return tv, agent
    except Exception as e:
        print(f"\n❌ Failed to create expert: {e}")
        return None, None


def main(**kwargs):
    """Main function that runs the PoE demo."""
    
    # Setup expert system
    tv, agent = asyncio.run(setup_expert_system())
    
    if tv is None or agent is None:
        print("❌ Setup failed - exiting")
        return
    
    # Render loop
    @tv.render
    def render():
        agent.update(dt=0.016)
        return tv.px
    
    print("\n🎮 Running simulation...")
    print("Press 'Esc' to exit")


if __name__ == "__main__":
    run(main)