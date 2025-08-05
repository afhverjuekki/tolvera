"""
Context-aware prompt builder for intelligent prompt generation.
Automatically selects relevant context based on behavior descriptions.
"""

from typing import Dict, List, Optional, Set
import re
from ..context.library_docs import (
    TOLVERA_CORE_API, 
    PIXELS_API, 
    TAICHI_ESSENTIALS,
    STATE_ACCESS_PATTERNS,
    BOUNDARY_HANDLING
)
from ..context.patterns import (
    MOVEMENT_PATTERNS,
    FLOCKING_PATTERNS,
    INTERACTION_PATTERNS,
    TEMPORAL_PATTERNS,
    CELLULAR_AUTOMATA_PATTERNS,
    EMERGENT_PATTERNS
)
from ..context.examples import EXAMPLE_EXPERTS
from ..context.vera_patterns import VERA_PATTERNS, INTERACTION_PATTERNS_VERA, SPECIES_INTERACTION_PATTERNS
from ..context.drawing_patterns import DRAWING_PATTERNS, DRAWING_API_REFERENCE
from ..context.taichi_patterns import (
    TAICHI_FUNDAMENTALS,
    ARTIFICIAL_LIFE_PATTERNS,
    IML_PATTERNS
)
from ..context.alife_patterns import (
    EVOLUTION_PATTERNS,
    ECOSYSTEM_PATTERNS,
    MORPHOGENETIC_PATTERNS,
    SWARM_INTELLIGENCE
)


class ContextAwarePromptBuilder:
    """Builds comprehensive prompts with relevant context for expert synthesis."""
    
    def __init__(self):
        self.contexts = {
            'core_api': TOLVERA_CORE_API,
            'pixels_api': PIXELS_API,
            'taichi': TAICHI_ESSENTIALS,
            'taichi_fundamentals': TAICHI_FUNDAMENTALS,
            'state_access': STATE_ACCESS_PATTERNS,
            'boundaries': BOUNDARY_HANDLING,
            'movement': MOVEMENT_PATTERNS,
            'flocking': FLOCKING_PATTERNS,
            'interaction': INTERACTION_PATTERNS,
            'temporal': TEMPORAL_PATTERNS,
            'cellular': CELLULAR_AUTOMATA_PATTERNS,
            'emergent': EMERGENT_PATTERNS,
            'vera_patterns': VERA_PATTERNS,
            'vera_interactions': INTERACTION_PATTERNS_VERA,
            'species_interactions': SPECIES_INTERACTION_PATTERNS,
            'drawing': DRAWING_PATTERNS,
            'drawing_api': DRAWING_API_REFERENCE,
            'alife_patterns': ARTIFICIAL_LIFE_PATTERNS,
            'iml_patterns': IML_PATTERNS,
            'evolution': EVOLUTION_PATTERNS,
            'ecosystem': ECOSYSTEM_PATTERNS,
            'morphogenesis': MORPHOGENETIC_PATTERNS,
            'swarm': SWARM_INTELLIGENCE,
            'examples': EXAMPLE_EXPERTS
        }
        
    def build_synthesis_prompt(
        self,
        description: str,
        available_states: Dict[str, List[str]],
        include_contexts: Optional[List[str]] = None,
        constrained: bool = True
    ) -> str:
        """Build comprehensive prompt for expert synthesis"""
        
        # Auto-detect relevant contexts
        if include_contexts is None:
            include_contexts = self._detect_relevant_contexts(description)
        
        # Always include core contexts
        include_contexts = ['core_api', 'taichi', 'taichi_fundamentals'] + include_contexts
        
        # Add state access if states are available
        if available_states:
            include_contexts.append('state_access')
        
        # Remove duplicates while preserving order
        seen = set()
        include_contexts = [x for x in include_contexts if not (x in seen or seen.add(x))]
        
        prompt_sections = []
        
        # System instruction
        if constrained:
            prompt_sections.append(
                "Generate a structured JSON representation of a Taichi expert function "
                "following the BehaviorSynthesisResponse schema. The response should include "
                "all necessary experts, states, species configuration, and integration kernel."
            )
        else:
            prompt_sections.append(
                "Generate complete Taichi expert functions for the Tölvera particle system."
            )
        
        # Add physics conventions (CRITICAL for correct behavior)
        prompt_sections.append("""
## PHYSICS CONVENTIONS (CRITICAL):
- Coordinate system: STANDARD MATHEMATICAL/PHYSICS CONVENTION
  - Origin (0,0) is at TOP-LEFT corner of screen
  - X increases to the RIGHT (positive X = rightward)  
  - Y increases UPWARD (positive Y = upward)
  - This is the standard physics/math coordinate system
- Gravity: Since Y+ points up, gravity force must be NEGATIVE Y
  - Correct: return ti.math.vec2(0.0, -gravity_strength * mass)  # Negative for downward
  - Wrong: return ti.math.vec2(0.0, gravity_strength * mass)  # This would make particles fly up!
  - Use gravity_strength values 300 to 800 (the negative sign goes in the force calculation)
- Upward forces: Use POSITIVE Y (e.g., jumping, rising)
- Screen bounds: (0, 0) to (tv.x, tv.y)
- Force scaling: Use larger values (200-1000) for visible motion
- ALWAYS use @ti.func decorator (NOT @ti.kernel) for expert functions

## FORCE BALANCING GUIDELINES:
Force magnitudes should create emergent behaviors without being overpowering:
- **Gravity**: Use strength 300-800, apply as negative Y: ti.math.vec2(0.0, -gravity_strength * mass)
- **Chase/Hunt**: 400-600 (strong but catchable)
- **Flee/Escape**: 300-500 (slightly weaker than chase for drama)
- **Flocking alignment**: 50-200 (gentle influences)
- **Cohesion**: 100-300 (group together)
- **Separation**: 100-300 (avoid collisions)
- **Random movement**: 20-100 (idle behavior, exploration)
- **Center attraction**: 200-400 (medium strength)
- **Orbital motion**: 300-500 (tangential force)
- **Repulsion**: 200-2000 (inverse with distance)

Detection ranges for interactions:
- **Predator vision**: 200-300 units
- **Prey awareness**: 250-350 units (larger for survival)
- **Flocking neighbors**: 50-100 units
- **Separation bubble**: 20-40 units
- **Long-range attraction**: 300-500 units

## TAICHI VARIABLE DECLARATION (CRITICAL):
ALL variables MUST be declared before conditional branches:
❌ WRONG:
if species == 0:
    strength = 150.0
else:
    strength = 50.0
return strength * vec  # ERROR!

✅ CORRECT:
strength = 50.0  # Default value
if species == 0:
    strength = 150.0
return strength * vec  # OK!

## TAICHI RETURN STATEMENTS (CRITICAL - THIS WILL CRASH IF WRONG):
NEVER use return inside if/for/while blocks - Taichi will crash with "Return inside non-static if"!

❌ WRONG - THIS EXACT PATTERN CAUSES CRASHES:
if species != 1:
    return ti.math.vec2(0.0, 0.0)  # CRASH: "Return inside non-static if"

❌ ALSO WRONG:
if species == 0:
    return ti.math.vec2(0.0, 0.0)  # CRASH!
else:
    return normal_force  # CRASH!

✅ CORRECT - ALWAYS USE THIS PATTERN:
# 1. Declare result variable BEFORE any conditionals
result = ti.math.vec2(0.0, 0.0)  # Default value

# 2. Set result inside conditionals (no return!)
if species == 1:
    # Do calculations here
    result = calculated_force
    
# 3. Single return at the END of function
return result

✅ ANOTHER CORRECT EXAMPLE:
force = ti.math.vec2(0.0, 0.0)  # Always declare first
if species == 0:  # Predator
    # Calculate predator behavior
    force = chase_force
elif species == 1:  # Prey
    # Calculate prey behavior
    force = flee_force
# Single return at end
return force

REMEMBER: Every expert function MUST follow this pattern or it will crash!""")
        
        # Add species detection hint
        prompt_sections.append(
            "\nIMPORTANT: Analyze the description to detect species mentions and requirements. "
            "Use the species parameter in your expert functions to implement species-specific behaviors."
        )
        
        # Add contexts
        prompt_sections.append("\n## AVAILABLE CONTEXT\n")
        for ctx_name in include_contexts:
            if ctx_name in self.contexts and ctx_name != 'examples':
                prompt_sections.append(f"### {ctx_name.replace('_', ' ').title()}")
                prompt_sections.append(self.contexts[ctx_name])
                prompt_sections.append("")
        
        # Add state context
        if available_states:
            prompt_sections.append("### Available States")
            prompt_sections.append(self._format_state_context(available_states))
            prompt_sections.append("")
        
        # Always add particle property context
        prompt_sections.append("### Built-in Particle Properties (Always Available)")
        prompt_sections.append("Every particle already has these properties - DO NOT create custom states for them:")
        prompt_sections.append("- `tv.p.field[i].pos`: Position (ti.math.vec2)")
        prompt_sections.append("- `tv.p.field[i].vel`: Velocity (ti.math.vec2)")
        prompt_sections.append("- `tv.p.field[i].mass`: Mass (ti.f32)")
        prompt_sections.append("- `tv.p.field[i].size`: Display size (ti.f32)")
        prompt_sections.append("- `tv.p.field[i].species`: Species ID (ti.i32)")
        prompt_sections.append("- `tv.p.field[i].active`: Activity level (ti.f32)")
        prompt_sections.append("")
        
        # Add specific examples if relevant
        relevant_examples = self._find_relevant_examples(description)
        if relevant_examples:
            prompt_sections.append("### Relevant Examples")
            for ex_name, ex_data in relevant_examples.items():
                prompt_sections.append(f"#### {ex_data['description']}")
                prompt_sections.append(f"```python\n{ex_data['code']}\n```")
            prompt_sections.append("")
        
        # Task specification
        prompt_sections.append("## TASK")
        prompt_sections.append(f"Create expert function(s) for: **{description}**")
        prompt_sections.append("")
        prompt_sections.append("IMPORTANT REQUIREMENTS:")
        prompt_sections.append("- Function names should NOT have 'expert_' prefix")
        prompt_sections.append("- Use descriptive names like 'gravity', 'random_walk', 'chase', etc.")
        prompt_sections.append("- Use the species parameter to implement species-specific behaviors")
        prompt_sections.append("- When detecting species, assign semantic colors in the colors field:")
        prompt_sections.append("  - Predators/hunters: red [1.0, 0.2, 0.2, 1.0]")
        prompt_sections.append("  - Prey/food/plants: green [0.2, 0.8, 0.2, 1.0]")
        prompt_sections.append("  - Protectors/guardians: blue [0.2, 0.2, 1.0, 1.0]")
        prompt_sections.append("  - Neutral/peaceful: yellow [1.0, 0.9, 0.2, 1.0]")
        prompt_sections.append("  - Always use RGBA format with alpha=1.0")
        prompt_sections.append("")
        
        if constrained:
            prompt_sections.append("Return a JSON object matching the BehaviorSynthesisResponse schema:")
            prompt_sections.append("""```json
{
    "experts": [
        {
            "name": "descriptive_name",
            "description": "What this expert does",
            "is_interaction": false,
            "weight": 1.0,
            "computation": {
                "state_accesses": [],
                "helper_variables": {},
                "conditionals": [],
                "force_expression": {"x": "0.0", "y": "0.0"}
            }
        }
    ],
    "states_needed": {
        "global": {},
        "particle": {},
        "species": {}
    },
    "species_config": {
        "species_ids": [0, 1],
        "species_names": {"0": "predator", "1": "prey"},
        "interaction_pairs": [[0, 1]],
        "colors": {
            "0": [1.0, 0.2, 0.2, 1.0],
            "1": [0.2, 0.8, 0.2, 1.0]
        }
    },
    "integration_kernel": {
        "single_experts": ["expert_name"],
        "interaction_experts": [],
        "species_conditions": null
    },
    "temporal_update": null
}
```""")
        else:
            prompt_sections.append("Requirements:")
            prompt_sections.append("1. Use the exact function signature shown in examples")
            prompt_sections.append("2. Access only available states listed above")
            prompt_sections.append("3. Return a ti.math.vec2 force vector")
            prompt_sections.append("4. Handle edge cases (division by zero, boundaries)")
            prompt_sections.append("5. Include descriptive docstring")
            prompt_sections.append("\nGenerate ONLY the @ti.func code:")
        
        return "\n".join(prompt_sections)
    
    def build_state_analysis_prompt(self, description: str) -> str:
        """Build prompt for analyzing what states are needed"""
        return f"""Analyze what custom states (if any) are needed for this behavior: "{description}"

IMPORTANT: The following properties are ALREADY AVAILABLE on every particle and should NOT be recreated:
- pos (ti.math.vec2): Position
- vel (ti.math.vec2): Velocity  
- mass (ti.f32): Mass
- size (ti.f32): Display size
- speed (ti.f32): Speed magnitude
- species (ti.i32): Species ID
- active (ti.f32): Activity level
- ppos, pvel: Previous position/velocity

Only create NEW states for properties that don't already exist.

Consider:
1. Does it need to track time or phases? (global states)
2. Does it need per-particle memory or properties NOT listed above? (particle states)  
3. Does it need species-specific configuration? (species states)
4. Can it be implemented with just the existing properties?

Return a JSON object with this structure:
{{
    "needs_states": true/false,
    "global_states": {{"state_name": {{"type": "ti.f32", "min": 0.0, "max": 1000.0, "description": "...", "initial": 300.0}}}},
    "particle_states": {{"state_name": {{"type": "ti.f32", "min": 0.0, "max": 100.0, "description": "...", "initial": null}}}},
    "species_states": {{"state_name": {{"type": "ti.f32", "min": 0.0, "max": 1.0, "description": "...", "initial": null}}}},
    "temporal_config": {{"requires_time": true/false, "day_duration": 10.0}}
}}

Common state types with PROPER RANGES:
- Gravity strength: ti.f32, min: 0.0, max: 1000.0, initial: 300.0
- Force magnitudes: ti.f32, min: 0.0, max: 1000.0
- Energy/Resource: ti.f32, min: 0.0, max: 100.0, initial: 80.0
- Day phase: ti.f32, min: 0.0, max: 1.0
- Memory positions: ti.math.vec2 (for home_pos, target_pos)
- Counters: ti.i32"""
    
    def _detect_relevant_contexts(self, description: str) -> List[str]:
        """Intelligently detect which contexts to include"""
        desc_lower = description.lower()
        contexts = []
        
        # Movement patterns
        if any(word in desc_lower for word in ['move', 'drift', 'wander', 'random', 'orbit', 'spiral', 'circle']):
            contexts.append('movement')
        
        # Flocking behaviors
        if any(word in desc_lower for word in ['flock', 'boid', 'align', 'cohesion', 'separation', 'swarm', 'school', 'herd']):
            contexts.append('flocking')
            contexts.append('vera_patterns')  # Include vera flocking examples
        
        # Interactions
        if any(word in desc_lower for word in ['chase', 'flee', 'hunt', 'escape', 'repel each other', 'attract each other', 'species']):
            contexts.append('interaction')
            contexts.append('species_interactions')  # Always include best practices
            if 'species' in desc_lower:
                contexts.append('vera_interactions')  # Multi-species patterns
        
        # Species-specific behaviors
        species_indicators = ['species', 'predator', 'prey', 'different types', 'types of particles', 
                            'red particles', 'blue particles', 'green particles']
        if any(indicator in desc_lower for indicator in species_indicators):
            contexts.append('interaction')  # Species often involve interactions
        
        # Temporal behaviors
        if any(word in desc_lower for word in ['day', 'night', 'time', 'energy', 'tired', 'phase', 'cycle', 'periodic', 'rhythm']):
            contexts.append('temporal')
        
        # Cellular automata
        if any(word in desc_lower for word in ['cellular', 'automaton', 'automata', 'game of life', 'conway', 'grid', 'cells', 'neighbors']):
            contexts.append('cellular')
        
        # Emergent behaviors
        if any(word in desc_lower for word in ['slime', 'physarum', 'ant', 'trail', 'pheromone', 'firefly', 'sync', 'emerge']):
            contexts.append('emergent')
            contexts.append('vera_patterns')  # Include slime mold implementation
        
        # Drawing and visualization
        if any(word in desc_lower for word in ['draw', 'drawing', 'line', 'circle', 'rect', 'color', 'visual', 'trail', 'streak', 'halo', 'glow', 'effect', 'overlay', 'transparency', 'fade', 'blur']):
            contexts.append('drawing')
            contexts.append('drawing_api')
        
        # Pixel operations
        if any(word in desc_lower for word in ['trail', 'pheromone', 'draw', 'pixel', 'mark', 'deposit', 'paint']):
            contexts.append('pixels_api')
            contexts.append('vera_patterns')  # Include trail deposition examples
        
        # Boundary handling
        if any(word in desc_lower for word in ['bounce', 'wrap', 'edge', 'boundary', 'wall', 'confine']):
            contexts.append('boundaries')
        
        # Force-based behaviors
        if any(word in desc_lower for word in ['gravity', 'spring', 'damping', 'force', 'attract', 'repel']):
            contexts.append('vera_patterns')  # Include force patterns
        
        # Evolution and genetics
        if any(word in desc_lower for word in ['evolve', 'evolution', 'genetic', 'breed', 'mutate', 'fitness', 'selection']):
            contexts.append('evolution')
            contexts.append('alife_patterns')
        
        # Ecosystem behaviors
        if any(word in desc_lower for word in ['ecosystem', 'predator', 'prey', 'food', 'resource', 'symbiosis', 'parasite']):
            contexts.append('ecosystem')
            contexts.append('alife_patterns')
        
        # Morphogenesis
        if any(word in desc_lower for word in ['grow', 'growth', 'morph', 'develop', 'differentiate', 'cell', 'divide']):
            contexts.append('morphogenesis')
            contexts.append('alife_patterns')
        
        # Swarm behaviors
        if any(word in desc_lower for word in ['swarm', 'ant', 'bee', 'colony', 'hive', 'quorum', 'collective']):
            contexts.append('swarm')
            contexts.append('alife_patterns')
        
        # Interactive behaviors
        if any(word in desc_lower for word in ['interactive', 'iml', 'gesture', 'control', 'map', 'feedback']):
            contexts.append('iml_patterns')
        
        # General artificial life
        if any(word in desc_lower for word in ['life', 'alive', 'living', 'creature', 'organism', 'artificial life']):
            contexts.append('alife_patterns')
        
        return contexts
    
    def _find_relevant_examples(self, description: str) -> Dict:
        """Find examples similar to the requested behavior"""
        desc_lower = description.lower()
        relevant = {}
        
        # Extract key concepts from description
        desc_words = set(re.findall(r'\b\w+\b', desc_lower))
        
        for name, example in EXAMPLE_EXPERTS.items():
            ex_desc_lower = example['description'].lower()
            ex_words = set(re.findall(r'\b\w+\b', ex_desc_lower))
            
            # Calculate similarity
            common_words = desc_words & ex_words
            
            # Strong matches - multiple word overlap or key concept match
            if len(common_words) >= 2:
                relevant[name] = example
                continue
            
            # Concept-based matches
            concept_matches = [
                ('gravity' in desc_lower and 'gravity' in ex_desc_lower),
                ('center' in desc_lower and 'center' in ex_desc_lower),
                ('chase' in desc_lower and 'chase' in ex_desc_lower),
                ('flee' in desc_lower and 'flee' in name),
                ('random' in desc_lower and 'random' in ex_desc_lower),
                ('drift' in desc_lower and 'drift' in ex_desc_lower),
                ('flock' in desc_lower and 'flock' in name),
                ('align' in desc_lower and 'align' in name),
                ('orbit' in desc_lower and 'orbit' in name),
                ('day' in desc_lower and 'day' in name),
                ('night' in desc_lower and 'night' in name),
                ('energy' in desc_lower and 'energy' in name),
                ('home' in desc_lower and 'home' in name),
                ('species' in desc_lower and 'species' in ex_desc_lower),
            ]
            
            if any(concept_matches):
                relevant[name] = example
        
        # Limit to most relevant examples
        if len(relevant) > 3:
            # Sort by relevance (word overlap count)
            sorted_examples = sorted(
                relevant.items(),
                key=lambda x: len(desc_words & set(re.findall(r'\b\w+\b', x[1]['description'].lower()))),
                reverse=True
            )
            relevant = dict(sorted_examples[:3])
        
        return relevant
    
    def _format_state_context(self, states: Dict[str, List[str]]) -> str:
        """Format available states for prompt"""
        lines = []
        
        if 'global' in states and states['global']:
            lines.append("**Global States** (access with `tv.s.llm_global.field[0].state_name`):")
            for state in states['global']:
                lines.append(f"- {state}")
            lines.append("")
        
        if 'particle' in states and states['particle']:
            lines.append("**Particle States** (access with `tv.s.llm_particle.field[particle_idx].state_name`):")
            for state in states['particle']:
                lines.append(f"- {state}")
            lines.append("")
        
        if 'species' in states and states['species']:
            lines.append("**Species States** (access with `tv.s.llm_species.field[species].state_name`):")
            for state in states['species']:
                lines.append(f"- {state}")
        
        return "\n".join(lines) if lines else "No custom states are currently available."
    
    def build_kernel_integration_prompt(
        self,
        expert_names: List[str],
        interaction_expert_names: List[str],
        species_info: Dict
    ) -> str:
        """Build prompt for kernel integration"""
        prompt = f"""Generate a Taichi kernel that integrates the following expert forces:

Single-particle experts: {expert_names}
Interaction experts: {interaction_expert_names}
Species in use: {species_info.get('species_ids', [0])}

The kernel should:
1. Loop through all particles
2. Apply single-particle experts to get forces
3. If interaction experts exist, nested loop for particle pairs
4. Update velocity with damping (0.99)
5. Update position
6. Handle boundaries (bounce with 0.8 restitution)

Use this exact structure:
```python
@ti.kernel
def apply_all_experts():
    dt = 0.016
    
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            # Get particle properties
            # Apply single-particle experts
            # Apply interaction experts (if any)
            # Update velocity and position
            # Handle boundaries
```"""
        return prompt