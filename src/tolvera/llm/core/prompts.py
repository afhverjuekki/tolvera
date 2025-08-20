"""
Context-aware prompt builder for intelligent prompt generation.
Automatically selects relevant context based on behavior descriptions.
"""

from typing import Dict, List, Optional
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
try:
    from ..context.initialization_patterns import INITIALIZATION_PATTERNS, SPECIES_INITIALIZATION_PATTERNS
    from ..context.temporal_patterns_extended import TEMPORAL_UPDATE_PATTERNS, TEMPORAL_CONFIGURATION_PATTERNS
except ImportError:
    INITIALIZATION_PATTERNS = ""
    SPECIES_INITIALIZATION_PATTERNS = ""
    TEMPORAL_UPDATE_PATTERNS = ""
    TEMPORAL_CONFIGURATION_PATTERNS = ""

try:
    from ..context.temporal_dynamics import TEMPORAL_DYNAMICS_PATTERNS, TEMPORAL_UPDATE_EXAMPLES
except ImportError:
    TEMPORAL_DYNAMICS_PATTERNS = ""
    TEMPORAL_UPDATE_EXAMPLES = ""


class ContextAwarePromptBuilder:
    """Builds comprehensive prompts with relevant context for expert synthesis."""
    
    def __init__(self):
        # Golden examples for intelligent selection
        self.golden_examples = {
            'interaction': {
                'source': 'particle-life.py:120-167',
                'description': 'Species interaction with matrix lookup and proper variable declaration',
                'keywords': ['interaction', 'species', 'matrix', 'attract', 'repel', 'chase', 'hunt'],
                'code': '''@ti.func
def particle_life_interaction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    for j in range(tv.pn):
        if particle_idx != j and tv.p.field[j].active > 0:
            # CRITICAL: Declare variables BEFORE conditionals
            direction = ti.math.vec2(0.0, 0.0)
            force_magnitude = 0.0
            
            diff = tv.p.field[j].pos - pos
            dist = diff.norm()
            
            if dist > 0.001 and dist < interaction_radius:
                direction = diff / dist
                force_magnitude = attraction * (1.0 - dist / interaction_radius)
                force += direction * force_magnitude
    return force'''
            },
            'flocking': {
                'source': 'boids.py:143-178',
                'description': 'Flocking behavior with proper variable declaration and species filtering',
                'keywords': ['flock', 'boid', 'separation', 'alignment', 'cohesion', 'school', 'swarm'],
                'code': '''@ti.func
def separation_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    count = 0
    
    for j in range(tv.pn):
        if particle_idx != j and tv.p.field[j].active > 0:
            diff = pos - tv.p.field[j].pos
            dist = diff.norm()
            
            # PROVEN PATTERN: Declare before nested conditional
            normalized_diff = ti.math.vec2(0.0, 0.0)
            
            if dist > 0.001 and dist < separation_radius:
                normalized_diff = diff / dist
                force += normalized_diff / dist
                count += 1
    
    result_force = ti.math.vec2(0.0, 0.0)
    if count > 0:
        force = force / ti.cast(count, ti.f32)
        result_force = (force - vel) * separation_weight
    return result_force'''
            },
            'simple_force': {
                'source': 'particle-life.py:170-174',
                'description': 'Simple single-particle force with direct calculation',
                'keywords': ['gravity', 'friction', 'drag', 'simple', 'basic'],
                'code': '''@ti.func
def friction_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    friction_coefficient = 0.5
    force = -vel * friction_coefficient
    return force'''
            }
        }
        
        self.contexts = {
            'core_api': TOLVERA_CORE_API,
            'pixels_api': PIXELS_API,
            'taichi': TAICHI_ESSENTIALS,
            'taichi_fundamentals': TAICHI_FUNDAMENTALS,
            'state_access': STATE_ACCESS_PATTERNS,
            'boundaries': BOUNDARY_HANDLING,
            'movement': MOVEMENT_PATTERNS,
            'initialization': INITIALIZATION_PATTERNS,
            'species_initialization': SPECIES_INITIALIZATION_PATTERNS,
            'temporal_updates': TEMPORAL_UPDATE_PATTERNS,
            'temporal_dynamics': TEMPORAL_DYNAMICS_PATTERNS,  # New comprehensive temporal patterns
            'temporal_examples': TEMPORAL_UPDATE_EXAMPLES,    # New temporal examples
            'temporal_patterns_extended': TEMPORAL_UPDATE_PATTERNS,  # Extended patterns
            'configuration': TEMPORAL_CONFIGURATION_PATTERNS,
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
    
    def build_prompt(
        self,
        description: str,
        expert_type: str,
        contexts: set,
        available_states: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Build a prompt for a specific expert type with relevant contexts.
        
        Args:
            description: Natural language description of the behavior
            expert_type: Type of expert to generate ('force', 'interaction', 'utility', 'temporal_update', etc.)
            contexts: Set of context names to include
            available_states: Optional available states dictionary
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add relevant contexts
        for context_name in contexts:
            if context_name in self.contexts:
                prompt_parts.append(self.contexts[context_name])
        
        # Add expert-type specific guidance
        if expert_type == 'utility':
            prompt_parts.append("""
UTILITY EXPERT GUIDELINES:
- Create @ti.func functions with NO particle parameters (no pos, vel, mass, species, particle_idx)
- Most utility functions take NO parameters at all: @ti.func def expert_name():
- These handle state updates, temporal dynamics, and helper calculations
- Access states directly: tv.s.llm_particle.field[i] or tv.s.llm_global.field[0]
- Do NOT return force vectors (no ti.math.vec2 or ti.Vector returns)
- Most have NO return statement (void functions)
- Loop over particles internally if needed: for i in range(tv.pn)

Example:
@ti.func
def expert_update_phase():
    # NO parameters, NO return
    for i in range(tv.pn):
        tv.s.llm_particle.field[i].phase += 0.01
""")
        elif expert_type == 'temporal_update':
            prompt_parts.append("""
TEMPORAL UPDATE GUIDELINES:
- CRITICAL: These are parameterless @ti.func functions: @ti.func def expert_name():
- ABSOLUTELY NO particle parameters (no pos, vel, mass, species, particle_idx)
- ABSOLUTELY NO return statement - these are void functions  
- Handle day/night cycles, energy depletion, growth, oscillations
- Access temporal states: tv.s.llm_temporal.field[0].state_name
- Loop over particles if needed: for i in range(tv.pn)
- Common pattern: increment/decrement state values over time

CORRECT EXAMPLE:
@ti.func  
def expert_day_night_cycle():
    # NO parameters, NO return
    tv.s.llm_temporal.field[0].day_phase = (tv.s.llm_temporal.field[0].day_phase + 0.001) % 1.0

WRONG (DO NOT GENERATE):
@ti.func
def expert_name(pos, vel, mass, species, particle_idx) -> ti.math.vec2:
    # This is completely wrong for temporal updates!
""")
        elif expert_type == 'initialization':
            prompt_parts.append("""
INITIALIZATION GUIDELINES:
- Create kernels that set initial particle positions and properties
- Use patterns like random, grid, clustered, ring distributions
- Set species IDs and colors appropriately
- Initialize custom states if needed
""")
        
        return "\n\n".join(prompt_parts)
    
    def detect_needed_contexts(self, description: str) -> set:
        """Auto-detect which contexts are relevant for a description."""
        contexts = set()
        desc_lower = description.lower()
        
        # Pattern detection
        if any(word in desc_lower for word in ['temporal', 'time', 'day', 'night', 'cycle', 'phase']):
            contexts.add('temporal_dynamics')
            contexts.add('temporal_patterns_extended')
        
        
        if any(word in desc_lower for word in ['cellular', 'automaton', 'game of life', 'conway']):
            contexts.add('cellular')
        
        if any(word in desc_lower for word in ['draw', 'trail', 'glow', 'visual', 'render']):
            contexts.add('drawing')
            contexts.add('drawing_api')
        
        
        return contexts
    
    def select_golden_examples(self, description: str) -> Dict[str, str]:
        """Select the most relevant golden examples based on description keywords."""
        desc_lower = description.lower()
        selected = {}
        
        for example_name, example_data in self.golden_examples.items():
            # Check if any keywords match
            for keyword in example_data['keywords']:
                if keyword in desc_lower:
                    selected[example_name] = {
                        'source': example_data['source'],
                        'description': example_data['description'],
                        'code': example_data['code']
                    }
                    break
        
        return selected
        
    def build_synthesis_prompt(
        self,
        description: str,
        available_states: Dict[str, List[str]],
        include_contexts: Optional[List[str]] = None,
        constrained: bool = True,
        context: Optional[Dict] = None
    ) -> str:
        """Build comprehensive prompt for expert synthesis with full decomposition context"""
        
        # Auto-detect relevant contexts
        if include_contexts is None:
            include_contexts = self._detect_relevant_contexts(description)
        
        # Add pattern-specific contexts if provided
        if context and context.get('pattern_type'):
            pattern_type = context['pattern_type']
            if pattern_type == 'cellular_automaton':
                include_contexts.extend(['cellular', 'temporal'])
            elif pattern_type == 'physarum':
                include_contexts.extend(['movement', 'alife_patterns'])
            elif pattern_type == 'ecosystem':
                include_contexts.extend(['ecosystem', 'species_interactions'])
            elif pattern_type == 'swarm':
                include_contexts.extend(['swarm', 'flocking'])
        
        # Always include core contexts
        include_contexts = ['core_api', 'taichi', 'taichi_fundamentals'] + include_contexts
        
        # Add state access if states are available
        if available_states:
            include_contexts.append('state_access')
        
        # Remove duplicates while preserving order
        seen = set()
        include_contexts = [x for x in include_contexts if not (x in seen or seen.add(x))]
        
        prompt_sections = []
        
        # Add 5-element structure at the beginning
        prompt_sections.append("""## ROLE
You are an EXPERT TAICHI COMPUTATIONAL PHYSICIST and BEHAVIOR SYNTHESIS SPECIALIST with deep expertise in:
- GPU-accelerated particle physics simulation using Taichi lang
- Force-based behavior synthesis and artificial life systems
- Multi-species ecosystem modeling and emergent behavior design
- Tölvera particle system architecture and custom state management
- Mathematical modeling of natural phenomena (flocking, predation, cellular automata)

Your expertise spans classical physics simulation, swarm intelligence, evolutionary algorithms, and complex adaptive systems. You understand how to translate natural language descriptions into precise mathematical force calculations that produce believable, emergent behaviors in particle simulations.

## OBJECTIVE
Your primary objective is to synthesize robust, efficient Taichi expert functions that transform natural language behavior descriptions into mathematically sound force calculations. You must generate functions that:
- Produce emergent, believable particle behaviors that match the description
- Execute efficiently on GPU hardware through Taichi compilation
- Handle edge cases gracefully (zero distances, boundary conditions, species mismatches)
- Integrate seamlessly with the Tölvera ecosystem and state management system
- Scale appropriately for systems with hundreds to thousands of particles

## TASK AT HAND
You must analyze the provided behavior description and create expert functions by:

1. Behavior Classification: Determine if this is a single-particle force, interaction between particles, temporal update, or visual effect
2. Species Detection: Identify any species mentioned in the description and assign semantic roles (predator, prey, neutral)
3. Force Physics Analysis: Translate the natural language into precise force calculations with appropriate magnitudes
4. State Requirement Analysis: Determine if custom states are needed beyond basic particle properties
5. Taichi Code Generation: Create syntactically correct @ti.func functions following all Taichi constraints
6. Integration Specification: Define how the expert integrates into the particle system's force calculation loop

## KEY EXAMPLES

### Example 1: Single-Particle Force (FROM: particle-life.py:170-174)
Input: "particles experience friction for stability"
Analysis: Universal force applied to all particles, no interaction needed
Working Code:
```python
@ti.func
def friction_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Apply velocity-dependent friction for stability.'''
    friction_coefficient = 0.5
    force = -vel * friction_coefficient
    return force
```
Key Patterns: Simple force calculation, direct return, proper parameter signature

### Example 2: Multi-Species Interaction Expert (FROM: particle-life.py:120-167)
Input: "species attract or repel each other based on interaction matrix"
Analysis: Complex species interaction using state matrix, demonstrates proper variable declaration
Working Code:
```python
@ti.func
def particle_life_interaction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Calculate attraction/repulsion forces based on species interaction matrix.'''
    force = ti.math.vec2(0.0, 0.0)
    
    for j in range(tv.pn):
        if particle_idx != j and tv.p.field[j].active > 0:
            other_pos = tv.p.field[j].pos
            other_species = tv.p.field[j].species
            
            # Get interaction parameters from the matrix
            attraction = tv.s.llm_species.field[species, other_species].attraction_force
            interaction_radius = tv.s.llm_species.field[species, other_species].interaction_radius
            
            # Calculate distance and direction
            diff = other_pos - pos
            dist = diff.norm()
            
            # CRITICAL: Declare variables BEFORE conditionals
            direction = ti.math.vec2(0.0, 0.0)
            force_magnitude = 0.0
            
            if dist > 0.001 and dist < interaction_radius:
                direction = diff / dist
                # Apply attraction/repulsion with distance falloff
                force_magnitude = attraction * (1.0 - dist / interaction_radius)
                force += direction * force_magnitude
    
    return force
```
Key Patterns: Proper particle_idx usage, variable declaration before conditionals, state matrix access

### Example 3: Flocking with Multiple Behaviors (FROM: boids.py:143-252)
Input: "particles flock together using separation, alignment, and cohesion"
Analysis: Multiple behavioral components working together, species filtering
Working Code:
```python
@ti.func
def separation_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    '''Steer to avoid crowding local flockmates.'''
    force = ti.math.vec2(0.0, 0.0)
    separation_radius = tv.s.llm_global.field[0].separation_radius
    count = 0
    
    for j in range(tv.pn):
        if particle_idx != j and tv.p.field[j].active > 0:
            other_pos = tv.p.field[j].pos
            diff = pos - other_pos
            dist = diff.norm()
            
            # CRITICAL: Declare normalized_diff before conditional
            normalized_diff = ti.math.vec2(0.0, 0.0)
            
            if dist > 0.001 and dist < separation_radius:
                # Repel from nearby boids
                normalized_diff = diff / dist
                # Weight by inverse distance (closer = stronger repulsion)
                force += normalized_diff / dist
                count += 1
    
    # Normalize and apply species weight
    result_force = ti.math.vec2(0.0, 0.0)
    if count > 0:
        force = force / ti.cast(count, ti.f32)
        force_norm = force.norm()
        if force_norm > 0.001:
            # Normalize and scale
            force = (force / force_norm) * tv.s.llm_global.field[0].max_speed
            # Apply steering force
            result_force = force - vel
            # Apply species-specific weight
            result_force *= tv.s.llm_species.field[species].separation_weight
    
    return result_force
```
Key Patterns: Variable declared before use, proper force normalization, state access, species-specific parameters

## SUCCESS VS. FAILURE CRITERIA

### SUCCESS CRITERIA:
✅ Syntactic Correctness: All Taichi code compiles without syntax errors and follows @ti.func conventions
✅ Physical Realism: Force magnitudes produce believable motion (300-800 for gravity, 200-600 for chase/flee)
✅ Edge Case Handling: Properly handles zero distances, out-of-bounds particles, and invalid species IDs
✅ Species Accuracy: Correctly identifies and implements species-specific behaviors from natural language
✅ Performance Optimization: Uses efficient algorithms suitable for GPU parallel execution
✅ State Minimization: Only creates necessary custom states, leverages existing particle properties
✅ Integration Compatibility: Functions work seamlessly with Tölvera's particle update loop

### FAILURE CRITERIA:
❌ Return Statement Errors: Any return statements inside conditional blocks (causes "Return inside non-static if")
❌ Variable Declaration Issues: Variables declared inside conditionals without default values outside them
❌ Wrong Vector Types: Mixing ti.Vector with ti.math.vec2 or using incorrect method calls
❌ Mathematical Errors: Division by zero, incorrect normalization, or NaN-producing calculations
❌ Species Misidentification: Incorrectly assigning species roles or missing multi-species interactions
❌ State Overuse: Creating unnecessary custom states for properties already available on particles
❌ Force Imbalance: Using inappropriate force magnitudes that produce unrealistic motion""")
        
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

## IMPORTANT EXPERT SYNTHESIS RULES:
1. DO NOT generate experts for species initialization or configuration - this is handled elsewhere
2. DO NOT generate experts that just set particle properties without returning forces
3. FOCUS on behavior experts that return actual force vectors
4. Each expert should have ONE clear behavioral purpose
5. Avoid creating multiple experts that do the same thing

## FORCE BALANCING GUIDELINES:
Force magnitudes should create emergent behaviors without being overpowering:
- Gravity: Use strength 300-800, apply as negative Y: ti.math.vec2(0.0, -gravity_strength * mass)
- Chase/Hunt: 400-600 (strong but catchable)
- Flee/Escape: 300-500 (slightly weaker than chase for drama)
- Flocking alignment: 50-200 (gentle influences)
- Cohesion: 100-300 (group together)
- Separation: 100-300 (avoid collisions)
- Random movement: 20-100 (idle behavior, exploration)
- Center attraction: 200-400 (medium strength)
- Orbital motion: 300-500 (tangential force)
- Repulsion: 200-2000 (inverse with distance)

Detection ranges for interactions:
- Predator vision: 200-300 units
- Prey awareness: 250-350 units (larger for survival)
- Flocking neighbors: 50-100 units
- Separation bubble: 20-40 units
- Long-range attraction: 300-500 units

## CRITICAL TAICHI SYNTAX (MUST FOLLOW TO AVOID CRASHES):

### VECTOR OPERATIONS - USE CORRECT METHODS:
❌ WRONG - These will crash with AttributeError:
```python
dist = diff.norm()  # ERROR if diff is not ti.math.vec2
dir = diff.normalized()  # ERROR: no normalized() method
vec = ti.Vector([x, y]).norm()  # ERROR on ti.Vector
```

✅ CORRECT - Use these patterns:
```python
# For ti.math.vec2 (PREFERRED):
force = ti.math.vec2(0.0, 0.0)
diff = p2.pos - p1.pos  # Assuming pos is ti.math.vec2
dist = diff.norm()  # Works for ti.math.vec2

# For normalization:
if dist > 0.001:
    direction = diff / dist  # Manual normalize
# OR use ti.math functions:
direction = ti.math.normalize(diff)
length = ti.math.length(diff)

# For ti.Vector (older style):
vec = ti.Vector([x, y])
dist = ti.sqrt(vec[0]2 + vec[1]2)  # Manual magnitude
```

### TAICHI VARIABLE DECLARATION (CRITICAL - CAUSES "NAME NOT DEFINED" ERRORS):
ALL variables MUST be declared before conditional branches:

❌ WRONG - Variable declared inside nested conditionals:
```python
if neighbor_count > 0:
    avg_velocity /= neighbor_count
    if avg_velocity.norm() > 0.001:
        desired_velocity = (avg_velocity / avg_velocity.norm()) * speed  # Declared here
    else:
        desired_velocity = vel  # Also declared here
    force = desired_velocity - vel  # ERROR: desired_velocity may not be defined!
```

✅ CORRECT - Declare BEFORE any conditionals:
```python
if neighbor_count > 0:
    avg_velocity /= neighbor_count
    desired_velocity = vel  # DECLARE with default value FIRST
    if avg_velocity.norm() > 0.001:
        desired_velocity = (avg_velocity / avg_velocity.norm()) * speed  # MODIFY
    force = desired_velocity - vel  # OK: always defined
```

❌ WRONG - Simple case:
```python
if species == 0:
    strength = 150.0
else:
    strength = 50.0
return strength * vec  # ERROR!
```

✅ CORRECT - Simple case:
```python
strength = 50.0  # Default value FIRST
if species == 0:
    strength = 150.0  # Modify if needed
return strength * vec  # OK!```

## MATH FUNCTIONS - USE TAICHI VERSIONS:
❌ WRONG - Python math module doesn't work in Taichi:
```python
import math
angle = math.sin(t)  # ERROR in Taichi scope
dist = math.sqrt(x*x + y*y)  # ERROR
```

✅ CORRECT - Use ti.* math functions:
```python
angle = ti.sin(t)
dist = ti.sqrt(x*x + y*y)
cos_val = ti.cos(angle)
abs_val = ti.abs(x)
max_val = ti.max(a, b)
min_val = ti.min(a, b)
random_val = ti.random()  # NOT random.random()
```

## PARTICLE INDEX PARAMETER (CRITICAL - COMMON ERROR):
When iterating through particles, you MUST use the `particle_idx` parameter passed to your function, NOT an undefined variable 'i'.

### ❌ WRONG - Undefined 'i' Error:
```python
@ti.func
def expert_function(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    for j in range(tv.pn):
        if i != j:  # ERROR: name 'i' is not defined!
            # ... rest of code
```

### ✅ CORRECT - Use particle_idx parameter:
```python
@ti.func
def expert_function(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    for j in range(tv.pn):
        if particle_idx != j:  # CORRECT: use the parameter passed to the function
            # ... rest of code
```

The `particle_idx` parameter is the index of the current particle being processed. Always use this parameter when you need to:
- Skip self in loops: `if particle_idx != j:`
- Access current particle's custom states: `tv.s.llm_particle.field[particle_idx].some_state`
- Create particle-specific randomness: `angle = (ti.random() + particle_idx * 0.1) * 2 * 3.14159`

## TEMPORAL STATE UPDATES (FOR TIME-BASED BEHAVIORS):
When implementing behaviors that change over time:

### State Access Pattern:
```python
# Access temporal states via llm_particle/llm_global containers
energy = tv.s.llm_particle.field[i].energy
phase = tv.s.llm_particle.field[i].phase
day_phase = tv.s.llm_global.field[0].day_phase
```

### Update Rates Based on Description:
- "slowly" → multiply by 0.999 or add/subtract 0.001
- "gradually" → multiply by 0.99 or add/subtract 0.01  
- "quickly" → multiply by 0.95 or add/subtract 0.05
- "rapidly" → multiply by 0.9 or add/subtract 0.1

### Behavioral Coupling:
```python
# Low energy affects movement
if energy < 20.0:
    vel *= 0.8  # Tired particles move slower
    
# Age affects behavior
if age > lifecycle_midpoint:
    size = base_size * (1.0 - (age - lifecycle_midpoint) / lifecycle_midpoint)
```

## TAICHI RETURN STATEMENTS (CRITICAL - #1 CAUSE OF CRASHES):
NEVER use return inside if/for/while blocks - Taichi will crash with "Return inside non-static if"!

### ❌ WRONG PATTERNS THAT CRASH:
```python
# CRASH EXAMPLE 1: Early return for species check
@ti.func
def predator_hunt(...) -> ti.math.vec2:
    if species != 0:  # Not a predator
        return ti.math.vec2(0.0, 0.0)  # CRASH!
    # Rest of code...

# CRASH EXAMPLE 2: Return in species branch
@ti.func
def species_force(...) -> ti.math.vec2:
    if species == 0:
        return predator_force()  # CRASH!
    elif species == 1:
        return prey_force()  # CRASH!
        
# CRASH EXAMPLE 3: Early exit when no target found
@ti.func
def chase_behavior(...) -> ti.math.vec2:
    target = find_target()
    if target < 0:
        return ti.math.vec2(0.0, 0.0)  # CRASH!
```

### ✅ CORRECT PATTERNS FROM VERIFIED EXEMPLARS:

# GOLDEN PATTERN 1: Variable Declaration (FROM: particle-life.py:139-141)
@ti.func
def particle_life_interaction(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    
    for j in range(tv.pn):
        if particle_idx != j and tv.p.field[j].active > 0:
            # CRITICAL: Declare variables BEFORE conditionals - this prevents crashes
            direction = ti.math.vec2(0.0, 0.0)  # ← MUST declare here
            force_magnitude = 0.0               # ← MUST declare here
            
            diff = tv.p.field[j].pos - pos
            dist = diff.norm()
            
            if dist > 0.001 and dist < interaction_radius:
                direction = diff / dist  # ← Now safe to assign
                force_magnitude = attraction * (1.0 - dist / interaction_radius)
                force += direction * force_magnitude
    
    return force  # Single return only

# GOLDEN PATTERN 2: Complex Conditional Logic (FROM: boids.py:155-156, 345-356) 
@ti.func
def separation_force(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    count = 0
    
    for j in range(tv.pn):
        if particle_idx != j and tv.p.field[j].active > 0:
            diff = pos - tv.p.field[j].pos
            dist = diff.norm()
            
            # PROVEN PATTERN: Declare before nested conditional
            normalized_diff = ti.math.vec2(0.0, 0.0)  # ← MUST be here
            
            if dist > 0.001 and dist < separation_radius:
                normalized_diff = diff / dist  # Safe to assign after declaration
                force += normalized_diff / dist
                count += 1
    
    # PROVEN PATTERN: Multiple variables for final calculation
    result_force = ti.math.vec2(0.0, 0.0)  # Default result
    if count > 0:
        force = force / ti.cast(count, ti.f32)
        force_norm = force.norm()
        if force_norm > 0.001:
            force = (force / force_norm) * max_speed
            result_force = force - vel  # SET result, don't return
    
    return result_force  # Single return pattern

# CORRECT EXAMPLE 2: Flocking behavior - MUST use particle_idx to skip self
@ti.func
def flocking_cohesion(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    # Calculate center of nearby same-species particles
    center = ti.math.vec2(0.0, 0.0)
    neighbor_count = 0
    perception_radius = 80.0
    
    for j in range(tv.pn):
        if particle_idx != j:  # CRITICAL: Skip self using particle_idx parameter!
            other = tv.p.field[j]
            if other.species == species and other.active > 0:
                diff = other.pos - pos
                if diff.norm() < perception_radius:
                    center += other.pos
                    neighbor_count += 1
    
    # Calculate cohesion force
    force = ti.math.vec2(0.0, 0.0)
    if neighbor_count > 0:
        center /= neighbor_count
        to_center = center - pos
        force = to_center * 80.0  # Strong cohesion for visible movement
    
    return force

# CORRECT EXAMPLE 3: Multi-species with different behaviors
@ti.func
def species_behavior(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    # Declare result first
    result = ti.math.vec2(0.0, 0.0)
    
    # Use if/elif to SET result, never return inside
    if species == 0:  # Predators
        # Hunt logic
        for j in range(tv.pn):
            if tv.p.field[j].species != species:
                diff = tv.p.field[j].pos - pos
                dist = diff.norm()
                if dist < 100.0 and dist > 0.001:
                    result = (diff / dist) * 300.0  # Manual normalize
                    break
    elif species == 1:  # Prey
        # Flee logic
        for j in range(tv.pn):
            if tv.p.field[j].species == 0:
                diff = pos - tv.p.field[j].pos
                dist = diff.norm()
                if dist < 150.0 and dist > 0.001:
                    result = (diff / dist) * 400.0  # Manual normalize
                    break
    elif species == 2:  # Neutral
        # Wander randomly
        t = ti.cast(tv.ctx.i[None], ti.f32) * 0.01
        result = ti.math.vec2(ti.sin(t + particle_idx), ti.cos(t + particle_idx)) * 50.0
    
    # Only ONE return at the very end
    return result
```

GOLDEN RULE: Declare your result variable FIRST, modify it in branches, return ONCE at the END!

## NOTE ON IMPLEMENTATION APPROACH:
The examples above show FROM-SCRATCH implementations using Taichi and Tölvera's state system.
These demonstrate how to build behaviors at the lowest level for maximum control and customization.
While Tölvera provides vera behaviors (tv.v.flock, tv.v.slime, etc.) that handle common patterns,
creating custom experts allows you to:
- Implement unique physics and interactions
- Fine-tune specific parameters and forces
- Combine behaviors in novel ways
- Access and modify particle states directly
- Create behaviors not covered by the vera library

## TOROIDAL WRAPPING (RECOMMENDED FOR ECOSYSTEMS):
For ecosystem simulations, use wrap-aware distance calculations to prevent boundary issues:

```python
@ti.func
def wrap_distance(pos1: ti.math.vec2, pos2: ti.math.vec2) -> ti.math.vec2:
    \"\"\"Calculate wrapped distance between two positions for toroidal topology.\"\"\"
    diff = pos2 - pos1
    
    # Wrap X
    if ti.abs(diff.x) > tv.x * 0.5:
        if diff.x > 0:
            diff.x -= tv.x
        else:
            diff.x += tv.x
    
    # Wrap Y
    if ti.abs(diff.y) > tv.y * 0.5:
        if diff.y > 0:
            diff.y -= tv.y
        else:
            diff.y += tv.y
    
    return diff
```

Use this in your experts:
```python
# Instead of: diff = target_pos - pos
diff = wrap_distance(pos, target_pos)
dist = diff.norm()
if dist > 0.001:
    direction = diff / dist
```""")
        
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
        
        # Add decomposition context if available
        if context and 'decomposition' in context:
            decomposition = context['decomposition']
            if decomposition:
                prompt_sections.append("## DECOMPOSITION CONTEXT")
                prompt_sections.append(f"Original description: {getattr(decomposition, 'original_description', description)}")
                prompt_sections.append(f"Interpretation: {getattr(decomposition, 'interpretation', '')}")
                
                # Add details about other components
                if 'other_components' in context:
                    prompt_sections.append("\n### Other Components in this Behavior:")
                    for comp in context['other_components']:
                        prompt_sections.append(f"- {comp['name']} ({comp['type']}): {comp['description']}")
                        if comp.get('implementation_details'):
                            prompt_sections.append(f"  Details: {'; '.join(comp['implementation_details'])}")
                        if comp.get('depends_on'):
                            prompt_sections.append(f"  Depends on: {', '.join(comp['depends_on'])}")
                    prompt_sections.append("")
                
                # Add current component details
                if 'component' in context:
                    component = context['component']
                    prompt_sections.append("### Current Component Being Synthesized:")
                    prompt_sections.append(f"Name: {getattr(component, 'expert_name', 'unknown')}")
                    prompt_sections.append(f"Type: {getattr(component, 'expert_type', 'force')}")
                    prompt_sections.append(f"Description: {getattr(component, 'description', '')}")
                    prompt_sections.append(f"Implementation: {getattr(component, 'implementation', '')}")
                    
                    # Add detailed implementation guidance if available
                    if 'implementation_details' in context and context['implementation_details']:
                        prompt_sections.append("\nImplementation Steps:")
                        for detail in context['implementation_details']:
                            prompt_sections.append(f"- {detail}")
                    
                    if 'parameters' in context and context['parameters']:
                        prompt_sections.append("\nRequired Parameters:")
                        for param_name in context['parameters']:
                            prompt_sections.append(f"- {param_name}")
                    
                    prompt_sections.append("")
        
        # Task specification
        prompt_sections.append("## TASK")
        prompt_sections.append(f"Create expert function(s) for: {description}")
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
        
        # Add implementation-specific guidance
        if context and 'implementation_details' in context:
            prompt_sections.append("\nFOLLOW THESE SPECIFIC IMPLEMENTATION STEPS PROVIDED BY THE DECOMPOSER:")
            for i, detail in enumerate(context['implementation_details'], 1):
                prompt_sections.append(f"{detail}")
        
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
    
    def build_state_analysis_prompt(self, description: str, expert_type: Optional[str] = None) -> str:
        """Build prompt for analyzing what states are needed"""
        
        # Add expert-type specific examples
        expert_examples = ""
        if expert_type == 'force':
            expert_examples = """
FORCE EXPERT EXAMPLES:
- "particles chase food" → needs 'consumed' state (particle, ti.i32) to mark eaten items
- "energy depletes as they move" → needs 'energy' state (particle, ti.f32, 0-100)
- "particles return home when tired" → needs 'home_pos' (particle, ti.math.vec2) AND 'energy'
- "magnetic particles" → needs 'charge' state (particle, ti.f32, -1 to 1)
"""
        elif expert_type == 'interaction':
            expert_examples = """
INTERACTION EXPERT EXAMPLES:
- "predators hunt within range" → needs 'hunt_radius' (species, ti.f32, 50-200)
- "particles form social bonds" → needs 'bond_count' (particle, ti.i32, 0-10)
- "remember last encounter" → needs 'last_interaction_time' (particle, ti.f32)
"""
        elif expert_type == 'visual':
            expert_examples = """
VISUAL EXPERT EXAMPLES:
- "particles blink periodically" → needs 'blink_phase' (particle, ti.f32, 0-6.28)
- "color cycles through spectrum" → needs 'hue_shift' (particle, ti.f32, 0-360)
- "pulsing size" → needs 'pulse_timer' (particle, ti.f32, 0-1)
"""
        
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
{expert_examples}

CRITICAL DECISION LOGIC:
1. Set "needs_states": true ONLY if you are adding at least one state to ANY category
2. Set "needs_states": false if ALL state dictionaries will be empty
3. If needs_states is true, you MUST include at least one state in global_states, particle_states, or species_states
4. If needs_states is false, ALL state dictionaries MUST be empty

Consider:
1. Does it need to track time, phases, or system-wide parameters? → Add to global_states
2. Does it need per-particle memory or properties NOT listed above? → Add to particle_states  
3. Does it need species-specific configuration? → Add to species_states
4. Can it be implemented with just the existing properties? → Set needs_states: false

IMPORTANT CONSISTENCY RULES:
- If needs_states is true, at least one state dictionary must contain states
- If needs_states is false, all state dictionaries must be empty {{}}
- Each state MUST have a unique, descriptive name based on its purpose
- State names should be snake_case (e.g., day_phase, home_pos, energy_level)

Return a JSON object with this EXACT structure:
{{
    "needs_states": true/false,  // true if ANY states are needed, false if NONE needed
    "global_states": {{           // Empty {{}} if no global states needed
        "state_name": {{
            "name": "state_name",  // MUST match the key
            "type": "ti.f32",      // Use official Taichi types
            "min": 0.0,            // Minimum value for numeric types
            "max": 1000.0,         // Maximum value for numeric types
            "description": "What this state represents",
            "initial": 300.0       // Initial value (can be null)
        }}
    }},
    "particle_states": {{         // Empty {{}} if no particle states needed
        "state_name": {{
            "name": "state_name",  // MUST match the key
            "type": "ti.f32",
            "min": 0.0,
            "max": 100.0,
            "description": "Per-particle state description",
            "initial": null
        }}
    }},
    "species_states": {{          // Empty {{}} if no species states needed
        "state_name": {{
            "name": "state_name",  // MUST match the key
            "type": "ti.f32",
            "min": 0.0,
            "max": 1.0,
            "description": "Per-species configuration",
            "initial": null
        }}
    }}
}}

Common state types with PROPER RANGES (use these as templates):
- Gravity strength (global): ti.f32, min: 0.0, max: 1000.0, initial: 300.0
- Force magnitudes (global): ti.f32, min: 0.0, max: 1000.0
- Energy/Resource (particle): ti.f32, min: 0.0, max: 100.0, initial: 80.0
- Day phase (global): ti.f32, min: 0.0, max: 1.0, initial: 0.25
- Time of day (global): ti.f32, min: 0.0, max: 24.0, initial: 12.0
- Season cycle (global): ti.f32, min: 0.0, max: 1.0, initial: 0.0
- Memory positions (particle): ti.math.vec2 (for home_pos, target_pos)
- Counters (global/particle): ti.i32, min: 0, max: 1000

EXAMPLE RESPONSES:

Example 1 - Behavior that needs states:
{{
    "needs_states": true,
    "global_states": {{
        "gravity_strength": {{
            "name": "gravity_strength",
            "type": "ti.f32",
            "min": 0.0,
            "max": 1000.0,
            "description": "Strength of gravitational force",
            "initial": 300.0
        }}
    }},
    "particle_states": {{}},
    "species_states": {{}}
}}

Example 2 - Behavior that doesn't need states:
{{
    "needs_states": false,
    "global_states": {{}},
    "particle_states": {{}},
    "species_states": {{}}
}}"""
    
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
        temporal_keywords = [
            'day', 'night', 'time', 'energy', 'tired', 'phase', 'cycle', 'periodic', 'rhythm',
            'over time', 'gradually', 'slowly', 'depletes', 'regenerates', 'ages', 'grows',
            'exhausted', 'hungry', 'loses energy', 'gains energy', 'weakens', 'strengthens',
            'decays', 'fades', 'oscillate', 'pulse', 'life cycle', 'generations'
        ]
        if any(keyword in desc_lower for keyword in temporal_keywords):
            contexts.append('temporal')
            contexts.append('temporal_dynamics')  # Include comprehensive temporal patterns
            contexts.append('temporal_examples')  # Include temporal examples
        
        # # Cellular automata
        # if any(word in desc_lower for word in ['cellular', 'automaton', 'automata', 'game of life', 'conway', 'grid', 'cells', 'neighbors']):
        #     contexts.append('cellular')
        
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
            # contexts.append('vera_patterns')  # Include trail deposition examples
        
        # Boundary handling
        if any(word in desc_lower for word in ['bounce', 'wrap', 'edge', 'boundary', 'wall', 'confine']):
            contexts.append('boundaries')
        
        # Force-based behaviors
        if any(word in desc_lower for word in ['gravity', 'spring', 'damping', 'force', 'attract', 'repel']):
            contexts.append('vera_patterns')  # Include force patterns
        
        # # Evolution and genetics
        # if any(word in desc_lower for word in ['evolve', 'evolution', 'genetic', 'breed', 'mutate', 'fitness', 'selection']):
        #     contexts.append('evolution')
        #     contexts.append('alife_patterns')
        
        # # Ecosystem behaviors
        # if any(word in desc_lower for word in ['ecosystem', 'predator', 'prey', 'food', 'resource', 'symbiosis', 'parasite']):
        #     contexts.append('ecosystem')
        #     contexts.append('alife_patterns')
        
        # Morphogenesis
        # if any(word in desc_lower for word in ['grow', 'growth', 'morph', 'develop', 'differentiate', 'cell', 'divide']):
        #     contexts.append('morphogenesis')
        #     contexts.append('alife_patterns')
        
        # # Swarm behaviors
        # if any(word in desc_lower for word in ['swarm', 'ant', 'bee', 'colony', 'hive', 'quorum', 'collective']):
        #     contexts.append('swarm')
        #     contexts.append('alife_patterns')
        
        # # Interactive behaviors
        # if any(word in desc_lower for word in ['interactive', 'iml', 'gesture', 'control', 'map', 'feedback']):
        #     contexts.append('iml_patterns')
    
        
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
            lines.append("Global States (access with `tv.s.llm_global.field[0].state_name`):")
            for state in states['global']:
                lines.append(f"- {state}")
            lines.append("")
        
        # Temporal states are now part of global states (removed llm_temporal category)
        
        if 'particle' in states and states['particle']:
            lines.append("Particle States (access with `tv.s.llm_particle.field[particle_idx].state_name`):")
            for state in states['particle']:
                lines.append(f"- {state}")
            lines.append("")
        
        if 'species' in states and states['species']:
            lines.append("Species States (access with `tv.s.llm_species.field[species].state_name`):")
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

Use this GOLDEN PATTERN from particle-life.py (lines 178-215):
```python
@ti.kernel
def apply_all_experts():
    '''Main physics kernel that applies all forces and updates particles.'''
    dt = 0.016  # 60 FPS timestep
    damping = tv.s.llm_global.field[0].damping
    
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            pos = tv.p.field[i].pos
            vel = tv.p.field[i].vel
            mass = tv.p.field[i].mass
            species = tv.p.field[i].species
            
            # Accumulate all forces
            total_force = ti.math.vec2(0.0, 0.0)
            
            # Main particle life interaction
            total_force += particle_life_interaction(pos, vel, mass, species, i)
            
            # Friction for stability
            total_force += friction_force(pos, vel, mass, species, i)
            
            # CRITICAL: Declare acceleration BEFORE conditional
            acceleration = ti.math.vec2(0.0, 0.0)
            
            # Update velocity (F = ma)
            if mass > 0:
                acceleration = total_force / mass
            else:
                acceleration = total_force
                
            tv.p.field[i].vel += acceleration * dt
            
            # Apply damping
            tv.p.field[i].vel *= damping
            
            # Update position
            tv.p.field[i].pos += tv.p.field[i].vel * tv.p.field[i].speed * dt
```

CRITICAL PATTERNS:
- Extract particle properties to local variables first
- Use proper expert function signatures with all 5 parameters
- Declare acceleration before conditional usage
- Apply damping after force integration
- Use particle index 'i' when calling experts (becomes particle_idx parameter)
"""
        return prompt