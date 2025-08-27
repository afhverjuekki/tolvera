# Import all documentation from modular files
from .tolvera_core_api import TOLVERA_CORE_API
from .particles_api import PARTICLES_API, PARTICLE_PATTERNS
from .pixels_api import PIXELS_API, PIXELS_PATTERNS
from .taichi_essentials import TAICHI_ESSENTIALS, TAICHI_CRASH_FIXES

# Import exemplars
from .exemplars.boids_exemplar import BOIDS_EXEMPLAR
from .exemplars.particle_life_exemplar import PARTICLE_LIFE_EXEMPLAR
from .exemplars.slime_exemplar import SLIME_EXEMPLAR

# Combine core APIs into base context
BASE_CONTEXT = f"""
{TOLVERA_CORE_API}

{PARTICLES_API}

{PIXELS_API}

{TAICHI_ESSENTIALS}

{TAICHI_CRASH_FIXES}
"""

# Combine patterns for supplementary context
PATTERNS_CONTEXT = f"""
{PARTICLE_PATTERNS}

{PIXELS_PATTERNS}
"""

# Combine exemplars
EXEMPLARS_CONTEXT = f"""
{BOIDS_EXEMPLAR}

{PARTICLE_LIFE_EXEMPLAR}

{SLIME_EXEMPLAR}
"""

# Legacy exports for backward compatibility
# These maintain the original variable names that may be referenced elsewhere
TOLVERA_PARTICLES_API = PARTICLES_API
TOLVERA_PIXELS_API = PIXELS_API
TOLVERA_STATE_API = """
# State System API (Extracted from Core API)

## State Categories
- Global States: System-wide parameters INCLUDING time-based (gravity, temperature, day_phase, time_of_day)
  - Access: `tv.s.llm_global.field[0].state_name`
  - Example ranges: gravity (0-1000), temperature (0-100), day_phase (0.0-1.0), time_of_day (0.0-24.0)

- Particle States: Per-particle custom data (energy, home_position, memory)
  - Access: `tv.s.llm_particle.field[i].state_name`

- Species States: Per-species configuration (aggression, speed_modifier)
  - Access: `tv.s.llm_species.field[species_id].state_name`

## When to Create Custom States
Only create custom states for properties that DON'T already exist:
- ✅ Create states for: energy, home_pos, pheromone_strength, day_phase, gravity_strength
- ❌ DON'T create states for: mass, pos, vel, species (already in particle struct)
"""

# Function to get base context (always included)
def get_base_context():
    """Get the base context that should always be included in prompts."""
    return BASE_CONTEXT

# Function to get supplementary patterns
def get_patterns_context():
    """Get supplementary patterns for additional context."""
    return PATTERNS_CONTEXT

# Function to get exemplars
def get_exemplars_context():
    """Get golden exemplars for reference."""
    return EXEMPLARS_CONTEXT

# Export all documentation strings
__all__ = [
    # Primary exports
    'BASE_CONTEXT',
    'PATTERNS_CONTEXT', 
    'EXEMPLARS_CONTEXT',
    
    # Functions
    'get_base_context',
    'get_patterns_context',
    'get_exemplars_context',
    
    # Legacy compatibility
    'TOLVERA_CORE_API',
    'TOLVERA_PARTICLES_API',
    'TOLVERA_PIXELS_API',
    'TOLVERA_STATE_API',
    
    # Individual components (if needed)
    'PARTICLES_API',
    'PARTICLE_PATTERNS',
    'PIXELS_API',
    'PIXELS_PATTERNS',
    'TAICHI_ESSENTIALS',
    'TAICHI_CRASH_FIXES',
    'BOIDS_EXEMPLAR',
    'PARTICLE_LIFE_EXEMPLAR',
    'SLIME_EXEMPLAR',
]