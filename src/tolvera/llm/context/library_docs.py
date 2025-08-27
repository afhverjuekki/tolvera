# Import necessary modules
from pathlib import Path

def load_context_file(filename):
    """Load a context file from the same directory."""
    path = Path(__file__).parent / filename
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_context_with_sections(filename):
    """Load a context file that has multiple sections separated by ---."""
    content = load_context_file(filename)
    # Split on the section separators
    sections = content.split('\n---\n')
    if len(sections) == 1:
        # No sections, just return the content
        return {'main': content}
    
    result = {}
    current_key = 'main'
    current_content = []
    
    for i, section in enumerate(sections):
        if i % 2 == 0:
            # Content section
            if current_content:
                result[current_key] = '\n'.join(current_content)
            current_content = [section]
            current_key = 'main' if i == 0 else current_key
        else:
            # Key section
            if current_content:
                result[current_key] = '\n'.join(current_content)
            current_key = section.strip()
            current_content = []
    
    # Don't forget the last section
    if current_content:
        result[current_key] = '\n'.join(current_content)
    
    return result

# Load all documentation from text files
TOLVERA_CORE_API = load_context_file('tolvera_core_api.txt')
PARTICLES_API = load_context_file('particles_api.txt')
PIXELS_API = load_context_file('pixels_api.txt')

# Load files with multiple sections
taichi_content = load_context_with_sections('taichi_essentials.txt')
TAICHI_ESSENTIALS = taichi_content.get('main', '') + '\n' + taichi_content.get('TAICHI_ESSENTIALS', '')
TAICHI_CRASH_FIXES = taichi_content.get('TAICHI_CRASH_FIXES', '')

# Load patterns
patterns_content = load_context_with_sections('patterns.txt')
PARTICLE_PATTERNS = patterns_content.get('PARTICLE_PATTERNS', '')

pixels_patterns_content = load_context_with_sections('pixels_api.txt')
PIXELS_PATTERNS = pixels_patterns_content.get('PIXELS_PATTERNS', '')

# Load exemplars
BOIDS_EXEMPLAR = load_context_file('exemplars/boids_exemplar.txt')
PARTICLE_LIFE_EXEMPLAR = load_context_file('exemplars/particle_life_exemplar.txt')
SLIME_EXEMPLAR = load_context_file('exemplars/slime_exemplar.txt')

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
    'SLIME_EXEMPLAR'
]