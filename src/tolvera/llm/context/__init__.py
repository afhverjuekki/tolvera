"""
Context injection framework for LLM prompts.
"""

# Import from the central library_docs module which loads from .txt files
from .library_docs import (
    TOLVERA_CORE_API,
    TAICHI_ESSENTIALS,
    TAICHI_CRASH_FIXES,
    PIXELS_API,
    PIXELS_PATTERNS,
    PARTICLES_API,
    PARTICLE_PATTERNS,
    BOIDS_EXEMPLAR,
    PARTICLE_LIFE_EXEMPLAR,
    SLIME_EXEMPLAR
)

# Load pattern constants directly from text files
from pathlib import Path

def _load_patterns_from_file(filename, section_name):
    """Helper to load specific pattern sections from text files."""
    path = Path(__file__).parent / filename
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split on section markers
    sections = content.split('\n---\n')
    for i in range(1, len(sections), 2):
        if i < len(sections) and sections[i].strip() == section_name:
            return sections[i + 1] if i + 1 < len(sections) else ''
    
    # If no sections, return the whole content
    return content if not '\n---\n' in content else ''

# Load patterns from text files
MOVEMENT_PATTERNS = _load_patterns_from_file('patterns.txt', 'MOVEMENT_PATTERNS')
FLOCKING_PATTERNS = _load_patterns_from_file('patterns.txt', 'FLOCKING_PATTERNS')
INTERACTION_PATTERNS = _load_patterns_from_file('patterns.txt', 'INTERACTION_PATTERNS')
TEMPORAL_PATTERNS = _load_patterns_from_file('patterns.txt', 'TEMPORAL_PATTERNS')
CELLULAR_AUTOMATA_PATTERNS = _load_patterns_from_file('patterns.txt', 'CELLULAR_AUTOMATA_PATTERNS')
EMERGENT_PATTERNS = _load_patterns_from_file('patterns.txt', 'EMERGENT_PATTERNS')

# Load vera patterns
VERA_PATTERNS = _load_patterns_from_file('vera_patterns.txt', 'VERA_PATTERNS')
INTERACTION_PATTERNS_VERA = _load_patterns_from_file('vera_patterns.txt', 'INTERACTION_PATTERNS_VERA')
SPECIES_INTERACTION_PATTERNS = _load_patterns_from_file('vera_patterns.txt', 'main')

# Load drawing patterns
DRAWING_PATTERNS = _load_patterns_from_file('drawing_patterns.txt', 'main')
DRAWING_API_REFERENCE = _load_patterns_from_file('drawing_patterns.txt', 'DRAWING_API_REFERENCE')

# Load taichi patterns
TAICHI_FUNDAMENTALS = _load_patterns_from_file('taichi_patterns.txt', 'main')
ARTIFICIAL_LIFE_PATTERNS = _load_patterns_from_file('taichi_patterns.txt', 'ARTIFICIAL_LIFE_PATTERNS')
IML_PATTERNS = _load_patterns_from_file('taichi_patterns.txt', 'IML_PATTERNS')

# Load alife patterns
EVOLUTION_PATTERNS = _load_patterns_from_file('alife_patterns.txt', 'main')
ECOSYSTEM_PATTERNS = _load_patterns_from_file('alife_patterns.txt', 'ECOSYSTEM_PATTERNS')
MORPHOGENETIC_PATTERNS = _load_patterns_from_file('alife_patterns.txt', 'MORPHOGENETIC_PATTERNS')
SWARM_INTELLIGENCE = _load_patterns_from_file('alife_patterns.txt', 'SWARM_INTELLIGENCE')

# Note: EXAMPLE_EXPERTS is not actually used anywhere, so we're removing it

# Backward compatibility alias
TOLVERA_PIXELS_API = PIXELS_API

__all__ = [
    'TOLVERA_CORE_API',
    'TOLVERA_PIXELS_API',
    'PIXELS_API',
    'PIXELS_PATTERNS',
    'PARTICLES_API',
    'PARTICLE_PATTERNS',
    'TAICHI_ESSENTIALS',
    'TAICHI_CRASH_FIXES',
    'MOVEMENT_PATTERNS',
    'FLOCKING_PATTERNS',
    'INTERACTION_PATTERNS',
    'TEMPORAL_PATTERNS',
    'CELLULAR_AUTOMATA_PATTERNS',
    'EMERGENT_PATTERNS',
    'VERA_PATTERNS',
    'INTERACTION_PATTERNS_VERA',
    'SPECIES_INTERACTION_PATTERNS',
    'TAICHI_FUNDAMENTALS',
    'ARTIFICIAL_LIFE_PATTERNS',
    'IML_PATTERNS',
    'EVOLUTION_PATTERNS',
    'ECOSYSTEM_PATTERNS',
    'MORPHOGENETIC_PATTERNS',
    'SWARM_INTELLIGENCE',
    'DRAWING_PATTERNS',
    'DRAWING_API_REFERENCE',
    'BOIDS_EXEMPLAR',
    'PARTICLE_LIFE_EXEMPLAR',
    'SLIME_EXEMPLAR'
]