"""
Utility for loading LLM prompts from external text files.
Provides centralized prompt management with variable substitution and intelligent context selection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import asyncio

from ..debug.tracing import get_collector

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loads and manages LLM prompts from external text files with intelligent context selection."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the prompt loader.
        
        Args:
            base_path: Base directory for prompt files. If None, uses default.
        """
        if base_path is None:
            # Default to prompts directory relative to this file
            self.base_path = Path(__file__).parent.parent / "prompts"
        else:
            self.base_path = Path(base_path)
        
        # Initialize context selector (lazy loading to avoid circular imports)
        self._context_selector = None
        
        logger.debug(f"PromptLoader initialized with base path: {self.base_path}")
    
    def _get_context_selector(self):
        """Lazy load the context selector to avoid circular imports."""
        if self._context_selector is None:
            from .context_selector import ContextSelector
            self._context_selector = ContextSelector()
        return self._context_selector
    
    def _import_context_patterns(self, context_names: List[str]) -> Dict[str, str]:
        """
        Dynamically import context patterns based on selected names.
        
        Args:
            context_names: List of context names to import
            
        Returns:
            Dictionary mapping context names to their content
        """
        contexts = {}
        
        # Import mapping - from existing prompts.py imports
        context_imports = {
            'core_api': ('tolvera.llm.context.library_docs', 'TOLVERA_CORE_API'),
            'pixels_api': ('tolvera.llm.context.library_docs', 'PIXELS_API'),
            'taichi_fundamentals': ('tolvera.llm.context.taichi_patterns', 'TAICHI_FUNDAMENTALS'),
            'taichi_crashes': ('tolvera.llm.context.taichi_patterns', 'TAICHI_CRASH_FIXES'),
            'state_access': ('tolvera.llm.context.library_docs', 'STATE_ACCESS_PATTERNS'),
            'boundaries': ('tolvera.llm.context.library_docs', 'BOUNDARY_HANDLING'),
            'movement': ('tolvera.llm.context.patterns', 'MOVEMENT_PATTERNS'),
            'flocking': ('tolvera.llm.context.patterns', 'FLOCKING_PATTERNS'),
            'interaction': ('tolvera.llm.context.patterns', 'INTERACTION_PATTERNS'),
            'temporal': ('tolvera.llm.context.patterns', 'TEMPORAL_PATTERNS'),
            'cellular': ('tolvera.llm.context.patterns', 'CELLULAR_AUTOMATA_PATTERNS'),
            'emergent': ('tolvera.llm.context.patterns', 'EMERGENT_PATTERNS'),
            'drawing': ('tolvera.llm.context.drawing_patterns', 'DRAWING_PATTERNS'),
            'drawing_api': ('tolvera.llm.context.drawing_patterns', 'DRAWING_API_REFERENCE'),
            'vera_patterns': ('tolvera.llm.context.vera_patterns', 'VERA_PATTERNS'),
            'vera_interactions': ('tolvera.llm.context.vera_patterns', 'INTERACTION_PATTERNS_VERA'),
            'species_interactions': ('tolvera.llm.context.vera_patterns', 'SPECIES_INTERACTION_PATTERNS'),
            'alife_patterns': ('tolvera.llm.context.taichi_patterns', 'ARTIFICIAL_LIFE_PATTERNS'),
            'iml_patterns': ('tolvera.llm.context.taichi_patterns', 'IML_PATTERNS'),
            'evolution': ('tolvera.llm.context.alife_patterns', 'EVOLUTION_PATTERNS'),
            'ecosystem': ('tolvera.llm.context.alife_patterns', 'ECOSYSTEM_PATTERNS'),
            'morphogenesis': ('tolvera.llm.context.alife_patterns', 'MORPHOGENETIC_PATTERNS'),
            'swarm': ('tolvera.llm.context.alife_patterns', 'SWARM_INTELLIGENCE'),
            'initialization': ('tolvera.llm.context.initialization_patterns', 'INITIALIZATION_PATTERNS'),
            'species_initialization': ('tolvera.llm.context.initialization_patterns', 'SPECIES_INITIALIZATION_PATTERNS'),
            'temporal_updates': ('tolvera.llm.context.temporal_patterns_extended', 'TEMPORAL_UPDATE_PATTERNS'),
            'temporal_dynamics': ('tolvera.llm.context.temporal_dynamics', 'TEMPORAL_DYNAMICS_PATTERNS'),
            'temporal_examples': ('tolvera.llm.context.temporal_dynamics', 'TEMPORAL_UPDATE_EXAMPLES'),
            'temporal_patterns_extended': ('tolvera.llm.context.temporal_patterns_extended', 'TEMPORAL_UPDATE_PATTERNS'),
            'configuration': ('tolvera.llm.context.temporal_patterns_extended', 'TEMPORAL_CONFIGURATION_PATTERNS')
        }
        
        for context_name in context_names:
            if context_name in context_imports:
                module_path, attr_name = context_imports[context_name]
                try:
                    # Import the module and get the attribute
                    import importlib
                    module = importlib.import_module(module_path)
                    content = getattr(module, attr_name, '')
                    contexts[context_name] = content
                    logger.debug(f"Imported context '{context_name}': {len(content)} chars")
                except ImportError as e:
                    logger.warning(f"Failed to import context '{context_name}': {e}")
                    contexts[context_name] = f"# {context_name} context not available"
                except AttributeError as e:
                    logger.warning(f"Context attribute '{attr_name}' not found in {module_path}: {e}")
                    contexts[context_name] = f"# {context_name} context not available"
            else:
                logger.warning(f"Unknown context '{context_name}' requested")
                contexts[context_name] = f"# Unknown context: {context_name}"
        
        return contexts
    
    async def build_prompt_with_dynamic_context(
        self,
        description: str,
        expert_type: str = "force",
        available_states: Optional[Dict[str, List[str]]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build a comprehensive prompt with LLM-selected contexts for expert synthesis.
        
        Args:
            description: Natural language behavior description  
            expert_type: Type of expert (force, interaction, visual, temporal_update, etc.)
            available_states: Available states dictionary
            additional_context: Additional context for context selection
            
        Returns:
            Complete prompt with dynamically selected contexts
        """
        logger.info(f"Building dynamic prompt for: {description} (type: {expert_type})")
        
        # Get context selector and select relevant contexts with tracing
        selector = self._get_context_selector()
        collector = get_collector()
        
        with collector.trace_node("context_selection", "context_selection",
                                 description=description,
                                 expert_type=expert_type,
                                 additional_context=additional_context) as node:
            selection_result = await selector.select_contexts(description, expert_type, additional_context)
            selected_contexts = selection_result.selected_contexts
            
            if node:
                node.output_data = {
                    "selected_contexts": selection_result.selected_contexts,
                    "reasoning": selection_result.reasoning,
                    "context_count": len(selection_result.selected_contexts)
                }
        
        logger.info(f"Selected {len(selected_contexts)} contexts: {selected_contexts}")
        
        # Import the selected context patterns
        context_patterns = self._import_context_patterns(selected_contexts)
        
        # Build the prompt sections
        prompt_sections = []
        
        # Add the 5-element structure for synthesis prompts
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
        
        # Add selected contexts
        prompt_sections.append("\n## AVAILABLE CONTEXT\n")
        for context_name in selected_contexts:
            if context_name in context_patterns:
                prompt_sections.append(f"### {context_name.replace('_', ' ').title()}")
                prompt_sections.append(context_patterns[context_name])
                prompt_sections.append("")
        
        # Add state context if available
        if available_states:
            prompt_sections.append("### Available States")
            prompt_sections.append(self._format_state_context(available_states))
            prompt_sections.append("")
        
        # Add task specification
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
        prompt_sections.append("")
        
        combined_prompt = "\n".join(prompt_sections)
        
        logger.info(f"Built dynamic prompt: {len(combined_prompt)} chars with {len(selected_contexts)} contexts")
        logger.debug(f"Context selection reasoning: {selection_result.reasoning}")
        
        return combined_prompt
    
    def load_prompt_with_dynamic_context(
        self,
        file_path: str,
        description: str = "",
        sketch_code: str = "",
        refinement_type: str = "general",
        **kwargs
    ) -> str:
        """
        Load a prompt template file and inject dynamically selected contexts.
        
        Args:
            file_path: Template file to load
            description: Description for context selection
            sketch_code: Current sketch code for analysis
            refinement_type: Type of refinement
            **kwargs: Additional template variables
            
        Returns:
            Loaded prompt with dynamic contexts injected
        """
        logger.info(f"Loading prompt with dynamic context: {file_path}")
        
        # Select contexts for refinement with tracing
        selector = self._get_context_selector()
        collector = get_collector()
        
        with collector.trace_node("context_selection_refinement", "context_selection",
                                 description=description,
                                 refinement_type=refinement_type,
                                 file_path=file_path) as node:
            selected_contexts = selector.select_contexts_for_refinement(
                description, sketch_code, refinement_type
            )
            
            if node:
                node.output_data = {
                    "selected_contexts": selected_contexts,
                    "context_count": len(selected_contexts),
                    "refinement_type": refinement_type,
                    "method": "pattern_matching"
                }
        
        # Import the selected contexts
        context_patterns = self._import_context_patterns(selected_contexts)
        
        # Add the context patterns to kwargs for template substitution
        # Map context names to template variable names
        context_to_template_map = {
            'movement': 'movement_patterns',
            'flocking': 'flocking_patterns', 
            'taichi_fundamentals': 'taichi_fundamentals',
            'taichi_crashes': 'taichi_crashes',
            'interaction': 'interaction_patterns',
            'temporal': 'temporal_patterns',
            'drawing': 'drawing_patterns',
            'cellular': 'cellular_patterns',
            'emergent': 'emergent_patterns'
        }
        
        for context_name, pattern_content in context_patterns.items():
            # Use mapped name if available, otherwise use context name
            template_var_name = context_to_template_map.get(context_name, context_name)
            if template_var_name not in kwargs:  # Don't override explicit kwargs
                kwargs[template_var_name] = pattern_content
        
        # Load the template with context patterns injected
        prompt = self.load_prompt(file_path, **kwargs)
        
        logger.info(f"Loaded prompt with {len(selected_contexts)} dynamic contexts: {len(prompt)} chars")
        
        return prompt
    
    def _format_state_context(self, states: Dict[str, List[str]]) -> str:
        """Format available states for prompt context."""
        lines = []
        
        if 'global' in states and states['global']:
            lines.append("Global States (access with `tv.s.llm_global.field[0].state_name`):")
            for state in states['global']:
                lines.append(f"- {state}")
            lines.append("")
        
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
    
    def load_prompt(self, file_path: str, **kwargs) -> str:
        """
        Load a single prompt from a file with variable substitution.
        
        Args:
            file_path: Relative path to the prompt file from base_path
            **kwargs: Variables for string formatting
            
        Returns:
            The loaded and formatted prompt content
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            ValueError: If string formatting fails
        """
        full_path = self.base_path / file_path
        
        logger.info(f"[PROMPT_LOADER] Loading prompt file: {full_path}")
        logger.debug(f"[PROMPT_LOADER] Substitution variables provided: {list(kwargs.keys())}")
        
        try:
            # Read the file content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            raw_length = len(content)
            logger.info(f"[PROMPT_LOADER] Raw content loaded: {raw_length} chars from {file_path}")
            
            # Check for placeholders in the content
            import re
            placeholders = re.findall(r'\{(\w+)\}', content)
            if placeholders:
                logger.debug(f"[PROMPT_LOADER] Found placeholders: {placeholders}")
            
            # Apply variable substitution if kwargs provided
            if kwargs:
                logger.debug(f"[PROMPT_LOADER] Applying variable substitution...")
                content = content.format(**kwargs)
                formatted_length = len(content)
                logger.info(f"[PROMPT_LOADER] Formatted content: {formatted_length} chars (delta: {formatted_length - raw_length})")
            else:
                logger.debug(f"[PROMPT_LOADER] No variable substitution needed")
            
            # Check if any placeholders remain
            remaining_placeholders = re.findall(r'\{(\w+)\}', content)
            if remaining_placeholders:
                logger.warning(f"[PROMPT_LOADER] Unsubstituted placeholders remain: {remaining_placeholders}")
            
            logger.info(f"[PROMPT_LOADER] Successfully loaded {file_path}: {len(content)} chars")
            return content
            
        except FileNotFoundError:
            logger.error(f"[PROMPT_LOADER] ERROR: Prompt file not found: {full_path}")
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        except KeyError as e:
            logger.error(f"[PROMPT_LOADER] ERROR: Missing variable in prompt {file_path}: {e}")
            logger.error(f"[PROMPT_LOADER] Required variable: {e}, Provided: {list(kwargs.keys())}")
            raise ValueError(f"Missing variable in prompt {file_path}: {e}")
        except Exception as e:
            logger.error(f"[PROMPT_LOADER] ERROR: Failed loading prompt {file_path}: {e}")
            logger.error(f"[PROMPT_LOADER] Exception type: {type(e).__name__}")
            raise
    
    def load_multi_part_prompt(self, parts: List[str], separator: str = "\n\n", **kwargs) -> str:
        """
        Load and combine multiple prompt parts into a single prompt.
        
        Args:
            parts: List of relative file paths to prompt parts
            separator: String to join the parts with
            **kwargs: Variables for string formatting
            
        Returns:
            The combined and formatted prompt content
        """
        logger.info(f"[PROMPT_LOADER] Loading multi-part prompt with {len(parts)} parts")
        logger.debug(f"[PROMPT_LOADER] Parts to load: {parts}")
        
        prompt_parts = []
        part_sizes = []
        
        for i, part_path in enumerate(parts):
            try:
                logger.debug(f"[PROMPT_LOADER] Loading part {i+1}/{len(parts)}: {part_path}")
                part_content = self.load_prompt(part_path, **kwargs)
                prompt_parts.append(part_content)
                part_sizes.append(len(part_content))
                logger.info(f"[PROMPT_LOADER] Part {i+1} loaded: {len(part_content)} chars")
            except Exception as e:
                logger.warning(f"[PROMPT_LOADER] WARNING: Failed to load prompt part {part_path}: {e}")
                continue
        
        combined = separator.join(prompt_parts)
        logger.info(f"[PROMPT_LOADER] Combined {len(prompt_parts)}/{len(parts)} parts successfully")
        logger.info(f"[PROMPT_LOADER] Total combined size: {len(combined)} chars")
        logger.debug(f"[PROMPT_LOADER] Part sizes: {part_sizes}")
        
        return combined
    
    def prompt_exists(self, file_path: str) -> bool:
        """
        Check if a prompt file exists.
        
        Args:
            file_path: Relative path to the prompt file
            
        Returns:
            True if the file exists, False otherwise
        """
        full_path = self.base_path / file_path
        return full_path.exists()
    
    def list_prompts(self, subdirectory: Optional[str] = None) -> List[str]:
        """
        List all available prompt files.
        
        Args:
            subdirectory: Optional subdirectory to search in
            
        Returns:
            List of relative paths to prompt files
        """
        search_path = self.base_path
        if subdirectory:
            search_path = search_path / subdirectory
        
        if not search_path.exists():
            return []
        
        prompt_files = []
        for file_path in search_path.rglob("*.txt"):
            relative_path = file_path.relative_to(self.base_path)
            prompt_files.append(str(relative_path))
        
        return sorted(prompt_files)


# Global instance for convenience
_default_loader = None

def get_prompt_loader() -> PromptLoader:
    """Get the default global prompt loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader