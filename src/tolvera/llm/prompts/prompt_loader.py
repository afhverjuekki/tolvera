"""
Utility for loading LLM prompts from external text files.
Provides centralized prompt management with variable substitution and intelligent context selection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

from ..debug.tracing import get_collector

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
            self.base_path = Path(__file__).parent
        else:
            self.base_path = Path(base_path)
        
        # Initialize context selector (lazy loading to avoid circular imports)
        self._context_selector = None
    
    def _get_context_selector(self):
        """Lazy load the context selector to avoid circular imports."""
        if self._context_selector is None:
            from ..context.context_selector import ContextSelector
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
        
        # Import mapping - now all patterns are loaded via __init__ or library_docs
        context_imports = {
            'core_api': ('tolvera.llm.context.library_docs', 'TOLVERA_CORE_API'),
            'pixels_api': ('tolvera.llm.context.library_docs', 'TOLVERA_PIXELS_API'),
            'taichi_fundamentals': ('tolvera.llm.context', 'TAICHI_FUNDAMENTALS'),
            'taichi_crashes': ('tolvera.llm.context.library_docs', 'TAICHI_CRASH_FIXES'),
            'state_access': ('tolvera.llm.context.library_docs', 'TOLVERA_STATE_API'),
            'boundaries': ('tolvera.llm.context.context_loader', 'load_section("patterns.txt", "BOUNDARY_PATTERNS")'),
            'movement': ('tolvera.llm.context', 'MOVEMENT_PATTERNS'),
            'flocking': ('tolvera.llm.context', 'FLOCKING_PATTERNS'),
            'interaction': ('tolvera.llm.context', 'INTERACTION_PATTERNS'),
            'temporal': ('tolvera.llm.context', 'TEMPORAL_PATTERNS'),
            'cellular': ('tolvera.llm.context', 'CELLULAR_AUTOMATA_PATTERNS'),
            'emergent': ('tolvera.llm.context', 'EMERGENT_PATTERNS'),
            'drawing': ('tolvera.llm.context', 'DRAWING_PATTERNS'),
            'drawing_api': ('tolvera.llm.context', 'DRAWING_API_REFERENCE'),
            'vera_patterns': ('tolvera.llm.context', 'VERA_PATTERNS'),
            'vera_interactions': ('tolvera.llm.context', 'INTERACTION_PATTERNS_VERA'),
            'species_interactions': ('tolvera.llm.context', 'SPECIES_INTERACTION_PATTERNS'),
            'alife_patterns': ('tolvera.llm.context', 'ARTIFICIAL_LIFE_PATTERNS'),
            'iml_patterns': ('tolvera.llm.context', 'IML_PATTERNS'),
            'evolution': ('tolvera.llm.context', 'EVOLUTION_PATTERNS'),
            'ecosystem': ('tolvera.llm.context', 'ECOSYSTEM_PATTERNS'),
            'morphogenesis': ('tolvera.llm.context', 'MORPHOGENETIC_PATTERNS'),
            'swarm': ('tolvera.llm.context', 'SWARM_INTELLIGENCE'),
            'initialization': ('tolvera.llm.context.context_loader', 'load_section("initialization_patterns.txt", "main")'),
            'species_initialization': ('tolvera.llm.context.context_loader', 'load_section("initialization_patterns.txt", "SPECIES_INITIALIZATION_PATTERNS")'),
            'temporal_updates': ('tolvera.llm.context.context_loader', 'load_section("temporal_patterns_extended.txt", "main")'),
            'temporal_dynamics': ('tolvera.llm.context.context_loader', 'load_section("temporal_dynamics.txt", "main")'),
            'temporal_examples': ('tolvera.llm.context.context_loader', 'load_section("temporal_dynamics.txt", "TEMPORAL_UPDATE_EXAMPLES")'),
            'temporal_patterns_extended': ('tolvera.llm.context.context_loader', 'load_section("temporal_patterns_extended.txt", "main")'),
            'configuration': ('tolvera.llm.context.context_loader', 'load_section("temporal_patterns_extended.txt", "TEMPORAL_CONFIGURATION_PATTERNS")')
        }
        
        for context_name in context_names:
            if context_name in context_imports:
                module_path, attr_name = context_imports[context_name]
                try:
                    # Import the module and get the attribute
                    import importlib
                    module = importlib.import_module(module_path)
                    
                    # Check if attr_name is a function call (for dynamic loading)
                    if '(' in attr_name:
                        # Execute the function call
                        from ..context.context_loader import load_section
                        content = eval(attr_name)
                    else:
                        # Get the attribute normally
                        content = getattr(module, attr_name, '')
                    
                    contexts[context_name] = content
                except ImportError:
                    contexts[context_name] = f"# {context_name} context not available"
                except AttributeError:
                    contexts[context_name] = f"# {context_name} context not available"
            else:
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
        Build a comprehensive prompt with base context + LLM-selected supplementary contexts.
        
        Args:
            description: Natural language behavior description  
            expert_type: Type of expert (force, interaction, visual, temporal_update, etc.)
            available_states: Available states dictionary
            additional_context: Additional context for context selection
            
        Returns:
            Complete prompt with base context and dynamically selected supplementary contexts
        """
        
        # Import base context (always included)
        from ..context.library_docs import get_base_context
        base_context = get_base_context()
        
        # Get context selector and select supplementary contexts with tracing
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
        
        
        # Import the selected supplementary context patterns
        context_patterns = self._import_context_patterns(selected_contexts)
        
        # Build the prompt sections
        prompt_sections = []
        
        # Add base context FIRST (core APIs)
        prompt_sections.append("## BASE CONTEXT - TÖLVERA CORE APIs")
        prompt_sections.append(base_context)
        prompt_sections.append("\n" + "="*80 + "\n")
        
        # Load the 5-element structure from external file
        five_element_structure = self.load_prompt("synthesis/five_element_structure.txt")
        prompt_sections.append(five_element_structure)
        
        # Add selected supplementary contexts
        if selected_contexts:
            prompt_sections.append("\n## SUPPLEMENTARY CONTEXT (Selected Based on Behavior)\n")
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
        
        
        return combined_prompt
    
    async def load_prompt_with_dynamic_context_async(
        self,
        file_path: str,
        description: str = "",
        sketch_code: str = "",
        refinement_type: str = "general",
        **kwargs
    ) -> str:
        """
        Load a prompt template file and inject dynamically selected contexts using LLM.
        
        Args:
            file_path: Template file to load
            description: Description for context selection
            sketch_code: Current sketch code for analysis
            refinement_type: Type of refinement
            **kwargs: Additional template variables
            
        Returns:
            Loaded prompt with dynamic contexts injected
        """
        
        # Select contexts for refinement with tracing
        selector = self._get_context_selector()
        collector = get_collector()
        
        with collector.trace_node("context_selection_refinement", "context_selection",
                                 description=description,
                                 refinement_type=refinement_type,
                                 file_path=file_path) as node:
            selected_contexts = await selector.select_contexts_for_refinement(
                description, sketch_code, refinement_type
            )
            
            if node:
                node.output_data = {
                    "selected_contexts": selected_contexts,
                    "context_count": len(selected_contexts),
                    "refinement_type": refinement_type,
                    "method": "llm"
                }
        
        # Import the selected contexts
        context_patterns = self._import_context_patterns(selected_contexts)
        
        # Add the context patterns to kwargs for template substitution
        # Map context names to template variable names
        context_to_template_map = {
            'core_api': 'core_api',
            'pixels_api': 'pixels_api',
            'state_access': 'state_access',
            'taichi_fundamentals': 'taichi_fundamentals',
            'taichi_crashes': 'taichi_crashes',
            'movement': 'movement_patterns',
            'flocking': 'flocking_patterns', 
            'interaction': 'interaction_patterns',
            'temporal': 'temporal_patterns',
            'drawing': 'drawing_patterns',
            'drawing_api': 'drawing_api',
            'cellular': 'cellular_patterns',
            'emergent': 'emergent_patterns',
            'boundaries': 'boundaries',
            'vera_patterns': 'vera_patterns',
            'vera_interactions': 'vera_interactions',
            'species_interactions': 'species_interactions',
            'alife_patterns': 'alife_patterns',
            'iml_patterns': 'iml_patterns',
            'evolution': 'evolution',
            'ecosystem': 'ecosystem',
            'morphogenesis': 'morphogenesis',
            'swarm': 'swarm',
            'initialization': 'initialization',
            'species_initialization': 'species_initialization',
            'temporal_updates': 'temporal_updates',
            'temporal_dynamics': 'temporal_dynamics',
            'temporal_examples': 'temporal_examples',
            'temporal_patterns_extended': 'temporal_patterns_extended',
            'configuration': 'configuration'
        }
        
        # Ensure critical contexts are always available for templates that expect them
        # Even if not selected, provide empty content to avoid template errors
        critical_contexts = ['taichi_fundamentals', 'taichi_crashes', 'movement', 'flocking']
        for critical in critical_contexts:
            template_var = context_to_template_map.get(critical, critical)
            if template_var not in kwargs:
                # Import the critical context even if not selected
                critical_pattern = self._import_context_patterns([critical])
                if critical_pattern and critical in critical_pattern:
                    kwargs[template_var] = critical_pattern[critical]
                else:
                    kwargs[template_var] = f"# {critical} context not selected by LLM"
        
        for context_name, pattern_content in context_patterns.items():
            # Use mapped name if available, otherwise use context name
            template_var_name = context_to_template_map.get(context_name, context_name)
            if template_var_name not in kwargs:  # Don't override explicit kwargs
                kwargs[template_var_name] = pattern_content
        
        # Load the template with context patterns injected
        prompt = self.load_prompt(file_path, **kwargs)
        
        
        return prompt
    
    def load_prompt_with_dynamic_context(
        self,
        file_path: str,
        description: str = "",
        sketch_code: str = "",
        refinement_type: str = "general",
        **kwargs
    ) -> str:
        """
        Synchronous wrapper for backward compatibility.
        Runs the async version properly handling existing event loops.
        
        Args:
            file_path: Template file to load
            description: Description for context selection
            sketch_code: Current sketch code for analysis
            refinement_type: Type of refinement
            **kwargs: Additional template variables
            
        Returns:
            Loaded prompt with dynamic contexts injected
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # We're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.load_prompt_with_dynamic_context_async(
                        file_path, description, sketch_code, refinement_type, **kwargs
                    )
                )
                return future.result()
        except RuntimeError:
            # No event loop, we can create one
            return asyncio.run(self.load_prompt_with_dynamic_context_async(
                file_path, description, sketch_code, refinement_type, **kwargs
            ))
    
    def get_expert_type_guidance(self, expert_type: Optional[str]) -> str:
        """
        Get expert-type specific guidance for state analysis.
        Moved from synthesizer.py for better separation of concerns.
        
        Args:
            expert_type: Type of expert (force, visual, interaction, etc.)
            
        Returns:
            Expert-type specific guidance string
        """
        if not expert_type:
            return ""
        
        # Map synonyms to canonical names
        synonym_map = {
            'drawing': 'visual',
            'drawing_interaction': 'visual',
        }
        canonical_type = synonym_map.get(expert_type, expert_type)
        
        # Load guidance from external file
        try:
            guidance = self.load_prompt("synthesis/expert_type_guidance.txt")
            
            # Extract the specific section for this expert type
            import re
            pattern = rf"## {canonical_type.replace('_', ' ').title()}.*?(?=##|$)"
            match = re.search(pattern, guidance, re.DOTALL | re.IGNORECASE)
            
            if match:
                return match.group(0).strip()
            else:
                return ""
        except Exception:
            return ""
    
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
        
        
        try:
            # Read the file content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply variable substitution if kwargs provided
            if kwargs:
                content = content.format(**kwargs)
            return content
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        except KeyError as e:
            raise ValueError(f"Missing variable in prompt {file_path}: {e}")
        except Exception:
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
        prompt_parts = []
        
        for part_path in parts:
            try:
                part_content = self.load_prompt(part_path, **kwargs)
                prompt_parts.append(part_content)
            except Exception:
                continue
        
        combined = separator.join(prompt_parts)
        
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