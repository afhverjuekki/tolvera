"""Species management for Tolvera LLM system.

This module provides centralized species detection, analysis, and configuration
for particle behaviors. It serves as the authoritative source for species-related
operations while delegating color resolution to ColorResolver.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from .color_resolver import ColorResolver


@dataclass
class SpeciesInfo:
    """Information about species detected in a behavior description."""
    species_ids: List[int]
    species_names: Dict[int, str]  # ID to semantic name mapping
    interaction_pairs: List[Tuple[int, int]]
    species_behaviors: Dict[int, List[str]]  # Species-specific behaviors
    requires_all_species: bool
    total_count: int
    color_hints: Dict[int, str]  # Species ID to color hint


class SpeciesManager:
    """Manages species detection, analysis, and initialization for particle systems.
    
    This class is responsible for:
    - Analyzing natural language descriptions to detect species
    - Determining species counts and relationships
    - Building species configurations for behavior synthesis
    - Providing species context for behavior synthesis
    """
    
    # Regex patterns for species detection (it's faster to not have to call the LLM for this)
    SPECIES_ID_PATTERN = re.compile(r'species\s+(\d+)', re.IGNORECASE)
    COLOR_SPECIES_PATTERNS = [
        re.compile(r'(red|blue|green|yellow|purple|orange|pink|cyan|brown|gray|white|black)\s+(?:species|particles?|agents?|predators?|prey)', re.IGNORECASE),
        re.compile(r'(red|blue|green|yellow|purple|orange|pink|cyan|brown|gray|white|black)\s+(?:ones?|group|type)', re.IGNORECASE),
    ]
    INTERACTION_PATTERNS = [
        re.compile(r'species\s+(\d+)\s+(?:chase|chases|hunts?|attracts?|repels?|flees?\s+from|avoids?)\s+species\s+(\d+)', re.IGNORECASE),
        re.compile(r'species\s+(\d+)\s+(?:and|&)\s+species\s+(\d+)\s+(?:interact|repel|attract)', re.IGNORECASE),
    ]
    
    def __init__(self, tolvera_instance):
        """Initialize species manager with Tolvera instance.
        
        Args:
            tolvera_instance: The Tolvera instance for accessing particle system
        """
        self.tv = tolvera_instance
        self.color_resolver = ColorResolver()
        self.current_species_info = None
    
    def analyze_description(self, description: str) -> SpeciesInfo:
        """Analyze a description to extract species information.
        
        Args:
            description: Natural language description of particle behavior
            
        Returns:
            SpeciesInfo containing detected species data
        """
        desc_lower = description.lower()
        
        # Extract species data in a single pass
        species_ids = self._extract_species_ids(desc_lower)
        species_names = self._extract_species_names(desc_lower)
        interaction_pairs = self._extract_interaction_pairs(desc_lower, species_ids, species_names)
        species_behaviors = self._extract_species_behaviors(description, species_ids, species_names)
        requires_all = self._detect_all_species_requirement(desc_lower)
        total_count = self._determine_total_species(species_ids, species_names, interaction_pairs, requires_all)
        
        # Delegate color extraction to ColorResolver
        color_hints = self.color_resolver.extract_color_hints(desc_lower, species_names)
        
        # Ensure species IDs are populated
        if not species_ids and total_count > 0:
            species_ids = list(range(total_count))
        
        return SpeciesInfo(
            species_ids=species_ids,
            species_names=species_names,
            interaction_pairs=interaction_pairs,
            species_behaviors=species_behaviors,
            requires_all_species=requires_all,
            total_count=total_count,
            color_hints=color_hints
        )
    
    def _extract_species_ids(self, text: str) -> List[int]:
        """Extract explicit species IDs mentioned in text."""
        matches = self.SPECIES_ID_PATTERN.findall(text)
        return sorted(set(int(m) for m in matches))
    
    def _extract_species_names(self, text: str) -> Dict[int, str]:
        """Extract species names from description, prioritizing colors over roles."""
        names = {}
        next_id = 0
        seen_terms = set()
        
        # First: color-based species (highest priority)
        for pattern in self.COLOR_SPECIES_PATTERNS:
            for match in pattern.finditer(text):
                color = match.group(1).lower()
                if color not in seen_terms:
                    names[next_id] = f"{color}_species"
                    seen_terms.add(color)
                    next_id += 1
        
        # Second: ecosystem roles from ColorResolver's authoritative list
        for term in self.color_resolver.SPECIES_TERMS:
            if term in text and term not in seen_terms:
                names[next_id] = term
                seen_terms.add(term)
                next_id += 1
        
        # Special case: food (often mentioned but not in standard terms)
        if 'food' in text and 'food' not in seen_terms:
            names[next_id] = 'food'
            next_id += 1
        
        return names
    
    def _extract_interaction_pairs(self, text: str, species_ids: List[int], 
                                   species_names: Dict[int, str]) -> List[Tuple[int, int]]:
        """Extract interaction pairs between species."""
        pairs = set()
        
        # Direct species ID interactions
        for pattern in self.INTERACTION_PATTERNS:
            for match in pattern.finditer(text):
                s1, s2 = int(match.group(1)), int(match.group(2))
                pairs.add((s1, s2))
        
        # Semantic interactions (predator-prey)
        if species_names:
            predator_ids = [id for id, name in species_names.items() if 'predator' in name or 'hunter' in name]
            prey_ids = [id for id, name in species_names.items() if 'prey' in name or 'food' in name]
            
            for pred in predator_ids:
                for prey in prey_ids:
                    pairs.add((pred, prey))
        
        # Multi-species mutual interactions
        if 'repel each other' in text or 'repel one another' in text:
            all_ids = species_ids or list(range(len(species_names)))
            for i in range(len(all_ids)):
                for j in range(i + 1, len(all_ids)):
                    pairs.add((all_ids[i], all_ids[j]))
        
        return list(pairs)
    
    def _extract_species_behaviors(self, text: str, species_ids: List[int], 
                                    species_names: Dict[int, str]) -> Dict[int, List[str]]:
        """Extract behaviors specific to each species."""
        behaviors = {}
        
        # Extract behaviors for explicitly mentioned species IDs
        for species_id in species_ids:
            pattern = re.compile(rf'species\s+{species_id}[^.!?]*', re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                behaviors[species_id] = matches
        
        # Add semantic behaviors from ColorResolver's species terms
        for species_id, species_name in species_names.items():
            if species_name in self.color_resolver.SPECIES_TERMS:
                term_info = self.color_resolver.SPECIES_TERMS[species_name]
                if species_id not in behaviors:
                    behaviors[species_id] = []
                behaviors[species_id].append(f"{species_name}: {term_info['behavior']}")
        
        return behaviors
    
    def _detect_all_species_requirement(self, text: str) -> bool:
        """Check if description implies all species should be affected."""
        indicators = ['all species', 'every species', 'each species', 
                     'all particles', 'every particle']
        return any(indicator in text for indicator in indicators)
    
    def _determine_total_species(self, species_ids: List[int], species_names: Dict[int, str],
                                 interaction_pairs: List[Tuple[int, int]], 
                                 requires_all: bool) -> int:
        """Determine total number of species from various indicators."""
        # Collect all evidence for species count
        counts = []
        
        if species_ids:
            counts.append(max(species_ids) + 1)
        
        if species_names:
            counts.append(len(species_names))
        
        if interaction_pairs:
            all_in_pairs = set()
            for s1, s2 in interaction_pairs:
                all_in_pairs.update([s1, s2])
            if all_in_pairs:
                counts.append(max(all_in_pairs) + 1)
        
        # Return the maximum count found, or sensible defaults
        if counts:
            return max(counts)
        return 1 if requires_all else 2  # Default: 1 for generic, 2 for interactions
    
    def get_species_context_for_prompts(self, species_info: SpeciesInfo) -> str:
        """Generate context string for LLM prompts about species configuration.
        
        Args:
            species_info: Species information
            
        Returns:
            Formatted context string for prompts
        """
        lines = [
            "## Species Configuration",
            f"Total species: {species_info.total_count}",
            f"Species IDs in use: {list(range(species_info.total_count))}\n"
        ]
        
        if species_info.species_names:
            lines.append("### Species Names and Roles:")
            for sid, name in species_info.species_names.items():
                lines.append(f"- Species {sid}: {name}")
            lines.append("")
        
        if species_info.species_behaviors:
            lines.append("### Species-Specific Behaviors:")
            for sid, behaviors in species_info.species_behaviors.items():
                name = species_info.species_names.get(sid, f"Species {sid}")
                lines.append(f"- {name}:")
                for behavior in behaviors:
                    lines.append(f"  - {behavior}")
            lines.append("")
        
        if species_info.interaction_pairs:
            lines.append("### Species Interactions:")
            for s1, s2 in species_info.interaction_pairs:
                n1 = species_info.species_names.get(s1, f"Species {s1}")
                n2 = species_info.species_names.get(s2, f"Species {s2}")
                lines.append(f"- {n1} interacts with {n2}")
            lines.append("")
        
        # Code generation hints
        lines.extend([
            "### Code Generation Hints:",
            "- Use `species` parameter to check particle species",
            "- Use conditional logic for species-specific behaviors"
        ])
        
        if species_info.total_count > 1:
            lines.extend([
                "- Consider different forces/behaviors for different species",
                f"- Species IDs range from 0 to {species_info.total_count - 1}"
            ])
        
        return "\n".join(lines)
    
    async def build_species_configuration(self, species_info, species_names_list):
        """Build a complete species configuration with resolved colors.
        
        Args:
            species_info: Species information from decomposer
            species_names_list: List of SpeciesNameMapping objects
            
        Returns:
            SpeciesConfiguration object
        """
        from .data_models import SpeciesConfiguration, SpeciesColorMapping
        
        # Resolve colors using ColorResolver
        colors = {}
        if hasattr(species_info, 'species_color_descriptions') and species_info.species_color_descriptions:
            for mapping in species_info.species_color_descriptions:
                species_id = mapping.species_id
                color_desc = mapping.color_description if hasattr(mapping, 'color_description') else 'gray'
                try:
                    colors[species_id] = await self.color_resolver.resolve_color_name(color_desc)
                except Exception:
                    colors[species_id] = [0.7, 0.7, 0.7, 1.0]
        
        return SpeciesConfiguration(
            species_ids=list(range(species_info.total_count)),
            species_names=species_names_list,
            colors=[
                SpeciesColorMapping(species_id=sid, rgba=rgba)
                for sid, rgba in colors.items()
            ] if colors else None
        )