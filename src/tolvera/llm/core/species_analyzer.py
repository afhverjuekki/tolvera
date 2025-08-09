import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpeciesInfo:
    species_ids: List[int]
    species_names: Dict[int, str]  # ID to semantic name mapping
    interaction_pairs: List[Tuple[int, int]]
    species_behaviors: Dict[int, List[str]]  # Species-specific behaviors
    requires_all_species: bool
    total_count: int
    color_hints: Dict[int, str]  # Species ID to color hint


class SpeciesAnalyzer:
    
    def __init__(self, color_resolver=None):
        self.color_resolver = color_resolver
        self.species_terms = {
            'predator': {'color': 'red', 'behavior': 'hunting'},
            'prey': {'color': 'green', 'behavior': 'fleeing'},
            'hunter': {'color': 'red', 'behavior': 'hunting'},
            'grazer': {'color': 'green', 'behavior': 'grazing'},
            'plant': {'color': 'green', 'behavior': 'stationary'},
            'carnivore': {'color': 'red', 'behavior': 'hunting'},
            'herbivore': {'color': 'green', 'behavior': 'grazing'},
            'parasite': {'color': 'purple', 'behavior': 'attaching'},
            'host': {'color': 'blue', 'behavior': 'normal'},
            'symbiont': {'color': 'cyan', 'behavior': 'cooperative'},
            'scavenger': {'color': 'brown', 'behavior': 'seeking'},
            'decomposer': {'color': 'gray', 'behavior': 'breaking down'},
        }
        
        self.color_mappings = {
            'red': [1.0, 0.3, 0.3, 1.0],
            'blue': [0.3, 0.3, 1.0, 1.0],
            'green': [0.3, 1.0, 0.3, 1.0],
            'yellow': [1.0, 1.0, 0.3, 1.0],
            'purple': [1.0, 0.3, 1.0, 1.0],
            'cyan': [0.3, 1.0, 1.0, 1.0],
            'orange': [1.0, 0.6, 0.3, 1.0],
            'pink': [1.0, 0.6, 0.8, 1.0],
            'brown': [0.6, 0.4, 0.2, 1.0],
            'gray': [0.6, 0.6, 0.6, 1.0],
            'white': [1.0, 1.0, 1.0, 1.0],
            'black': [0.2, 0.2, 0.2, 1.0],
        }
        
    def analyze(self, description: str) -> SpeciesInfo:
        desc_lower = description.lower()
        
        species_ids = self._extract_species_ids(desc_lower)
        
        species_names = self._extract_species_names(desc_lower)
        
        interaction_pairs = self._detect_interaction_pairs(desc_lower, species_ids, species_names)
        
        species_behaviors = self._extract_species_behaviors(description, species_ids, species_names)
        
        requires_all = self._check_requires_all_species(desc_lower)
        
        total_count = self._determine_species_count(species_ids, species_names, interaction_pairs, requires_all)
        
        color_hints = self._extract_color_hints(desc_lower, species_names)
        
        if not species_ids and (species_names or total_count > 0):
            species_ids = list(range(total_count))
        
        logger.info(f"Species analysis complete: {total_count} species detected")
        
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
        pattern = r'species\s+(\d+)'
        matches = re.findall(pattern, text)
        return sorted({int(m) for m in matches})
    
    def _extract_species_names(self, text: str) -> Dict[int, str]:
        names = {}
        next_id = 0
        
        # First check for color-based species (priority over ecosystem roles)
        # Updated pattern to catch more variations
        color_patterns = [
            r'(red|blue|green|yellow|purple|orange|pink|cyan|brown|gray|white|black)\s+(?:species|particles?|agents?|predators?|prey)',
            r'(red|blue|green|yellow|purple|orange|pink|cyan|brown|gray|white|black)\s+(?:ones?|group|type)',
            r'the\s+(red|blue|green|yellow|purple|orange|pink|cyan|brown|gray|white|black)\s+(?:species)?'
        ]
        
        seen_colors = set()
        for pattern in color_patterns:
            color_matches = re.findall(pattern, text)
            for color in color_matches:
                if color not in seen_colors:
                    names[next_id] = f"{color}_species"
                    seen_colors.add(color)
                    next_id += 1
        
        # Check for ecosystem roles (but don't duplicate if color already assigned)
        for term in self.species_terms:
            if term in text or f"{term}s" in text:
                # Check if this term is already associated with a color
                already_exists = False
                for existing_name in names.values():
                    if term in existing_name:
                        already_exists = True
                        break
                
                if not already_exists:
                    names[next_id] = term
                    next_id += 1
        
        # Special case: if we see "food" mentioned, add it as a species
        if 'food' in text.lower() and 'food' not in str(names.values()):
            names[next_id] = 'food'
            next_id += 1
        
        return names
    
    def _detect_interaction_pairs(
        self, 
        text: str, 
        species_ids: List[int], 
        species_names: Dict[int, str]
    ) -> List[Tuple[int, int]]:
        pairs = []
        
        # Direct species interaction patterns
        interaction_patterns = [
            r'species\s+(\d+)\s+(?:chase|chases|hunts?|attracts?|repels?|flees?\s+from|avoids?)\s+species\s+(\d+)',
            r'species\s+(\d+)\s+(?:and|&)\s+species\s+(\d+)\s+(?:interact|repel|attract)',
            r'species\s+(\d+)\s+(?:is|are)\s+attracted\s+to\s+species\s+(\d+)',
        ]
        
        for pattern in interaction_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                s1, s2 = int(match[0]), int(match[1])
                pairs.append((s1, s2))
        
        # Named species interactions
        if species_names:
            # Predator-prey relationships
            if any('predator' in name for name in species_names.values()) and \
               any('prey' in name for name in species_names.values()):
                pred_ids = [id for id, name in species_names.items() if 'predator' in name]
                prey_ids = [id for id, name in species_names.items() if 'prey' in name]
                for pred in pred_ids:
                    for prey in prey_ids:
                        pairs.append((pred, prey))
        
        # Generic multi-species interactions
        if 'repel each other' in text or 'repel one another' in text:
            # All species repel all others
            all_ids = species_ids if species_ids else list(range(len(species_names)))
            for i in range(len(all_ids)):
                for j in range(i + 1, len(all_ids)):
                    pairs.append((all_ids[i], all_ids[j]))
        
        return list(set(pairs))  # Remove duplicates
    
    def _extract_species_behaviors(
        self, 
        text: str, 
        species_ids: List[int], 
        species_names: Dict[int, str]
    ) -> Dict[int, List[str]]:
        behaviors = {}
        
        for species_id in species_ids:
            species_behaviors = []
            
            # Find sentences mentioning this species
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                if f'species {species_id}' in sentence.lower():
                    # Extract the behavior description
                    behavior = sentence.strip()
                    if behavior:
                        species_behaviors.append(behavior)
            
            if species_behaviors:
                behaviors[species_id] = species_behaviors
        
        # Add behaviors for named species
        for species_id, species_name in species_names.items():
            if species_name in self.species_terms:
                default_behavior = self.species_terms[species_name]['behavior']
                if species_id not in behaviors:
                    behaviors[species_id] = []
                behaviors[species_id].append(f"{species_name} exhibits {default_behavior} behavior")
        
        return behaviors
    
    def _check_requires_all_species(self, text: str) -> bool:
        all_species_indicators = [
            'all species',
            'every species',
            'each species',
            'particles',  # Generic mention often means all
            'all particles',
            'every particle',
        ]
        
        return any(indicator in text for indicator in all_species_indicators)
    
    def _determine_species_count(
        self, 
        species_ids: List[int], 
        species_names: Dict[int, str],
        interaction_pairs: List[Tuple[int, int]],
        requires_all: bool
    ) -> int:
        counts = []
        
        # From explicit IDs
        if species_ids:
            counts.append(max(species_ids) + 1)
        
        # From named species
        if species_names:
            counts.append(len(species_names))
        
        # From interaction pairs
        if interaction_pairs:
            all_species_in_pairs = set()
            for s1, s2 in interaction_pairs:
                all_species_in_pairs.add(s1)
                all_species_in_pairs.add(s2)
            if all_species_in_pairs:
                counts.append(max(all_species_in_pairs) + 1)
        
        # Default based on context
        if not counts:
            if requires_all:
                return 1  # Single species if talking about all particles generically
            else:
                return 2  # Default to 2 for interesting interactions
        
        return max(counts)
    
    def _extract_color_hints(self, text: str, species_names: Dict[int, str]) -> Dict[int, str]:
        hints = {}
        
        # From explicit color mentions
        color_pattern = r'species\s+(\d+)\s+(?:is|are|should\s+be|colored?)\s+(\w+)'
        matches = re.findall(color_pattern, text)
        for species_id, color in matches:
            if color in self.color_mappings:
                hints[int(species_id)] = color
        
        # From species names
        for species_id, name in species_names.items():
            # Check if name contains a color
            for color in self.color_mappings:
                if color in name:
                    hints[species_id] = color
                    break
            
            # Check semantic colors
            if name in self.species_terms and species_id not in hints:
                hints[species_id] = self.species_terms[name]['color']
        
        return hints
    
    def get_color_for_species(self, species_id: int, species_info: SpeciesInfo) -> List[float]:
        # Check explicit hints
        if species_id in species_info.color_hints:
            color_name = species_info.color_hints[species_id]
            
            # Use simple mappings for now (async resolution happens elsewhere)
            if color_name in self.color_mappings:
                return self.color_mappings[color_name]
        
        # Use ColorResolver for defaults if available
        if self.color_resolver:
            from ..core.color_resolver import ColorResolver
            if not isinstance(self.color_resolver, ColorResolver):
                self.color_resolver = ColorResolver()
            defaults = self.color_resolver.get_default_species_colors(max(8, species_id + 1))
            if species_id in defaults:
                return defaults[species_id]
        
        # Fallback to hardcoded defaults
        default_colors = [
            [1.0, 0.3, 0.3, 1.0],  # Red
            [0.3, 0.3, 1.0, 1.0],  # Blue
            [0.3, 1.0, 0.3, 1.0],  # Green
            [1.0, 1.0, 0.3, 1.0],  # Yellow
            [1.0, 0.3, 1.0, 1.0],  # Magenta
            [0.3, 1.0, 1.0, 1.0],  # Cyan
            [1.0, 0.6, 0.3, 1.0],  # Orange
            [0.6, 0.4, 0.2, 1.0],  # Brown
        ]
        
        if species_id < len(default_colors):
            return default_colors[species_id]
        
        # Generate distinct color for high species counts
        import colorsys
        hue = (species_id * 0.618033988749895) % 1.0  # Golden ratio
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        return [rgb[0], rgb[1], rgb[2], 1.0]