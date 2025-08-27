"""
Color Resolver using LLM for complex color name resolution.
"""

from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .species_manager import SpeciesInfo

# Mathematical Constants
GOLDEN_RATIO = 0.618033988749895
DEFAULT_GRAY = [0.7, 0.7, 0.7, 1.0]

# Color Intensity Modifiers
INTENSITY_MODIFIERS = {
    'light': {'factor': 0.7, 'offset': 0.3, 'operation': 'lighten'},
    'pale': {'factor': 0.7, 'offset': 0.3, 'operation': 'lighten'},
    'dark': {'factor': 0.7, 'offset': 0.0, 'operation': 'darken'},
    'bright': {'factor': 1.2, 'offset': 0.0, 'operation': 'saturate'},
    'vivid': {'factor': 1.2, 'offset': 0.0, 'operation': 'saturate'},
    'neon': {'factor': 1.0, 'offset': 0.4, 'operation': 'neon'}
}


class ColorResolver:
    """Centralized color resolution and management for the Tölvera LLM system.
    
    This class handles ALL color-related operations including:
    - Color name to RGBA conversion with LLM support for complex colors
    - Species color assignment with semantic awareness
    - Color hint extraction from natural language descriptions
    - Default color generation using golden ratio distribution
    """
    
    # Base color definitions - single source of truth
    BASE_COLORS = {
        'red': [1.0, 0.3, 0.3, 1.0],
        'blue': [0.3, 0.3, 1.0, 1.0],
        'green': [0.3, 1.0, 0.3, 1.0],
        'yellow': [1.0, 1.0, 0.3, 1.0],
        'purple': [1.0, 0.3, 1.0, 1.0],
        'magenta': [1.0, 0.3, 1.0, 1.0],  # Same as purple
        'cyan': [0.3, 1.0, 1.0, 1.0],
        'orange': [1.0, 0.6, 0.3, 1.0],
        'pink': [1.0, 0.6, 0.8, 1.0],
        'brown': [0.6, 0.4, 0.2, 1.0],
        'gray': [0.6, 0.6, 0.6, 1.0],
        'grey': [0.6, 0.6, 0.6, 1.0],  # British spelling
        'white': [1.0, 1.0, 1.0, 1.0],
        'black': [0.2, 0.2, 0.2, 1.0],
    }
    
    # Extended common colors (references base colors where possible)
    COMMON_COLORS = {
        **BASE_COLORS,
        # Additional semantic colors
        'crimson': [0.86, 0.08, 0.24, 1.0],
        'lime': [0.75, 1.0, 0.0, 1.0],
        'navy': [0.0, 0.0, 0.5, 1.0],
        'gold': [1.0, 0.84, 0.0, 1.0],
        'silver': [0.75, 0.75, 0.75, 1.0],
    }
    
    # Semantic role-based colors (references base colors)
    ROLE_COLORS = {
        'predator': BASE_COLORS['red'],
        'hunter': BASE_COLORS['red'],
        'carnivore': [0.9, 0.2, 0.2, 1.0],  # Darker red variant
        'prey': [0.3, 0.8, 0.3, 1.0],  # Slightly different green
        'food': [0.3, 0.9, 0.3, 1.0],  # Brighter green
        'plant': [0.2, 0.7, 0.2, 1.0],  # Darker green
        'grazer': BASE_COLORS['green'],
        'herbivore': [0.4, 0.8, 0.4, 1.0],  # Medium green
        'scavenger': BASE_COLORS['brown'],
        'decomposer': BASE_COLORS['gray'],
        'parasite': [0.7, 0.2, 0.7, 1.0],  # Purple variant
        'symbiont': BASE_COLORS['cyan'],
        'host': BASE_COLORS['blue'],
    }
    
    # Species term metadata with semantic behavior mappings
    SPECIES_TERMS = {
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
    
    def __init__(self, llm_client=None):
        """Initialize the ColorResolver with optional LLM client for complex color resolution.
        
        Args:
            llm_client: Optional LLM client for resolving complex color names.
                       If not provided, falls back to simple color mapping.
        """
        self.llm_client = llm_client
        self._default_colors_cache = {}  # Cache for default species colors
    
    async def resolve_color_name(self, color_name: str) -> List[float]:
        """Resolve a color name to RGBA values.
        
        Attempts resolution in the following order:
        1. Common colors dictionary
        2. Role-based colors dictionary
        3. LLM resolution for complex colors
        4. Compound color parsing (e.g., "light blue")
        5. Default gray fallback
        
        Args:
            color_name: The name of the color to resolve (e.g., "red", "lime green", "crimson")
            
        Returns:
            List of RGBA values [r, g, b, a] in 0-1 range
        """
        color_lower = color_name.lower().strip()
        
        # Try common colors first (fastest)
        if color_lower in self.COMMON_COLORS:
            return self.COMMON_COLORS[color_lower]
        
        # Try role-based colors
        if color_lower in self.ROLE_COLORS:
            return self.ROLE_COLORS[color_lower]
        
        # Try LLM for complex colors if available
        if self.llm_client:
            try:
                return await self._resolve_with_llm(color_name)
            except Exception:
                pass
        
        # Try to parse compound colors (e.g., "light blue", "dark green")
        compound_result = self._parse_compound_color(color_lower)
        if compound_result:
            return compound_result
        
        # Ultimate fallback
        return DEFAULT_GRAY
    
    def _parse_compound_color(self, color_name: str) -> Optional[List[float]]:
        """Parse compound color names like 'light blue' or 'dark green'.
        
        Args:
            color_name: Lowercase color name to parse
            
        Returns:
            RGBA values if successfully parsed, None otherwise
        """
        # Check if any base color appears in the name
        for base_color, base_rgba in self.COMMON_COLORS.items():
            if base_color in color_name:
                return self._apply_intensity_modifier(base_rgba, color_name)
        return None
    
    def _apply_intensity_modifier(self, base_color: List[float], full_name: str) -> List[float]:
        """Apply intensity modifiers to a base color.
        
        Args:
            base_color: Base RGBA color values
            full_name: Full color name potentially containing modifiers
            
        Returns:
            Modified RGBA color values
        """
        color = base_color.copy()
        
        # Check each modifier type
        for modifier_name, modifier_config in INTENSITY_MODIFIERS.items():
            if modifier_name in full_name:
                operation = modifier_config['operation']
                
                if operation == 'lighten':
                    # Mix with white
                    for i in range(3):
                        color[i] = color[i] * modifier_config['factor'] + modifier_config['offset']
                        
                elif operation == 'darken':
                    # Reduce intensity
                    for i in range(3):
                        color[i] = color[i] * modifier_config['factor']
                        
                elif operation == 'saturate':
                    # Increase saturation
                    for i in range(3):
                        color[i] = min(1.0, color[i] * modifier_config['factor'])
                        
                elif operation == 'neon':
                    # Make very bright and saturated
                    max_idx = color[:3].index(max(color[:3]))
                    color[max_idx] = 1.0
                    for i in range(3):
                        if i != max_idx:
                            color[i] = min(modifier_config['offset'], color[i])
                
                break  # Apply only the first matching modifier
        
        return color
    
    async def _resolve_with_llm(self, color_name: str) -> List[float]:
        """Use LLM to resolve complex color names to RGBA.
        
        Args:
            color_name: Complex color name to resolve
            
        Returns:
            RGBA values resolved by the LLM
            
        Raises:
            Exception: If LLM resolution fails
        """
        from pydantic_ai import Agent
        from pydantic import BaseModel, Field
        from ..debug.tracing import get_collector
        from ..prompts.prompt_loader import get_prompt_loader
        
        # Build color resolution request
        class ColorRGBA(BaseModel):
            """RGBA color values in 0-1 range."""
            color_name: str = Field(description="The color name")
            r: float = Field(description="Red component (0.0 to 1.0)", ge=0.0, le=1.0)
            g: float = Field(description="Green component (0.0 to 1.0)", ge=0.0, le=1.0)
            b: float = Field(description="Blue component (0.0 to 1.0)", ge=0.0, le=1.0)
            a: float = Field(default=1.0, description="Alpha component (0.0 to 1.0)", ge=0.0, le=1.0)
            
            def to_list(self) -> List[float]:
                """Convert to list format [r, g, b, a]."""
                return [self.r, self.g, self.b, self.a]
        
        # Setup model
        model, model_name = self._setup_llm_model()
        if not model:
            return self.COMMON_COLORS.get(color_name.lower(), DEFAULT_GRAY)
        
        # Load prompt and create agent
        loader = get_prompt_loader()
        system_prompt = loader.load_prompt("utilities/color_resolution.txt")
        agent = Agent(model, output_type=ColorRGBA, system_prompt=system_prompt)
        
        # Execute resolution with tracing
        collector = get_collector()
        with collector.trace_node("color_resolution", "color_resolution", 
                                 color_name=color_name) as resolution_node:
            try:
                result = await self._execute_llm_resolution(
                    agent, color_name, system_prompt, model_name, collector
                )
                
                if resolution_node:
                    resolution_node.output_data = {
                        "resolved_color": result,
                        "color_name": color_name
                    }
                
                return result
                
            except Exception as e:
                # Note: resolution_node will be completed with error status automatically
                # by the context manager when exception is raised
                raise
    
    def _setup_llm_model(self):
        """Setup the LLM model for color resolution.
        
        Returns:
            Tuple of (model, model_name) or (None, None) if setup fails
        """
        from .llm_factory import ModelFactory
        import os
        
        if self.llm_client:
            model = self.llm_client
            # Determine model name from client
            if hasattr(model, 'model_name'):
                model_name = model.model_name
            elif hasattr(model, '__class__'):
                model_name = model.__class__.__name__
            else:
                model_name = "custom"
            return model, model_name
        
        # Create model using factory
        default_model = os.getenv('DEFAULT_MODEL', 'gemini-2.0-flash')
        try:
            model = ModelFactory.create_model(default_model)
            return model, default_model
        except (ValueError, ImportError):
            return None, None
    
    async def _execute_llm_resolution(self, agent, color_name, system_prompt, model_name, collector):
        """Execute the LLM color resolution with tracing.
        
        Args:
            agent: Pydantic AI agent
            color_name: Color to resolve
            system_prompt: System prompt for the agent
            model_name: Name of the model being used
            collector: Trace collector for debugging
            
        Returns:
            List of RGBA values
        """
        from ..debug.tracing import LLMCallData
        from .llm_factory import ModelFactory
        
        with collector.trace_node("llm_color_resolution", "llm_call",
                                 model=model_name) as llm_node:
            user_prompt = f"Convert the color '{color_name}' to RGBA values"
            result = await agent.run(user_prompt)
            
            # Log LLM call details
            if llm_node:
                provider = ModelFactory.get_provider_for_model(model_name) if model_name != "custom" else "custom"
                llm_data = LLMCallData(
                    model=model_name,
                    provider=provider,
                    system_prompt=system_prompt.strip(),
                    user_prompt=user_prompt,
                    full_prompt=f"{system_prompt}\n\n{user_prompt}",
                    raw_response=str(result.output),
                    parsed_response={
                        "color_name": result.output.color_name,
                        "r": result.output.r,
                        "g": result.output.g,
                        "b": result.output.b,
                        "a": result.output.a,
                        "rgba_list": result.output.to_list()
                    }
                )
                llm_node.llm_call = llm_data
            
            return result.output.to_list()
    
    def extract_color_hints(self, text: str, species_names: Dict[int, str]) -> Dict[int, str]:
        """Extract color hints from description text and species names.
        
        Analyzes text for explicit color mentions and species name patterns
        to determine appropriate colors for each species.
        
        Args:
            text: Description text to analyze
            species_names: Mapping of species ID to name
            
        Returns:
            Dictionary mapping species ID to color name
        """
        import re
        hints = {}
        
        # Pattern for explicit color mentions (e.g., "species 0 is red")
        color_pattern = r'species\s+(\d+)\s+(?:is|are|should\s+be|colored?)\s+(\w+)'
        matches = re.findall(color_pattern, text, re.IGNORECASE)
        
        for species_id_str, color in matches:
            if color.lower() in self.COMMON_COLORS:
                hints[int(species_id_str)] = color.lower()
        
        # Extract colors from species names
        for species_id, name in species_names.items():
            if species_id in hints:
                continue  # Skip if already has explicit color
            
            name_lower = name.lower()
            
            # Check if name contains a color
            for color in self.COMMON_COLORS:
                if color in name_lower:
                    hints[species_id] = color
                    break
            
            # Check semantic colors from species terms
            if species_id not in hints and name_lower in self.SPECIES_TERMS:
                hints[species_id] = self.SPECIES_TERMS[name_lower]['color']
        
        return hints
    
    def get_species_color(self, species_id: int, species_info: 'SpeciesInfo') -> List[float]:
        """Get color for a specific species based on hints and defaults.
        
        Args:
            species_id: The species ID
            species_info: Species information containing color hints
            
        Returns:
            RGBA color values as list [r, g, b, a]
        """
        # Check explicit hints first
        if hasattr(species_info, 'color_hints') and species_id in species_info.color_hints:
            color_name = species_info.color_hints[species_id]
            
            # Resolve the color name
            if color_name in self.COMMON_COLORS:
                return self.COMMON_COLORS[color_name]
            elif color_name in self.ROLE_COLORS:
                return self.ROLE_COLORS[color_name]
        
        # Use default colors with good distribution
        max_species = max(8, species_id + 1)
        default_colors = self.get_default_species_colors(max_species)
        
        if species_id in default_colors:
            return default_colors[species_id]
        
        # Ultimate fallback
        return DEFAULT_GRAY
    
    def get_default_species_colors(self, num_species: int) -> Dict[int, List[float]]:
        """Generate default colors for species using golden ratio for optimal visual distinction.
        
        Uses a combination of predefined distinct colors for the first 8 species,
        then generates additional colors using the golden ratio for good distribution
        in HSV color space.
        
        Args:
            num_species: Number of species to generate colors for
            
        Returns:
            Dictionary mapping species ID (0-indexed) to RGBA values
        """
        # Check cache first
        if num_species in self._default_colors_cache:
            return self._default_colors_cache[num_species]
        
        colors = {}
        
        # Predefined distinct colors for first 8 species
        predefined = [
            [1.0, 0.3, 0.3, 1.0],  # Red
            [0.3, 0.5, 1.0, 1.0],  # Blue
            [0.3, 0.9, 0.3, 1.0],  # Green
            [1.0, 0.9, 0.3, 1.0],  # Yellow
            [0.9, 0.3, 0.9, 1.0],  # Magenta
            [0.3, 0.9, 0.9, 1.0],  # Cyan
            [1.0, 0.6, 0.3, 1.0],  # Orange
            [0.6, 0.4, 0.2, 1.0],  # Brown
        ]
        
        for i in range(num_species):
            if i < len(predefined):
                colors[i] = predefined[i]
            else:
                # Generate using golden ratio for good distribution
                colors[i] = self._generate_golden_ratio_color(i)
        
        # Cache the result
        self._default_colors_cache[num_species] = colors
        return colors
    
    def _generate_golden_ratio_color(self, index: int) -> List[float]:
        """Generate a color using golden ratio distribution in HSV space.
        
        Args:
            index: Index for color generation
            
        Returns:
            RGBA color values
        """
        import colorsys
        
        # Use golden ratio for hue distribution
        hue = (index * GOLDEN_RATIO) % 1.0
        saturation = 0.7  # Good saturation for visibility
        value = 0.9  # Good brightness for contrast
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return [rgb[0], rgb[1], rgb[2], 1.0]