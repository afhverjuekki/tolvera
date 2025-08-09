"""Color Resolver using LLM for complex color name resolution."""

import logging
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ColorRGBA(BaseModel):
    """RGBA color values in 0-1 range."""
    color_name: str = Field(description="The color name (e.g., 'lime green', 'aqua', 'crimson')")
    r: float = Field(description="Red component (0.0 to 1.0)", ge=0.0, le=1.0)
    g: float = Field(description="Green component (0.0 to 1.0)", ge=0.0, le=1.0)
    b: float = Field(description="Blue component (0.0 to 1.0)", ge=0.0, le=1.0)
    a: float = Field(default=1.0, description="Alpha component (0.0 to 1.0)", ge=0.0, le=1.0)
    
    def to_list(self) -> List[float]:
        """Convert to list format [r, g, b, a]."""
        return [self.r, self.g, self.b, self.a]


class ColorExtractionResult(BaseModel):
    """Result of extracting colors from description."""
    species_colors: Dict[int, ColorRGBA] = Field(
        description="Map of species ID to resolved color"
    )
    unresolved_species: List[int] = Field(
        default_factory=list,
        description="Species IDs without explicit color mentions"
    )


class ColorResolver:
    """Resolves complex color names to RGBA values using LLM."""
    
    # Fallback colors for common terms
    COMMON_COLORS = {
        'red': [1.0, 0.2, 0.2, 1.0],
        'blue': [0.2, 0.4, 1.0, 1.0],
        'green': [0.2, 0.8, 0.2, 1.0],
        'yellow': [1.0, 0.9, 0.2, 1.0],
        'purple': [0.8, 0.2, 0.8, 1.0],
        'orange': [1.0, 0.6, 0.2, 1.0],
        'cyan': [0.2, 0.8, 0.8, 1.0],
        'magenta': [1.0, 0.2, 1.0, 1.0],
        'pink': [1.0, 0.6, 0.8, 1.0],
        'brown': [0.6, 0.4, 0.2, 1.0],
        'gray': [0.6, 0.6, 0.6, 1.0],
        'grey': [0.6, 0.6, 0.6, 1.0],
        'white': [1.0, 1.0, 1.0, 1.0],
        'black': [0.2, 0.2, 0.2, 1.0],
    }
    
    # Semantic role-based colors
    ROLE_COLORS = {
        'predator': [1.0, 0.3, 0.3, 1.0],  # Red
        'hunter': [1.0, 0.3, 0.3, 1.0],
        'carnivore': [0.9, 0.2, 0.2, 1.0],
        'prey': [0.3, 0.8, 0.3, 1.0],  # Green
        'food': [0.3, 0.9, 0.3, 1.0],
        'plant': [0.2, 0.7, 0.2, 1.0],
        'herbivore': [0.4, 0.8, 0.4, 1.0],
        'scavenger': [0.6, 0.4, 0.2, 1.0],  # Brown
        'decomposer': [0.5, 0.5, 0.5, 1.0],  # Gray
        'parasite': [0.7, 0.2, 0.7, 1.0],  # Purple
        'symbiont': [0.3, 0.8, 0.8, 1.0],  # Cyan
    }
    
    def __init__(self, llm_client=None):
        """Initialize with optional LLM client for complex color resolution."""
        self.llm_client = llm_client
    
    async def resolve_color_name(self, color_name: str) -> List[float]:
        """
        Resolve a color name to RGBA values.
        
        First checks common colors, then uses LLM for complex colors.
        """
        color_lower = color_name.lower().strip()
        
        # Check common colors first
        if color_lower in self.COMMON_COLORS:
            return self.COMMON_COLORS[color_lower]
        
        # Check role-based colors
        if color_lower in self.ROLE_COLORS:
            return self.ROLE_COLORS[color_lower]
        
        # Use LLM for complex colors
        if self.llm_client:
            try:
                rgba = await self._resolve_with_llm(color_name)
                return rgba
            except Exception as e:
                logger.warning(f"LLM color resolution failed for '{color_name}': {e}")
        
        # Fallback: try to extract base color from compound name
        for base_color in self.COMMON_COLORS:
            if base_color in color_lower:
                return self._modify_color_intensity(
                    self.COMMON_COLORS[base_color], 
                    color_lower
                )
        
        # Ultimate fallback
        logger.warning(f"Could not resolve color '{color_name}', using default")
        return [0.7, 0.7, 0.7, 1.0]  # Gray default
    
    def _modify_color_intensity(self, base_color: List[float], full_name: str) -> List[float]:
        """Modify color intensity based on modifiers like 'light', 'dark', 'bright'."""
        color = base_color.copy()
        
        if 'light' in full_name or 'pale' in full_name:
            # Lighten by mixing with white
            for i in range(3):
                color[i] = color[i] * 0.7 + 0.3
        elif 'dark' in full_name:
            # Darken by reducing intensity
            for i in range(3):
                color[i] = color[i] * 0.7
        elif 'bright' in full_name or 'vivid' in full_name:
            # Increase saturation
            max_val = max(color[:3])
            if max_val > 0:
                for i in range(3):
                    color[i] = min(1.0, color[i] * 1.2)
        elif 'neon' in full_name:
            # Make very bright and saturated
            max_idx = color[:3].index(max(color[:3]))
            color[max_idx] = 1.0
            for i in range(3):
                if i != max_idx:
                    color[i] = min(0.4, color[i])
        
        return color
    
    async def _resolve_with_llm(self, color_name: str) -> List[float]:
        """Use LLM to resolve complex color names to RGBA."""
        from pydantic_ai import Agent
        from pydantic_ai.models.gemini import GeminiModel
        import os
        
        # Use provided LLM client or create one
        if self.llm_client:
            model = self.llm_client
        else:
            # Create Gemini model (requires API key)
            if not os.getenv('GEMINI_API_KEY'):
                # Fallback to simple resolution
                logger.warning(f"No GEMINI_API_KEY found, using fallback for '{color_name}'")
                return self.COMMON_COLORS.get(color_name.lower(), [0.7, 0.7, 0.7, 1.0])
            model = GeminiModel("gemini-2.0-flash")
        
        agent = Agent(
            model,
            output_type=ColorRGBA,
            system_prompt="""You are a color expert. Convert color names to RGBA values.
            
            Rules:
            - Output red, green, blue as floats from 0.0 to 1.0
            - Alpha is always 1.0 unless transparency is mentioned
            - Be accurate with color names (e.g., 'lime green' is bright green with yellow tint)
            - Consider modifiers like 'light', 'dark', 'bright', 'pale', 'neon'
            
            Examples:
            - "red" -> r=1.0, g=0.2, b=0.2
            - "lime green" -> r=0.5, g=1.0, b=0.2
            - "aqua" -> r=0.0, g=1.0, b=1.0
            - "crimson" -> r=0.86, g=0.08, b=0.24
            - "coral" -> r=1.0, g=0.5, b=0.31
            - "turquoise" -> r=0.25, g=0.88, b=0.82
            """
        )
        
        result = await agent.run(f"Convert the color '{color_name}' to RGBA values")
        return result.output.to_list()
    
    def get_default_species_colors(self, num_species: int) -> Dict[int, List[float]]:
        """Generate default colors for species using golden ratio for distinction."""
        colors = {}
        
        # Predefined distinct colors for first few species
        defaults = [
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
            if i < len(defaults):
                colors[i] = defaults[i]
            else:
                # Generate using golden ratio for good distribution
                import colorsys
                hue = (i * 0.618033988749895) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                colors[i] = [rgb[0], rgb[1], rgb[2], 1.0]
        
        return colors