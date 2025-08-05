import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from ..core.prompts import ContextAwarePromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class DrawingExpert:
    """Container for a drawing expert function"""
    name: str
    description: str
    code: str
    draw_order: str = "post"  # "pre" or "post" particle rendering
    requires_interaction: bool = False
    
    def to_code(self) -> str:
        """Return the complete function code"""
        return self.code


class DrawingClassifier:
    """Classifies behavior descriptions to identify drawing behaviors"""
    
    # Keywords that indicate drawing/visualization behavior
    DRAWING_KEYWORDS = [
        "draw", "draws", "drawing", "drawn",
        "trail", "trails", "streak", "streaks",
        "visualize", "visualization", "show", "display",
        "line", "lines", "circle", "circles", "rect", "rectangle",
        "color", "colors", "colored", "coloring", "colour",
        "halo", "glow", "glowing", "shine", "shining",
        "fade", "fading", "blur", "blurring",
        "ripple", "ripples", "wave", "waves",
        "mark", "marks", "marking", "marker",
        "paint", "paints", "painting",
        "render", "renders", "rendering",
        "effect", "effects", "visual",
        "opacity", "transparency", "alpha",
        "bright", "brightness", "dim", "dimming",
        "highlight", "highlighting",
        "vector", "vectors", "arrow", "arrows"
    ]
    
    # Keywords that indicate particle-particle drawing interactions
    INTERACTION_DRAWING_KEYWORDS = [
        "between", "connect", "connection", "link",
        "proximity", "near", "close", "distance",
        "collision", "collide", "impact",
        "network", "web", "mesh",
        "line between", "lines between"
    ]
    
    # Keywords for pre-particle rendering (background effects)
    PRE_RENDER_KEYWORDS = [
        "background", "behind", "under", "beneath",
        "trail", "history", "path", "trajectory"
    ]
    
    # Keywords for post-particle rendering (overlay effects)
    POST_RENDER_KEYWORDS = [
        "overlay", "over", "on top", "above",
        "highlight", "halo", "glow", "outline"
    ]
    
    @classmethod
    def classify(cls, description: str) -> Dict[str, Any]:
        """
        Classify a behavior description.
        
        Returns:
            Dict with:
                - is_drawing: bool
                - draw_order: "pre" or "post"
                - requires_interaction: bool
                - confidence: float
        """
        desc_lower = description.lower()
        
        # Check for drawing keywords
        drawing_matches = sum(1 for kw in cls.DRAWING_KEYWORDS if kw in desc_lower)
        is_drawing = drawing_matches > 0
        
        # Determine if it requires particle-particle interaction
        interaction_matches = sum(1 for kw in cls.INTERACTION_DRAWING_KEYWORDS if kw in desc_lower)
        requires_interaction = interaction_matches > 0
        
        # Determine draw order
        pre_matches = sum(1 for kw in cls.PRE_RENDER_KEYWORDS if kw in desc_lower)
        post_matches = sum(1 for kw in cls.POST_RENDER_KEYWORDS if kw in desc_lower)
        draw_order = "pre" if pre_matches > post_matches else "post"
        
        # Calculate confidence
        total_words = len(desc_lower.split())
        keyword_density = (drawing_matches + interaction_matches) / max(total_words, 1)
        confidence = min(keyword_density * 2, 1.0)  # Scale up but cap at 1.0
        
        return {
            "is_drawing": is_drawing,
            "draw_order": draw_order,
            "requires_interaction": requires_interaction,
            "confidence": confidence
        }


class DrawingKernelGenerator:
    """Generates integration kernels for drawing behaviors"""
    
    def generate(
        self,
        pre_draw_experts: List[str],
        post_draw_experts: List[str],
        interaction_draw_experts: List[str],
        expert_weights: Dict[str, float],
        tolvera_instance: Any
    ) -> str:
        """
        Generate a drawing integration kernel.
        
        Args:
            pre_draw_experts: Expert names for pre-particle drawing
            post_draw_experts: Expert names for post-particle drawing
            interaction_draw_experts: Expert names requiring particle pairs
            expert_weights: Weights for each expert
            tolvera_instance: Tolvera instance for field info
            
        Returns:
            Generated kernel code
        """
        kernel_parts = []
        
        # Header
        kernel_parts.append("@ti.kernel")
        kernel_parts.append("def apply_drawing_behaviors(tv: ti.template()):")
        kernel_parts.append("    # Drawing integration kernel")
        
        # Pre-particle drawing phase
        if pre_draw_experts:
            kernel_parts.append("\n    # Pre-particle drawing effects")
            kernel_parts.append("    for i in range(tv.pn):")
            kernel_parts.append("        if tv.p.active[i] > 0:")
            
            for expert in pre_draw_experts:
                weight = expert_weights.get(expert, 1.0)
                if weight > 0:
                    kernel_parts.append(f"            {expert}(tv.px, tv.p[i], i)")
        
        # Interaction drawing phase
        if interaction_draw_experts:
            kernel_parts.append("\n    # Interaction-based drawing")
            kernel_parts.append("    for i in range(tv.pn):")
            kernel_parts.append("        if tv.p.active[i] > 0:")
            kernel_parts.append("            for j in range(i + 1, tv.pn):")
            kernel_parts.append("                if tv.p.active[j] > 0:")
            
            for expert in interaction_draw_experts:
                weight = expert_weights.get(expert, 1.0)
                if weight > 0:
                    kernel_parts.append(f"                    {expert}(tv.px, tv.p[i], tv.p[j])")
        
        # Post-particle drawing phase
        if post_draw_experts:
            kernel_parts.append("\n    # Post-particle drawing effects")
            kernel_parts.append("    for i in range(tv.pn):")
            kernel_parts.append("        if tv.p.active[i] > 0:")
            
            for expert in post_draw_experts:
                weight = expert_weights.get(expert, 1.0)
                if weight > 0:
                    kernel_parts.append(f"            {expert}(tv.px, tv.p[i], i)")
        
        return "\n".join(kernel_parts)


class DrawingSynthesizer:
    """Synthesizes drawing expert functions from natural language"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.classifier = DrawingClassifier()
        self.prompt_builder = ContextAwarePromptBuilder()
    
    async def synthesize_drawing_expert(
        self,
        description: str,
        available_states: Dict[str, List[str]]
    ) -> DrawingExpert:
        """
        Synthesize a drawing expert from description.
        
        Args:
            description: Natural language description
            available_states: Available state fields
            
        Returns:
            DrawingExpert instance
        """
        # Classify the drawing behavior
        classification = self.classifier.classify(description)
        
        if not classification["is_drawing"]:
            raise ValueError(f"Description does not appear to be a drawing behavior: {description}")
        
        # Build comprehensive prompt using ContextAwarePromptBuilder
        base_prompt = self.prompt_builder.build_synthesis_prompt(
            description=description,
            available_states=available_states,
            constrained=False  # Direct code generation
        )
        
        # Add drawing-specific instructions
        drawing_instructions = self._get_drawing_instructions(classification)
        system_prompt = base_prompt + "\n\n" + drawing_instructions
        
        # Create specific user prompt for drawing
        user_prompt = f"Generate a drawing function for: {description}"
        
        # Generate the expert function
        response = await self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7
        )
        
        # Parse the generated code
        code = self._extract_code(response)
        expert_name = self._extract_function_name(code)
        
        return DrawingExpert(
            name=expert_name,
            description=description,
            code=code,
            draw_order=classification["draw_order"],
            requires_interaction=classification["requires_interaction"]
        )
    
    def _get_drawing_instructions(self, classification: Dict[str, Any]) -> str:
        """Get drawing-specific instructions based on classification"""
        
        if classification["requires_interaction"]:
            return """## DRAWING EXPERT REQUIREMENTS (Particle-Particle)
Generate a Taichi drawing function for particle-particle visualizations.

Function signature:
@ti.func
def draw_<descriptive_name>(px: ti.template(), p1: ti.template(), p2: ti.template()):
    # Drawing based on relationship between p1 and p2
    pass

IMPORTANT:
- Function name should start with 'draw_' (NOT 'expert_draw_')
- Access pixels with px.px.rgba[x, y] where x, y are integer coordinates
- Use p1.pos and p2.pos for particle positions
- Consider distance, species, and other particle properties
- Handle boundaries with modulo or clamp operations
- Focus on visualizing relationships, connections, and interactions
- CRITICAL: Declare ALL variables before conditional branches:
  distance = 0.0  # Default
  if condition:
      distance = calculate_distance()"""
        else:
            return f"""## DRAWING EXPERT REQUIREMENTS (Single Particle)
Generate a Taichi drawing function for single-particle visualization.

Function signature:
@ti.func
def draw_<descriptive_name>(px: ti.template(), p: ti.template(), particle_idx: ti.i32):
    # Drawing effects for individual particle
    pass

Draw order: {"PRE-particle rendering (background effects)" if classification["draw_order"] == "pre" else "POST-particle rendering (overlay effects)"}

IMPORTANT:
- Function name should start with 'draw_' (NOT 'expert_draw_')
- Access pixels with px.px.rgba[x, y] where x, y are integer coordinates
- Use p.pos for particle position, p.vel for velocity
- Cast positions to integers: px_x = ti.cast(p.pos.x, ti.i32)
- Check bounds: if 0 <= px_x < tv.x and 0 <= px_y < tv.y
- For trails: use p.ppos (previous position) if available
- For glows/halos: iterate over nearby pixels with appropriate falloff
- Use alpha blending for transparency effects
- CRITICAL: Declare ALL variables before conditional branches:
  intensity = 0.5  # Default
  if p.species == 0:
      intensity = 1.0"""
    
    def _extract_code(self, response: str) -> str:
        """Extract the function code from LLM response"""
        # Simple extraction - in production would be more robust
        lines = response.strip().split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith("@ti.func"):
                in_function = True
            if in_function:
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def _extract_function_name(self, code: str) -> str:
        """Extract function name from code"""
        import re
        match = re.search(r'def\s+(draw_\w+)', code)
        if match:
            return match.group(1)
        raise ValueError("Could not extract function name from generated code")