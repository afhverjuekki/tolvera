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
    """Generates integration for drawing behaviors"""
    
    def generate(
        self,
        pre_draw_experts: List[str],
        post_draw_experts: List[str],
        interaction_draw_experts: List[str],
        expert_weights: Dict[str, float],
        tolvera_instance: Any
    ) -> str:
        """
        Generate code to call drawing kernels in the render pipeline.
        
        Note: Drawing behaviors are implemented as kernels, not functions,
        so they're called directly in the render pipeline, not integrated.
        
        Args:
            pre_draw_experts: Kernel names for pre-particle drawing
            post_draw_experts: Kernel names for post-particle drawing
            interaction_draw_experts: Kernel names requiring particle pairs
            expert_weights: Weights for each expert
            tolvera_instance: Tolvera instance for field info
            
        Returns:
            Generated code to add to render pipeline
        """
        render_calls = []
        
        # Pre-particle drawing phase
        if pre_draw_experts:
            render_calls.append("# Pre-particle drawing effects")
            for kernel in pre_draw_experts:
                weight = expert_weights.get(kernel, 1.0)
                if weight > 0:
                    render_calls.append(f"{kernel}()  # Draw before particles")
        
        # Note: Drawing kernels should be called directly, not through an integration kernel
        # They're added to the render pipeline like:
        # @tv.render
        # def _():
        #     tv.px.clear()
        #     draw_trails()  # Pre-draw
        #     tv.p()  # Particles
        #     draw_glows()  # Post-draw
        #     return tv.px
        
        # Interaction drawing phase
        if interaction_draw_experts:
            render_calls.append("# Interaction-based drawing")
            for kernel in interaction_draw_experts:
                weight = expert_weights.get(kernel, 1.0)
                if weight > 0:
                    render_calls.append(f"{kernel}()  # Draw particle interactions")
        
        # Post-particle drawing phase
        if post_draw_experts:
            render_calls.append("# Post-particle drawing effects")
            for kernel in post_draw_experts:
                weight = expert_weights.get(kernel, 1.0)
                if weight > 0:
                    render_calls.append(f"{kernel}()  # Draw after particles")
        
        return "\n".join(render_calls)


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
            return """## DRAWING KERNEL REQUIREMENTS (Particle-Particle)
Generate a Taichi kernel for particle-particle visualizations.

Function signature:
@ti.kernel
def draw_<descriptive_name>():
    # Drawing based on relationships between particles
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            for j in range(i + 1, tv.pn):
                if tv.p.field[j].active > 0:
                    # Draw connection/interaction

IMPORTANT:
- Use @ti.kernel NOT @ti.func
- Access particles with tv.p.field[i] and tv.p.field[j]
- Use Pixels API methods:
  * tv.px.line(x1, y1, x2, y2, color) for lines
  * tv.px.circle(x, y, radius, color) for circles
  * tv.px.rect(x, y, width, height, color) for rectangles
- NEVER use px.px.rgba[x,y] - this is incorrect!
- Colors are ti.math.vec4(r, g, b, a) with values 0.0-1.0
- Cast positions to appropriate types for drawing:
  x = ti.cast(tv.p.field[i].pos[0], ti.i32)
- Handle boundaries with modulo: x % tv.x
- Example:
  tv.px.line(p1.pos[0], p1.pos[1], p2.pos[0], p2.pos[1], ti.math.vec4(1, 0, 0, 0.5))"""
        else:
            return f"""## DRAWING KERNEL REQUIREMENTS (Single Particle)
Generate a Taichi kernel for single-particle visualization.

Function signature:
@ti.kernel
def draw_<descriptive_name>():
    # Drawing effects for particles
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            p = tv.p.field[i]
            # Draw effect for particle

Draw order: {"PRE-particle rendering (background effects)" if classification["draw_order"] == "pre" else "POST-particle rendering (overlay effects)"}

IMPORTANT:
- Use @ti.kernel NOT @ti.func
- Access particles with tv.p.field[i]
- Use Pixels API methods:
  * tv.px.line(x1, y1, x2, y2, color) for trails
  * tv.px.circle(x, y, radius, color) for halos/glows
  * tv.px.rect(x, y, width, height, color) for blocks
- NEVER use px.px.rgba[x,y] - this is incorrect!
- Colors are ti.math.vec4(r, g, b, a) or ti.Vector([r, g, b, a])
- Positions can be float, internally cast by API
- Example for trails:
  tv.px.line(p.pos[0], p.pos[1], p.pos[0] - p.vel[0]*10, p.pos[1] - p.vel[1]*10, color)
- Example for glow:
  tv.px.circle(p.pos[0], p.pos[1], 10, ti.math.vec4(1, 1, 0, 0.3))"""
    
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