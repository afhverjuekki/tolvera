"""
Drawing Classifier Module - Business logic for classifying and managing drawing behaviors
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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


class DrawingInstructionBuilder:
    """Builds drawing instructions for LLM prompts based on classification"""
    
    @staticmethod
    def build_drawing_instructions(classification: Dict[str, Any]) -> Dict[str, str]:
        """
        Build drawing instruction parameters based on classification.
        
        Args:
            classification: Classification result from DrawingClassifier
            
        Returns:
            Dictionary with template parameters for drawing instructions
        """
        interaction_type = "Particle-Particle" if classification["requires_interaction"] else "Single Particle"
        description = "particle-particle visualizations" if classification["requires_interaction"] else "single-particle visualization"
        
        if classification["requires_interaction"]:
            function_signature = """@ti.kernel
def draw_<descriptive_name>():
    # Drawing based on relationships between particles
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            for j in range(i + 1, tv.pn):
                if tv.p.field[j].active > 0:
                    # Draw connection/interaction"""
            additional_access = " and tv.p.field[j]"
            example_code = "tv.px.line(p1.pos[0], p1.pos[1], p2.pos[0], p2.pos[1], ti.math.vec4(1, 0, 0, 0.5))"
            draw_order_info = ""
        else:
            function_signature = """@ti.kernel
def draw_<descriptive_name>():
    # Drawing effects for particles
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            p = tv.p.field[i]
            # Draw effect for particle"""
            additional_access = ""
            example_code = """Example for trails:
  tv.px.line(p.pos[0], p.pos[1], p.pos[0] - p.vel[0]*10, p.pos[1] - p.vel[1]*10, color)
- Example for glow:
  tv.px.circle(p.pos[0], p.pos[1], 10, ti.math.vec4(1, 1, 0, 0.3))"""
            draw_order_info = f'Draw order: {"PRE-particle rendering (background effects)" if classification["draw_order"] == "pre" else "POST-particle rendering (overlay effects)"}'
        
        return {
            "interaction_type": interaction_type,
            "description": description,
            "function_signature": function_signature,
            "additional_access": additional_access,
            "example_code": example_code,
            "draw_order_info": draw_order_info
        }