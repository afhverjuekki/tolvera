"""
Tölvera code generation module.

This module handles generation of Tölvera Python code from sketch configurations
using Jinja2 templates. 

Including is also functionality for converting color names
to RGBA values, using a lookup table and LLM-based color mapping when needed.
This can be difficult for LLMs to handle when using natural language so
these mapping are used to reduce successive calls to the LLM and increase
the chances for successfully rendering a script.
"""

import jinja2
from jinja2.sandbox import SandboxedEnvironment
from pathlib import Path
import json
import ollama
from typing import Optional, Dict, List
from .definitions import SketchConfig, BackgroundBehavior, ParticleShape
from .llm import LLM

# Lookup table for common colors -> RGBA (0.0-1.0) (saves time on processing color choices)
KNOWN_COLORS_RGBA = {
    "red":     [1.0, 0.0, 0.0, 1.0], "green":   [0.0, 1.0, 0.0, 1.0],
    "blue":    [0.0, 0.0, 1.0, 1.0], "yellow":  [1.0, 1.0, 0.0, 1.0],
    "cyan":    [0.0, 1.0, 1.0, 1.0], "magenta": [1.0, 0.0, 1.0, 1.0],
    "white":   [1.0, 1.0, 1.0, 1.0], "black":   [0.0, 0.0, 0.0, 1.0],
    "grey":    [0.5, 0.5, 0.5, 1.0], "gray":    [0.5, 0.5, 0.5, 1.0],
    "orange":  [1.0, 0.647, 0.0, 1.0], "purple": [0.5, 0.0, 0.5, 1.0],
    "brown":   [0.647, 0.165, 0.165, 1.0], "pink": [1.0, 0.753, 0.796, 1.0],
    "lime":    [0.0, 1.0, 0.0, 1.0], "teal":    [0.0, 0.502, 0.502, 1.0],
}
DEFAULT_RGBA = [0.8, 0.8, 0.8, 1.0]  # Default to grey if color lookup fails

# --- Jinja Setup ---
TEMPLATE_DIR = Path(__file__).parent / "templates"
if not TEMPLATE_DIR.is_dir():
    alt_template_dir = Path("./tolvera/llm/templates")
    if alt_template_dir.is_dir():
        TEMPLATE_DIR = alt_template_dir
    else:
        raise FileNotFoundError(
            f"Jinja template directory not found. Checked: {TEMPLATE_DIR.resolve()}, {alt_template_dir.resolve()}"
        )

template_loader = jinja2.FileSystemLoader(searchpath=str(TEMPLATE_DIR))
template_env = SandboxedEnvironment(loader=template_loader, trim_blocks=True, lstrip_blocks=True)

FLOCK_TEMPLATE_NAME = "template_flock.py.j2"
SLIME_TEMPLATE_NAME = "template_slime.py.j2"


def _get_rgba_from_llm(color_name: str, llm_model_name: str) -> list[float] | None:
    """
    Asks LLM for normalized RGBA values for an unknown color name.
    Returns [r, g, b, a] list (0.0-1.0) or None if failed.
    """
    prompt = (
        f"Provide the normalized RGBA values (each component between 0.0 and 1.0, alpha=1.0) "
        f"for the color name '{color_name}'. "
        f"Respond ONLY with a valid JSON list containing exactly four float numbers, like [0.1, 0.2, 0.3, 1.0]. "
        f"Do not include any other text, markdown, or explanation."
    )
    
    try:
        response = ollama.chat(
            model=llm_model_name,
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'temperature': 0.1}
        )
        
        if not response or 'message' not in response or 'content' not in response['message']:
            return None
            
        json_string = response['message']['content']
        
        cleaned_json_string = json_string.strip().strip('```json').strip('```').strip()
        if not cleaned_json_string:
            return None

        parsed_data = json.loads(cleaned_json_string)
        
        if (isinstance(parsed_data, list) and # Validate the parsed data for the color; it must be a list of 4 numbers
                len(parsed_data) == 4 and 
                all(isinstance(x, (int, float)) for x in parsed_data)):
                
            # Make sure to normalize values (assuming model might give 0-255)
            normalized_rgba = [
                max(0.0, min(1.0, float(x)/255.0)) if float(x) > 1.0 
                else max(0.0, min(1.0, float(x))) 
                for x in parsed_data
            ]
            normalized_rgba[3] = 1.0  # alpha should just be 1.0
            
            return normalized_rgba
        return None
            
    except Exception:
        return None


def generate_code_from_sketch_config(
    config: SketchConfig,
    llm_module: Optional[LLM] = None
    ) -> str:
    """
    Generates Tölvera Python code from a SketchConfig object using Jinja templates.
    
    Args:
        config: SketchConfig object with the sketch configuration
        llm_module: Used for color conversion (if None, will skip unknown colors)
        
    Returns:
        str: Generated Python code for the sketch
    """
    if config.sketch_type == "flock":
        template_name = FLOCK_TEMPLATE_NAME
    elif config.sketch_type == "slime":
        template_name = SLIME_TEMPLATE_NAME
    else:
        raise ValueError(f"Unsupported sketch_type '{config.sketch_type}' in SketchConfig")

    try:
        template = template_env.get_template(template_name)
    except jinja2.TemplateNotFound:
        raise FileNotFoundError(f"Template '{template_name}' not found in {TEMPLATE_DIR.resolve()}")
    except Exception as e:
        raise RuntimeError(f"Error loading template '{template_name}': {e}")

    context = {}
    species_rgba_map: Dict[int, List[float]] = {}  # Map species index to [r,g,b,a] list

    vera_modules = []
    if config.sketch_type == "flock":
        vera_modules.append("flock")
    elif config.sketch_type == "slime":
        vera_modules.append("slime")
    
    context['vera_modules'] = vera_modules

    llm_model_name = "qwen2.5:latest"  # Default fallback
    if llm_module is not None:
        llm_model_name = llm_module.ollama_model
    
    for sp_config in config.species_configs:
        color_name_raw = sp_config.color if sp_config.color else "random" # Default to 'random' if color is None or empty string
        color_name = color_name_raw.lower().strip().replace(" ", "_") 

        if color_name != "random":
            rgba = _get_color_rgba(color_name, color_name_raw, llm_module, llm_model_name)
            species_rgba_map[sp_config.species_index] = rgba

    context['species_rgba_map'] = species_rgba_map
    context['BackgroundBehavior'] = BackgroundBehavior
    context['ParticleShape'] = ParticleShape

    try:
        return template.render(config=config, **context)
    except jinja2.TemplateError as e:
        raise jinja2.TemplateError(f"Error rendering template '{template_name}' (Line: {e.lineno}): {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during template rendering: {e}")

def _get_color_rgba(
    color_name: str, 
    color_name_raw: str, 
    llm_module: Optional[LLM], 
    llm_model_name: str
) -> List[float]:
    """
    Get RGBA values for a color name, using the lookup table first,
    then querying the LLM if available, and falling back to default (grey) if needed.
    
    Args:
        color_name: Normalized color name
        color_name_raw: Original color name
        llm_module: LLM module instance for unknown colors
        llm_model_name: Name of LLM model to use
        
    Returns:
        List of RGBA values [r, g, b, a] (0.0-1.0)
    """
    rgba = KNOWN_COLORS_RGBA.get(color_name)
    
    if rgba:
        return rgba  # Use cached value
    
    if llm_module is not None:
        rgba = _get_rgba_from_llm(color_name_raw, llm_model_name) # Ask LLM if not a match
        
        if rgba:
            KNOWN_COLORS_RGBA[color_name] = rgba  # Cache the result for future use
            return rgba
    
    return DEFAULT_RGBA.copy()  # Use default if nothing else works