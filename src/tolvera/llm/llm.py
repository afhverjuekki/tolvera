"""
Tölvera Module for Large Language Model interaction.

This modules takes care of the generation and modifications of Tölvera sketch 
configurations by interacting with a Large Language Model (LLM) via the 
Ollama library.

It takes natural language descriptions from the user, combines them with 
prompting (including rules, schema details, and examples loaded 
from the 'prompts/' directory), and sends requests to the specified model.

This module then processes the LLM's JSON response, performs cleaning and 
attempts auto-correction for common formatting errors. This also validates 
the output JSON against the Pydantic models defined in definitions.py 
to maintain and adhere to structural correctness before returning 
the validated configuration object.
"""

import ollama
import json
import logging
from typing import Optional, Dict, Any
from pydantic import ValidationError
import textwrap
import traceback
from pathlib import Path
import time
from .definitions import SketchConfig

PROMPT_DIR = Path(__file__).parent / "prompts"
logger = logging.getLogger("tv.llm")

class LLM:
    """
    Tölvera module for handling interaction with local LLMs (via Ollama)
    to process natural language descriptions and generate Tölvera sketch configurations
    for different sketch types (flock and slime for now).
    """
    def __init__(self, **kwargs):
        """
        Initializes the LLM module.

        Args:
            **kwargs: Configuration options.
                ollama_model (str): The name of the Ollama model to use.
        """
        self.ollama_model: str = kwargs.get("ollama_model", "qwen2.5:latest")

        try:
            if not PROMPT_DIR.is_dir():
                 raise FileNotFoundError(f"Prompt directory not found at {PROMPT_DIR.resolve()}")
            self.system_prompt = self._build_system_prompt()
            self._check_ollama_connection()
        except FileNotFoundError as e:
            logger.error(f"Could not load prompt example files: {e}")
            raise
        except Exception as e:
            logger.error(f"Error building system prompt: {e}")
            raise

    def _check_ollama_connection(self):
        """ Checks if the Ollama server is running by listing local models. """
        try:
            ollama.list()
        except Exception:
            logger.warning("Could not connect to Ollama server.")

    def _load_prompt_examples(self, filename: str) -> str:
        """Loads content from a specific file in the PROMPT_DIR."""
        file_path = PROMPT_DIR / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Required prompt example file not found: {file_path.resolve()}")
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
             raise IOError(f"Error reading prompt example file {file_path.resolve()}: {e}")

    def _build_system_prompt(self) -> str:
        """Builds the system prompt by combining base prompt with examples."""
        flock_examples_text = self._load_prompt_examples("prompt_flock.txt")
        slime_examples_text = self._load_prompt_examples("prompt_slime.txt")

        required_list = self._get_required_fields_list()

        prompt_header = textwrap.dedent(f"""
        You are an assistant that generates configuration JSON for the Tölvera library.
        Your ONLY task is to output a single, valid JSON object based on the user's request and the strict rules below.

        --- CRITICAL FORMAT VIOLATIONS TO AVOID ---
        ❌ NEVER use these incorrect field names:
        ❌ NEVER use 'name' (use 'sketch_name' instead)
        ❌ NEVER use 'particles' (use 'particle_count' instead) 
        ❌ NEVER use 'species' (use 'species_configs' instead)
        ❌ NEVER use 'id' (use 'species_index' instead)
        ❌ NEVER use hex color codes like '#FF0000' (use color STRING NAMES like 'red' instead)
        ❌ NEVER put 'particle_shape' at the top level (it MUST be inside 'render_config')
        ❌ NEVER organize species with 'count'; put everything in 'species_configs' list
        ❌ NEVER create nested structures like this:
        {{
        "particle_count": {{
            "count": 400,
            "species": [...]
        }}
        }}

        ✅ ALWAYS include a 'render_config' object with the required fields.
        ✅ ALWAYS use color names as strings like "red", "blue", "green", "orange", etc.
        ✅ ALWAYS follow the exact field structure shown in examples.
        ✅ ALWAYS use flat, separate fields at the top level like this:
        {{
        "particle_count": 400,
        "species_configs": [...]
        }}

        --- CRITICAL RESPONSE REQUIREMENTS ---
        1.  **OUTPUT RAW JSON ONLY:** Respond ONLY with the JSON object. NO markdown, NO explanations.
        2.  **EXACT FIELD NAMES:** You MUST use the EXACT field names specified (e.g., `sketch_name`, `particle_count`, `species_count`, `species_configs`, `species_index`, `color`, `render_config`, etc.). DO NOT use `name`, `particles`, `species`, `id`.
        3.  **`sketch_type`:** MUST be set to "flock" or "slime".
        4.  **CONDITIONAL CONFIGS:** Include `flock_config` (object or null) if type is "flock", set `slime_config` to `null`. Include `slime_config` (object or null) if type is "slime", set `flock_config` to `null`. The relevant config object MUST be present.
        5.  **REQUIRED FIELDS:** MUST include fields: {required_list}. `render_config` is ALWAYS required.
        6.  **`species_configs` STRUCTURE:**
            * MUST be a list named `species_configs` with EXACTLY `species_count` items.
            * Each item MUST have `species_index` (integer 0 to `species_count - 1`).
            * Each item MUST have `color`. The value MUST be a **string** representing a common color name (e.g., "red", "lime green", "dark blue", "orange") OR the string "random" OR the JSON literal `null`.
            * DO NOT use hex codes (#FF0000) or RGB objects {{r:1, g:0, b:0}} for the `color` field value. ONLY use color NAMES as strings.
            * Include optional slime parameters (e.g., `slime_sense_angle`) ONLY if `sketch_type` is "slime" and specified by user.
        7.  **`render_config` STRUCTURE:** MUST be an object named `render_config` containing `background_behavior`, `diffuse_rate` (float or `null`), and `particle_shape` ("circle" or "point").
        8.  **STRICT ADHERENCE:** Follow all rules precisely. Double-check field names and value formats. Your ONLY output is the JSON.
        """)

        schema_overview = textwrap.dedent("""
        --- SCHEMA OVERVIEW ---
        ```json
        {{
          "sketch_name": "string",             // ✅ CORRECT: Use 'sketch_name', not 'name'!
          "sketch_type": "'flock' | 'slime'",  // Must be "flock" or "slime" 
          "particle_count": integer,           // ✅ CORRECT: Use 'particle_count', not 'particles'!
          "species_count": integer,            // Number of species (e.g., 1, 2, 3)
          // ... other required global fields ...
          "flock_config": {{...}} | null,      // Required if sketch_type: 'flock'
          "slime_config": {{...}} | null,      // Required if sketch_type: 'slime'
          "species_configs": [                 // ✅ CORRECT: Use 'species_configs', not 'species'!
            {{ 
              "species_index": int,            // ✅ CORRECT: Use 'species_index', not 'id'!
              "color": "string|null",          // ✅ CORRECT: Use color names as strings ("red"), NOT hex codes
              ... 
            }}
          ],
          "render_config": {{                  // ✅ REQUIRED object
            "background_behavior": "diffuse|clear|none",
            "diffuse_rate": float | null,
            "particle_shape": "circle" | "point" // ✅ CORRECT: 'particle_shape' goes here, not at top level
          }}
        }}
        ```
        (Examples below show the full required structure - follow them exactly)
        """)

        prompt_footer = textwrap.dedent("""
        --- REMINDER OF CRITICAL FORMAT RULES ---
        1. NEVER use hex codes (#FF0000) for colors - use string names like "red", "blue", etc.
        2. NEVER use 'name', 'species', 'particles', or 'id' as field names
        3. 'render_config' is ALWAYS required as an object with correct fields
        4. The required structure is NOT optional - it MUST be followed exactly
        
        Now, process the user request below. Generate ONLY the raw JSON object conforming strictly to the rules and examples provided. Pay close attention to field names and use STRING names for the 'color' field.
        """)

        full_prompt = "\n\n".join([
            prompt_header.strip(),
            schema_overview.strip(),
            flock_examples_text.strip(), 
            slime_examples_text.strip(), 
            prompt_footer.strip()
        ])

        return full_prompt

    def _get_required_fields_list(self) -> str:
        """Extract required fields from schema and format them for the prompt."""
        try:
            try: 
                schema_info = SketchConfig.model_json_schema(ref_template="{model}")
            except AttributeError: 
                schema_info = SketchConfig.schema() 
                
            required_fields = schema_info.get('required', [])
            required_fields = [f for f in required_fields if f not in ('flock_config', 'slime_config', 'species_configs')]
            required_list = ", ".join([f"`{req}`" for req in required_fields])
            
            if not required_list: 
                required_list = "(Core fields like `sketch_name`, `sketch_type`, etc.)"
                
            return required_list
            
        except Exception as e:
            logger.warning(f"Could not generate schema summary. Error: {e}")
            return "(`sketch_name`, `sketch_type`, `particle_count`, `species_count`, `render_config`, etc.)"

    def _correct_common_json_mistakes(self, json_str: str) -> str:
        """Attempt to fix common format errors in the LLM's JSON output."""
        try:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                return json_str  # If it's invalid JSON, just return it unchanged
            
            fixed_data = {}
            
            # Fix sketch_name
            if "name" in data and "sketch_name" not in data:
                fixed_data["sketch_name"] = data.pop("name")
                logger.info("Auto-fixing: Changed 'name' to 'sketch_name'")
            else:
                fixed_data["sketch_name"] = data.get("sketch_name", "auto_generated_sketch")
                
            fixed_data["sketch_type"] = data.get("sketch_type", "flock")
            
            particle_count = None
            if "particle_count" in data:
                if isinstance(data["particle_count"], dict) and "count" in data["particle_count"]:
                    particle_count = data["particle_count"]["count"]
                    logger.info(f"Auto-fixing: Extracted particle_count={particle_count} from nested object")
                elif isinstance(data["particle_count"], int):
                    particle_count = data["particle_count"]
            elif "particles" in data:
                particle_count = data["particles"]
                logger.info("Auto-fixing: Changed 'particles' to 'particle_count'")
                
            # Set a default if we couldn't extract it
            fixed_data["particle_count"] = particle_count if particle_count is not None else 500
            
            species_configs = self._extract_species_configs(data)
            species_count = len(species_configs) if species_configs else 1
            
            fixed_data["species_count"] = species_count
            fixed_data["species_configs"] = species_configs
            
            # Handle render_config
            fixed_data["render_config"] = self._extract_render_config(data)
            
            # Handle module-specific configs (flock or slime)
            self._set_module_configs(fixed_data, data)
            
            # Add default values for other required fields
            fixed_data["window_width"] = data.get("window_width", 1920)
            fixed_data["window_height"] = data.get("window_height", 1080)
            fixed_data["global_speed"] = data.get("global_speed", 1.0)
            fixed_data["num_substeps"] = data.get("num_substeps", 1)
            
            logger.info("Auto-fixing complete")
            return json.dumps(fixed_data, indent=2)
            
        except Exception as e:
            logger.warning(f"Error when trying to fix JSON: {e}")
            return json_str

    def _extract_species_configs(self, data: Dict[str, Any]) -> list:
        """Extract species configurations from various potential locations in data."""
        species_configs = []
        
        if "species_configs" in data:
            return data["species_configs"]
            
        nested_species = None
        if isinstance(data.get("particle_count"), dict) and "species" in data["particle_count"]:
            nested_species = data["particle_count"]["species"]
            logger.info("Auto-fixing: Found species data in nested 'particle_count.species'")
        elif "species" in data:
            nested_species = data["species"]
            logger.info("Auto-fixing: Found species data in 'species'")
            
        # Process nested species if found
        if nested_species:
            logger.info(f"Auto-fixing: Converting {len(nested_species)} nested species to proper species_configs")
            for i, sp in enumerate(nested_species):
                species_index = sp.get("id", i)
                color = sp.get("color", "random")
                
                # Convert hex colors to names if needed (this is problem for some LLMs)
                if isinstance(color, str) and color.startswith("#"):
                    color = self._convert_hex_to_color_name(color)
                
                species_configs.append({
                    "species_index": species_index,
                    "color": color
                })
        
        # Ensure we have at least one species config
        if not species_configs:
            species_configs.append({
                "species_index": 0,
                "color": "random"
            })
            
        return species_configs

    def _convert_hex_to_color_name(self, hex_color: str) -> str:
        """Convert hex color code to a color name."""
        color_map = {
            "#FF0000": "red", "#00FF00": "green", "#0000FF": "blue",
            "#FFFF00": "yellow", "#FFA500": "orange", "#800080": "purple",
            "#FFC0CB": "pink", "#A52A2A": "brown", "#808080": "gray",
            "#FFFFFF": "white", "#000000": "black"
        }
        
        mapped_color = color_map.get(hex_color.upper())
        if mapped_color:
            logger.info(f"Auto-fixing: Converted hex color '{hex_color}' to name '{mapped_color}'")
            return mapped_color
        else:
            logger.info(f"Auto-fixing: Unknown hex color '{hex_color}', using 'random'")
            return "random"

    def _extract_render_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize render_config."""
        render_config = data.get("render_config", {})
        if not render_config:
            render_config = {}
            
        particle_shape = "circle"  # Default
        
        nested_species = None
        if isinstance(data.get("particle_count"), dict) and "species" in data["particle_count"]:
            nested_species = data["particle_count"]["species"]
        elif "species" in data:
            nested_species = data["species"]
            
        if nested_species and len(nested_species) > 0 and "shape" in nested_species[0]:
            shape = nested_species[0].get("shape")
            if shape == "line":
                particle_shape = "point"  # Map "line" to "point" for fallback based on vera module
                logger.info("Auto-fixing: Converted 'line' shape to 'point'")
            elif shape in ["point", "circle"]:
                particle_shape = shape
        
        if "particle_shape" in data:
            shape = data["particle_shape"]
            if shape == "line":
                particle_shape = "point"
                logger.info("Auto-fixing: Converted 'line' shape to 'point'")
            elif shape in ["point", "circle"]:
                particle_shape = shape
            logger.info("Auto-fixing: Moved 'particle_shape' to render_config")
        
        render_config["particle_shape"] = render_config.get("particle_shape", particle_shape)
        if "background_behavior" not in render_config:
            render_config["background_behavior"] = "diffuse"
        if "diffuse_rate" not in render_config and render_config["background_behavior"] == "diffuse":
            render_config["diffuse_rate"] = 0.95
        
        return render_config

    def _set_module_configs(self, fixed_data: Dict[str, Any], data: Dict[str, Any]):
        """Set the appropriate module configuration (flock or slime)."""
        if fixed_data["sketch_type"] == "flock":
            if "flock_config" not in data or data["flock_config"] is None:
                fixed_data["flock_config"] = {
                    "module_name": "flock",
                    "rules": None
                }
            else:
                fixed_data["flock_config"] = data["flock_config"]
            
            # Always set slime_config to null for flock
            fixed_data["slime_config"] = None
            
        elif fixed_data["sketch_type"] == "slime":
            if "slime_config" not in data or data["slime_config"] is None:
                fixed_data["slime_config"] = {
                    "module_name": "slime",
                    "evaporate_rate": 0.95,
                    "trail_brightness": 1.0
                }
            else:
                fixed_data["slime_config"] = data["slime_config"]
            
            # Always set flock_config to null for slime
            fixed_data["flock_config"] = None

    def generate_sketch_config_from_prompt(
            self,
            user_input: str,
            current_config_json: Optional[str] = None
        ) -> Optional[SketchConfig]:
        """
        Takes natural language input, queries the LLM using the built system prompt,
        validates the JSON structure, and returns the validated SketchConfig object.
        Includes debugging output for prompts sent to the LLM and auto-correction of JSON.
        """

        if current_config_json:
            mode = "Modifying"
            user_message_content = f"""
            You are modifying an existing Tölvera sketch configuration.
            Here is the CURRENT configuration JSON:
            ```json
            {current_config_json}
            ```
            Apply the following modification request: "{user_input}"

            Generate the NEW, COMPLETE configuration JSON object reflecting the request.
            Ensure the output strictly follows all structural rules outlined in the system prompt (especially field names and the STRING format for 'color').
            Output ONLY the raw JSON.
            """
        else:
            mode = "Generating initial"
            user_message_content = f"""
            Generate a complete Tölvera sketch configuration JSON based on the following request: "{user_input}"

            Determine the `sketch_type` ("flock" or "slime").
            Ensure the output strictly follows all structural rules outlined in the system prompt (especially field names and the STRING format for 'color').
            Output ONLY the raw JSON.
            """
        logger.info(f"{mode} sketch config for request.")

        # LLM Call
        try:
            messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_message_content.strip()}
            ]

            logger.info(f"Sending request to Ollama model: {self.ollama_model}...")
            response = ollama.chat(
                model=self.ollama_model,
                messages=messages,
                format='json',
                options={'temperature': 0.1}  # Keep low temperature (we want predictable things)
            )

            # Response checking
            if not response or 'message' not in response or 'content' not in response['message']: 
                logger.error("Invalid response structure.")
                return None
                
            json_output_string = response['message']['content']
            logger.debug(f"Raw JSON response received: {json_output_string}")
            
            # Clean up JSON string
            cleaned_json_string = self._clean_json_string(json_output_string)
            if not cleaned_json_string:
                logger.error("LLM returned empty response after cleaning.")
                return None
            
            # Apply auto-correction
            logger.info("Attempting auto-correction of JSON format...")
            original_json = cleaned_json_string
            cleaned_json_string = self._correct_common_json_mistakes(cleaned_json_string)
            if original_json != cleaned_json_string:
                logger.info("Auto-correction applied successfully")
                
            # Validate with Pydantic
            logger.info("Validating configuration structure using Pydantic...")
            try:
                validated_config = SketchConfig.model_validate_json(cleaned_json_string) 
            except AttributeError:
                validated_config = SketchConfig.parse_raw(cleaned_json_string) 
            logger.info("Pydantic validation successful!")
            return validated_config

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON from LLM: {json_err}")
            if 'json_output_string' in locals():
                logger.debug(f"LLM Output was: {json_output_string}")
            return None

        except ValidationError as e:
            logger.error("Pydantic Validation Error: The LLM's JSON does not conform to the SketchConfig structure.")

            try:
                error_details = json.dumps(e.errors(), indent=2)
                logger.debug(f"Validation Errors: {error_details}")
            except Exception as inner_e:
                logger.debug(f"Could not format detailed Pydantic errors ({type(inner_e).__name__}): {e}")

            # Try advanced auto-correction as last resort
            if 'cleaned_json_string' in locals():
                return self._attempt_advanced_json_fix(cleaned_json_string, user_input)
            return None

        except Exception as e:
            logger.error(f"An unexpected error occurred: {type(e).__name__} - {e}")
            if 'mode' in locals():
                logger.debug(f"Request details - Mode: {mode}, Model: {self.ollama_model}")
            traceback.print_exc()
            return None

    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string by removing markdown code blocks and other formatting."""
        cleaned_str = json_str.strip()
        
        # Remove markdown code blocks
        if cleaned_str.startswith("```") and cleaned_str.endswith("```"):
            first_newline = cleaned_str.find('\n')
            cleaned_str = cleaned_str[first_newline+1:-3].strip() if first_newline != -1 else cleaned_str[3:-3].strip()
        elif cleaned_str.startswith("```json"):
            cleaned_str = cleaned_str[7:].strip()
            cleaned_str = cleaned_str[:-3].strip() if cleaned_str.endswith("```") else cleaned_str
        elif cleaned_str.startswith("```"):
            cleaned_str = cleaned_str[3:].strip()
            cleaned_str = cleaned_str[:-3].strip() if cleaned_str.endswith("```") else cleaned_str
            
        return cleaned_str

    def _attempt_advanced_json_fix(self, json_str: str, user_input: str) -> Optional[SketchConfig]:
        """Attempt advanced JSON fixing as a last resort."""
        logger.info("Attempting advanced auto-correction of JSON...")
        try:
            data = json.loads(json_str)
            
            # Add user input to help with color extraction
            data["user_input"] = user_input
            
            corrected = {
                "sketch_name": data.get("sketch_name", data.get("name", f"auto_fixed_sketch_{int(time.time())}")),
                "sketch_type": data.get("sketch_type", "flock"),
                "particle_count": data.get("particle_count", data.get("particles", 200)),
                "species_count": data.get("species_count", len(data.get("species_configs", data.get("species", [])))),
                "window_width": data.get("window_width", 1920),
                "window_height": data.get("window_height", 1080),
                "global_speed": data.get("global_speed", 1.0),
                "num_substeps": data.get("num_substeps", 1),
                "flock_config": {"module_name": "flock", "rules": None} if data.get("sketch_type") != "slime" else None,
                "slime_config": {"module_name": "slime", "evaporate_rate": 0.95, "trail_brightness": 1.0} if data.get("sketch_type") == "slime" else None,
                "render_config": {
                    "background_behavior": "diffuse",
                    "diffuse_rate": 0.95,
                    "particle_shape": "circle"
                }
            }
            
            # Process species configs
            species_configs = []
            species_count = corrected["species_count"]
            source_configs = data.get("species_configs", [])
            if not source_configs and "species" in data:
                for i, sp in enumerate(data["species"]):
                    color = "random"
                    if "color" in sp:
                        color_value = sp["color"]
                        if isinstance(color_value, str) and color_value.startswith("#"):
                            color = "red" if "red" in user_input.lower() else "blue" if "blue" in user_input.lower() else "random"
                    species_configs.append({"species_index": i, "color": color})
            else:
                for i in range(species_count):
                    species_configs.append({"species_index": i, "color": "random"})
                    
            corrected["species_configs"] = species_configs
            
            final_json = json.dumps(corrected, indent=2)
            logger.debug(f"Advanced corrected JSON: {final_json}")
            
            try:
                # Try to validate with Pydantic one last time at the end of everything if we can
                try: 
                    validated_config = SketchConfig.model_validate_json(final_json)  # v2
                except AttributeError: 
                    validated_config = SketchConfig.parse_raw(final_json)  # v1
                logger.info("Advanced auto-correction successful!")
                return validated_config
            except Exception as validation_err:
                logger.error(f"Advanced auto-correction failed validation: {validation_err}")
        
        except Exception as fix_err:
            logger.error(f"Advanced auto-correction attempt failed: {fix_err}")
        
        return None