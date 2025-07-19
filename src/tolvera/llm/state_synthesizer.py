"""
State Synthesizer for dynamic state generation in the PoE behavior system.

This module analyzes behavior descriptions and generates required state properties
that can be dynamically created in Tölvera.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from .poe_ollama import OllamaClient
from .prompt_loader import load_prompt

logger = logging.getLogger(__name__)


@dataclass
class StateProperty:
    """Represents a single state property."""
    name: str
    type: str  # "ti.f32", "ti.i32", "ti.math.vec2", etc.
    min_value: float
    max_value: float
    description: str = ""
    update_frequency: str = "per_frame"  # "per_frame", "per_10_frames", "per_100_frames"


@dataclass
class TemporalConfig:
    """Configuration for temporal behaviors."""
    requires_time: bool = False
    time_units: List[str] = None  # ["day", "night", "hour", etc.]
    suggested_day_duration: float = 10.0  # seconds
    frame_rate: float = 60.0
    time_scale: float = 1.0
    
    def __post_init__(self):
        if self.time_units is None:
            self.time_units = []
    
    @property
    def frames_per_day(self) -> float:
        return self.suggested_day_duration * self.frame_rate * self.time_scale


class StateSynthesizer:
    """Analyzes behavior descriptions and generates required state properties."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the state synthesizer with an LLM client."""
        self.client = OllamaClient(model_name or "qwen3:4b")
    
    
    async def analyze_state_requirements(self, description: str) -> Dict[str, Any]:
        """
        Analyze behavior description and return required states.
        
        Returns:
            Dictionary containing:
            - global_states: Global properties (e.g., day_phase, time)
            - particle_states: Per-particle properties (e.g., energy, age)
            - species_states: Per-species properties
            - temporal_config: Time configuration if needed
        """
        logger.info(f"Analyzing state requirements for: '{description}'")
        
        # Use LLM for comprehensive analysis without any pre-detection
        system_prompt = load_prompt("state_analysis_system")
        user_prompt = load_prompt("state_analysis_user").format(
            description=description
        )
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        try:
            response = await self.client.chat(messages, temperature=0.1)
            logger.debug(f"State analysis raw response length: {len(response)}")
            logger.debug(f"State analysis raw response first 500 chars: {repr(response[:500])}")
            logger.debug(f"State analysis raw response last 500 chars: {repr(response[-500:])}")
            
            # Parse JSON response
            state_spec = self._parse_state_response(response)
            
            # Add temporal config if present in LLM response
            if state_spec.get('temporal_config'):
                temporal_data = state_spec['temporal_config']
                state_spec['temporal_config'] = TemporalConfig(
                    requires_time=True,
                    time_units=temporal_data.get('time_units', []),
                    suggested_day_duration=temporal_data.get('day_duration', 10.0),
                    frame_rate=temporal_data.get('frame_rate', 60.0),
                    time_scale=temporal_data.get('time_scale', 1.0)
                )
            
            logger.info(f"Analyzed states: {self._summarize_states(state_spec)}")
            return state_spec
            
        except Exception as e:
            logger.error(f"State analysis failed: {e}")
            logger.error(f"LLM response was: {repr(response) if 'response' in locals() else 'No response available'}")
            # Return empty state spec - no fallbacks
            return {
                'global_states': {},
                'particle_states': {},
                'species_states': {},
                'pixel_states': {},
                'temporal_config': None
            }
    
    def _parse_state_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a state specification."""
        # Log the original response for debugging
        logger.debug(f"Original response to parse (first 200 chars): {repr(response[:200])}")
        
        # Clean response more thoroughly
        response_clean = response.strip()
        
        # Remove markdown code blocks
        if "```json" in response_clean:
            start = response_clean.find("```json") + 7
            end = response_clean.rfind("```")
            if end > start:
                response_clean = response_clean[start:end].strip()
            else:
                logger.warning("Found ```json but no closing ``` - response may be truncated")
        elif "```" in response_clean:
            # Handle plain ``` blocks
            start = response_clean.find("```") + 3
            end = response_clean.rfind("```", start)
            if end > start:
                response_clean = response_clean[start:end].strip()
            else:
                logger.warning("Found ``` but no closing ``` - response may be truncated")
        
        # Try to extract JSON if there's extra text
        if response_clean and not response_clean.startswith('{'):
            # Look for the first { and last }
            start_idx = response_clean.find('{')
            end_idx = response_clean.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                response_clean = response_clean[start_idx:end_idx + 1]
            else:
                logger.warning(f"Could not find valid JSON boundaries. Start: {start_idx}, End: {end_idx}")
        
        logger.debug(f"Cleaned JSON response length: {len(response_clean)}")
        logger.debug(f"Cleaned JSON response (first 200 chars): {repr(response_clean[:200])}")
        logger.debug(f"Cleaned JSON response (last 200 chars): {repr(response_clean[-200:])}")
        
        try:
            state_spec = json.loads(response_clean)
            
            # Validate and clean the spec
            validated_spec = {
                'global_states': {},
                'particle_states': {},
                'species_states': {},
                'pixel_states': {},
                'temporal_config': None
            }
            
            # Process each state category (including pixel_states)
            for category in ['global_states', 'particle_states', 'species_states', 'pixel_states']:
                if category in state_spec and isinstance(state_spec[category], dict):
                    for state_name, state_info in state_spec[category].items():
                        if self._validate_state_info(state_info):
                            validated_spec[category][state_name] = state_info
            
            # Copy temporal config if present
            if 'temporal_config' in state_spec:
                validated_spec['temporal_config'] = state_spec['temporal_config']
            
            return validated_spec
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse state response as JSON: {e}")
            logger.warning(f"JSON error at line {e.lineno}, column {e.colno}: {e.msg}")
            logger.warning(f"Response preview around error position: {repr(response_clean[max(0, e.pos-50):e.pos+50])}")
            
            # Try to fix common JSON errors
            try:
                # Fix trailing commas
                response_fixed = re.sub(r',\s*}', '}', response_clean)
                response_fixed = re.sub(r',\s*]', ']', response_fixed)
                # Fix missing commas between properties
                response_fixed = re.sub(r'}\s*"', '}, "', response_fixed)
                response_fixed = re.sub(r'"\s*"', '", "', response_fixed)
                # Fix single quotes (LLMs sometimes use them)
                response_fixed = response_fixed.replace("'", '"')
                
                logger.debug(f"Attempting to parse fixed JSON (first 200 chars): {repr(response_fixed[:200])}")
                state_spec = json.loads(response_fixed)
                logger.info("Successfully parsed JSON after fixing common errors")
                
                validated_spec = {
                    'global_states': {},
                    'particle_states': {},
                    'species_states': {},
                    'pixel_states': {},
                    'temporal_config': None
                }
                
                # Process each state category (including pixel_states)
                for category in ['global_states', 'particle_states', 'species_states', 'pixel_states']:
                    if category in state_spec and isinstance(state_spec[category], dict):
                        for state_name, state_info in state_spec[category].items():
                            if self._validate_state_info(state_info):
                                validated_spec[category][state_name] = state_info
                
                # Copy temporal config if present
                if 'temporal_config' in state_spec:
                    validated_spec['temporal_config'] = state_spec['temporal_config']
                
                return validated_spec
                
            except Exception as fix_error:
                logger.error(f"Could not parse state response even after fixes: {fix_error}")
                logger.error(f"Clean response was: {repr(response_clean[:500])}")
                # Re-raise the exception with more context
                raise ValueError(f"Invalid JSON in state analysis response: {str(e)}. Response preview: {repr(response_clean[:100])}")
    
    def _validate_state_info(self, state_info: Dict) -> bool:
        """Validate that state info has required fields."""
        required_fields = ['type', 'min', 'max']
        if not all(field in state_info for field in required_fields):
            return False
        
        # Special validation for vector types
        type_str = state_info.get('type', '')
        if 'vec' in type_str:
            # For vector types, min/max should be arrays or lists
            min_val = state_info.get('min')
            max_val = state_info.get('max')
            
            # Accept both array format and single values (which will be expanded)
            if isinstance(min_val, (int, float)):
                state_info['min'] = [min_val, min_val]
            if isinstance(max_val, (int, float)):
                state_info['max'] = [max_val, max_val]
                
        return True
    
    def _summarize_states(self, state_spec: Dict) -> str:
        """Create a summary of the state specification."""
        summary_parts = []
        
        for category in ['global_states', 'particle_states', 'species_states']:
            if category in state_spec and state_spec[category]:
                count = len(state_spec[category])
                names = list(state_spec[category].keys())
                summary_parts.append(f"{category}: {count} ({', '.join(names)})")
        
        if state_spec.get('requires_temporal'):
            summary_parts.append("temporal: yes")
        
        return "; ".join(summary_parts)
    
    
    async def generate_state_update_code(self, state_spec: Dict, temporal_config: Optional[TemporalConfig] = None, behavior_description: Optional[str] = None) -> str:
        """Generate Taichi code for updating temporal states using LLM."""
        # Check if we need temporal updates
        has_temporal_states = False
        if temporal_config:
            has_temporal_states = True
        else:
            # Check if any states might be temporal
            for category in ['global_states', 'particle_states']:
                for state_name in state_spec.get(category, {}):
                    if any(keyword in state_name.lower() for keyword in ['time', 'day', 'phase', 'cycle', 'age', 'counter']):
                        has_temporal_states = True
                        break
        
        if not has_temporal_states:
            return ""
        
        # Use LLM to generate appropriate update logic
        system_prompt = load_prompt("temporal_update_system")
        
        # Format temporal config for prompt
        temporal_config_str = ""
        if temporal_config:
            temporal_config_str = f"""
frames_per_day: {temporal_config.frames_per_day if hasattr(temporal_config, 'frames_per_day') else 600}
day_duration: {temporal_config.suggested_day_duration if hasattr(temporal_config, 'suggested_day_duration') else 10.0} seconds
time_scale: {temporal_config.time_scale if hasattr(temporal_config, 'time_scale') else 1.0}
"""
        else:
            temporal_config_str = "No specific temporal configuration provided. Infer appropriate update logic from state names."
        
        # Format state spec for prompt with clear category separation
        state_spec_str = self._format_state_spec_for_prompt(state_spec)
        
        user_prompt = load_prompt("temporal_update_user").format(
            state_spec=state_spec_str,
            temporal_config=temporal_config_str,
            behavior_description=behavior_description or "No specific behavior description provided"
        )
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        try:
            response = await self.client.chat(messages, temperature=0.1)
            logger.debug(f"Temporal update code generated: {response[:200]}...")
            
            # Extract code from response
            code = self._extract_code_from_response(response)
            return code
            
        except Exception as e:
            logger.error(f"Failed to generate temporal update code: {e}")
            # Return minimal fallback
            return "@ti.kernel\ndef update_temporal_states():\n    pass"
    
    def _format_state_spec_for_prompt(self, state_spec: Dict[str, Any]) -> str:
        """Format state specification with clear category separation for LLM prompt."""
        lines = []
        
        # Format global states
        lines.append("=== GLOBAL STATES (access with tv.s.llm_global.field[0].state_name) ===")
        if state_spec.get('global_states'):
            for state_name, state_info in state_spec['global_states'].items():
                lines.append(f"  - {state_name}: {state_info['type']} [{state_info['min']}, {state_info['max']}]")
                if state_info.get('description'):
                    lines.append(f"    Description: {state_info['description']}")
        else:
            lines.append("  (none)")
        lines.append("")
        
        # Format particle states
        lines.append("=== PARTICLE STATES (access with tv.s.llm_particle.field[i].state_name) ===")
        if state_spec.get('particle_states'):
            for state_name, state_info in state_spec['particle_states'].items():
                lines.append(f"  - {state_name}: {state_info['type']} [{state_info['min']}, {state_info['max']}]")
                if state_info.get('description'):
                    lines.append(f"    Description: {state_info['description']}")
        else:
            lines.append("  (none)")
        lines.append("")
        
        # Format species states
        lines.append("=== SPECIES STATES (access with tv.s.llm_species.field[s].state_name) ===")
        if state_spec.get('species_states'):
            for state_name, state_info in state_spec['species_states'].items():
                lines.append(f"  - {state_name}: {state_info['type']} [{state_info['min']}, {state_info['max']}]")
                if state_info.get('description'):
                    lines.append(f"    Description: {state_info['description']}")
        else:
            lines.append("  (none)")
        
        return "\n".join(lines)
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                code = response[start:end].strip()
                return self._validate_and_fix_common_mistakes(code)
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                code = response[start:end].strip()
                return self._validate_and_fix_common_mistakes(code)
        
        # If no code blocks, assume entire response is code
        code = response.strip()
        return self._validate_and_fix_common_mistakes(code)
    
    def _validate_and_fix_common_mistakes(self, code: str) -> str:
        """Validate and fix common mistakes in generated temporal update code."""
        import re
        
        # Check for common mistakes
        mistakes_found = []
        
        # Check for .position instead of .pos
        if re.search(r'tv\.s\.llm_particle\.field\[\w+\]\.position', code):
            mistakes_found.append("Found '.position' - should use 'tv.p.field[i].pos' for particle position")
            # Fix it
            code = re.sub(r'(tv\.s\.llm_particle\.field\[(\w+)\])\.position', r'tv.p.field[\2].pos', code)
        
        # Check for .velocity instead of .vel
        if re.search(r'tv\.s\.llm_particle\.field\[\w+\]\.velocity', code):
            mistakes_found.append("Found '.velocity' - should use 'tv.p.field[i].vel' for particle velocity")
            # Fix it
            code = re.sub(r'(tv\.s\.llm_particle\.field\[(\w+)\])\.velocity', r'tv.p.field[\2].vel', code)
        
        # Check for ti.math.length instead of .norm()
        if 'ti.math.length(' in code:
            mistakes_found.append("Found 'ti.math.length()' - should use '.norm()' method on vectors")
            # Fix it - this is more complex, need to extract the vector expression
            code = re.sub(r'ti\.math\.length\(([^)]+)\)', r'(\1).norm()', code)
        
        # Check for .direction which doesn't exist
        if re.search(r'tv\.s\.llm_particle\.field\[\w+\]\.direction', code):
            mistakes_found.append("Found '.direction' - this is not a valid particle property")
            # Can't auto-fix this as we don't know the intent
        
        if mistakes_found:
            logger.warning(f"Fixed common mistakes in temporal update code: {mistakes_found}")
        
        return code