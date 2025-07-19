"""
This module provides LLM-based error correction for Taichi code.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

from .taichi_error_detector import TaichiErrorDetector
from .poe_ollama import OllamaClient

logger = logging.getLogger(__name__)


class TaichiErrorCorrector:

    def __init__(self, model_name: Optional[str] = None):
        self.detector = TaichiErrorDetector()
        self.client = OllamaClient(model_name or "qwen2.5:3b")

        # Simple rule-based corrections for common error patterns
        self.simple_corrections = [
            # Pattern to find, replacement pattern
            (r'ti\.norm\s*\((.*?)\)', r'(\1).norm()'),  # ti.norm(x) -> x.norm()
            (r'ti\.dot\s*\((.*?),\s*(.*?)\)', r'(\1).dot(\2)'),  # ti.dot(a, b) -> a.dot(b)
            (r'ti\.pi', 'pi'),  # ti.pi -> pi (assuming math import)
            (r'(\w+)\.x\b', r'\1[0]'),  # vec.x -> vec[0]
            (r'(\w+)\.y\b', r'\1[1]'),  # vec.y -> vec[1]
            (r'tv\.p\.p\.field', 'tv.p.field'),  # double field access
            (r'math\.(sin|cos|tan|sqrt|atan2)\s*\(',
             r'ti.\1('),  # math.sin -> ti.sin
            (r'tv\[0\]', 'tv.x'),  # tv[0] -> tv.x
            (r'tv\[1\]', 'tv.y'),  # tv[1] -> tv.y
            # Fix wrong parameter order in expert functions
            (r'def\s+(expert_\w+)\s*\(\s*species\s*:\s*ti\.i32\s*,\s*pos\s*:\s*ti\.math\.vec2\s*,\s*vel\s*:\s*ti\.math\.vec2\s*,\s*mass\s*:\s*ti\.f32\s*,\s*particle_idx\s*:\s*ti\.i32\s*\)',
             r'def \1(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32)'),
        ]

    def fix_parameter_order(self, code: str) -> Tuple[str, bool]:
        """Fix parameter order in expert functions."""
        import re
        
        # Pattern to match expert function definition
        func_pattern = r'(@ti\.func\s*\n\s*def\s+expert_\w+\s*\()([^)]+)(\)\s*->\s*ti\.math\.vec2\s*:)'
        
        match = re.search(func_pattern, code, re.MULTILINE | re.DOTALL)
        if not match:
            return code, False
        
        decorator_and_def = match.group(1)
        params = match.group(2)
        return_type = match.group(3)
        
        # Check if parameters are in wrong order
        if 'species:' in params and (params.strip().startswith('species:') or 
                                     'vel:' in params and params.strip().startswith('vel:')):
            # Parse parameters
            param_list = [p.strip() for p in params.split(',')]
            param_dict = {}
            
            for param in param_list:
                if 'pos:' in param:
                    param_dict['pos'] = param
                elif 'vel:' in param:
                    param_dict['vel'] = param
                elif 'mass:' in param:
                    param_dict['mass'] = param
                elif 'species:' in param:
                    param_dict['species'] = param
                elif 'particle_idx:' in param or 'i:' in param:
                    param_dict['particle_idx'] = param
            
            # Reconstruct in correct order
            correct_order = ['pos', 'vel', 'mass', 'species', 'particle_idx']
            new_params = []
            
            for key in correct_order:
                if key in param_dict:
                    new_params.append(param_dict[key])
            
            if len(new_params) == 5:
                new_param_str = ', '.join(new_params)
                fixed_code = code.replace(
                    decorator_and_def + params + return_type,
                    decorator_and_def + new_param_str + return_type
                )
                return fixed_code, True
        
        return code, False

    def apply_simple_corrections(self, code: str) -> Tuple[str, List[str]]:
        corrected = code
        applied = []

        for pattern, replacement in self.simple_corrections:
            if re.search(pattern, corrected):
                corrected = re.sub(pattern, replacement, corrected)
                applied.append(f"Fixed: {pattern} -> {replacement}")

        # Apply special parameter reordering correction
        corrected, param_fixes = self._fix_parameter_order(corrected)
        applied.extend(param_fixes)

        return corrected, applied
    
    def _fix_parameter_order(self, code: str) -> Tuple[str, List[str]]:
        """Fix parameter order in expert functions where species comes first."""
        applied = []
        
        # Pattern to match expert functions with wrong parameter order
        pattern = r'(def\s+(expert_\w+)\s*\(\s*species\s*:\s*ti\.i32\s*,\s*pos\s*:\s*ti\.math\.vec2\s*,\s*vel\s*:\s*ti\.math\.vec2\s*,\s*mass\s*:\s*ti\.f32\s*,\s*particle_idx\s*:\s*ti\.i32\s*\)\s*->.*?:\n)(.*?)(?=\n@|\n\ndef|\Z)'
        
        def reorder_params(match):
            func_def = match.group(1)
            func_name = match.group(2)
            func_body = match.group(3)
            
            # Fix the function definition
            new_def = re.sub(
                r'species\s*:\s*ti\.i32\s*,\s*pos\s*:\s*ti\.math\.vec2\s*,\s*vel\s*:\s*ti\.math\.vec2\s*,\s*mass\s*:\s*ti\.f32\s*,\s*particle_idx\s*:\s*ti\.i32',
                'pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32',
                func_def
            )
            
            applied.append(f"Fixed parameter order in {func_name}")
            
            # Return the corrected function
            return new_def + func_body
        
        # Apply the correction
        corrected = re.sub(pattern, reorder_params, code, flags=re.DOTALL | re.MULTILINE)
        
        return corrected, applied

    async def correct_with_llm(self, code: str, errors: List[Dict]) -> str:
        # Build error summary for the prompt
        error_summary = []
        for error in errors:
            if error['severity'] == 'error':
                error_summary.append(
                    f"Line {error['line']}: {error['message']}\n"
                    f"  Code: {error.get('match', 'N/A')}\n"
                    f"  Fix: {error['hint']}"
                )

        if not error_summary:
            return code

        prompt = f"""Fix the following Taichi code errors. Make minimal changes to fix only the specified errors.

        ERRORS TO FIX:
        {chr(10).join(error_summary)}

        IMPORTANT TAICHI RULES:
        1. Only fix the specific errors mentioned above
        2. Do not change the overall logic or structure
        3. Ensure all variables are properly initialized: force = ti.math.vec2(0.0, 0.0)
        4. Use ti.math.vec2(x, y) for 2D vectors, not tuples or ti.Vector([x, y])
        5. Use vector[0] and vector[1] for x,y components, not .x or .y
        6. For math constants like pi, assume "from math import pi" exists
        7. Use ti.sin(), ti.cos() instead of math.sin(), math.cos() in Taichi functions
        8. Always include return statements in @ti.func functions
        9. Do NOT add imports inside the function
        10. Use (vector).norm() not ti.norm(vector)
        11. Use vec1.dot(vec2) not ti.dot(vec1, vec2)
        12. CRITICAL: Expert function parameters MUST be in this order: (pos, vel, mass, species, particle_idx)

        CODE TO FIX:
        ```python
        {code}
        ```

        Return ONLY the corrected code with the same function signature, no imports or explanations:"""

        messages = [
            {'role': 'system', 'content': 'You are a Taichi programming expert. Fix only the specific errors mentioned, making minimal changes.'},
            {'role': 'user', 'content': prompt}
        ]

        try:
            response = await self.client.chat(messages, temperature=0.1)

            code_match = re.search(
                r'```(?:python)?\n(.*?)```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

            # If no code blocks, assume entire response is code
            return response.strip()

        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            return code

    async def correct_errors(self, code: str, max_attempts: int = 2) -> Dict:
        original_code = code
        attempt = 0
        correction_history = []

        while attempt < max_attempts:
            attempt += 1

            errors = self.detector.detect_errors(code)
            error_summary = self.detector.get_error_summary(errors)

            if error_summary['error'] == 0:
                # No errors, we're done!
                return {
                    'success': True,
                    'corrected_code': code,
                    'original_code': original_code,
                    'attempts': attempt,
                    'correction_history': correction_history,
                    'final_errors': errors
                }

            logger.info(
                f"Correction attempt {attempt}: Found {error_summary['error']} errors")

            # Step 1: Fix parameter order if needed
            code, param_fixed = self.fix_parameter_order(code)
            if param_fixed:
                logger.info("Fixed parameter order in expert function")
                correction_history.append({
                    'attempt': attempt,
                    'type': 'parameter_order',
                    'fixes': ['Fixed parameter order']
                })
            
            # Step 2: Apply simple corrections if possible
            code, simple_fixes = self.apply_simple_corrections(code)
            if simple_fixes:
                logger.info(f"Applied {len(simple_fixes)} simple corrections")
                correction_history.append({
                    'attempt': attempt,
                    'type': 'simple',
                    'fixes': simple_fixes
                })

            # Re-detect errors after simple fixes
            errors = self.detector.detect_errors(code)
            error_summary = self.detector.get_error_summary(errors)

            if error_summary['error'] == 0:
                continue  # We're all set here

            # Step 3: Use LLM for remaining errors
            logger.info(
                f"Using LLM to fix {error_summary['error']} remaining errors")
            code = await self.correct_with_llm(code, errors)
            correction_history.append({
                'attempt': attempt,
                'type': 'llm',
                'errors_addressed': len([e for e in errors if e['severity'] == 'error'])
            })

        # Final error check
        final_errors = self.detector.detect_errors(code)
        final_summary = self.detector.get_error_summary(final_errors)

        return {
            'success': final_summary['error'] == 0,
            'corrected_code': code,
            'original_code': original_code,
            'attempts': attempt,
            'correction_history': correction_history,
            'final_errors': final_errors,
            'final_error_summary': final_summary
        }
