"""
This module provides pattern-based error detection for common Taichi programming mistakes in LLM-generated code.
"""

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TaichiErrorDetector:
    def __init__(self):
        # Each pattern here is a tuple of (regex pattern, error message, severity, hint)
        # These are all documented from what I've found after generating a ton
        # of examples
        self.error_patterns: List[Tuple[str, str, str, str]] = [
            # Uninitialized variables
            (r'(\w+)\s*=\s*\1\s*\+',
             'Uninitialized variable: possible use before definition',
             'error',
             'Initialize the variable before using it, e.g., force = ti.math.vec2(0.0, 0.0)'),
            
            # Variable defined only inside conditional blocks
            (r'if\s+.*?:\s*\n\s+(\w+)\s*=(?!\s*\1).*?\n.*?else\s*:\s*\n\s+\1\s*=.*?\n.*?\n.*?\1',
             'Variable defined inside conditional may not be accessible',
             'error',
             'Initialize variables before conditional blocks to ensure they are always defined'),

            # Using ti.norm() instead of .norm()
            (r'ti\.norm\s*\(',
             'Incorrect usage: ti.norm() does not exist',
             'error',
             'Use vector.norm() method instead, e.g., (p2.pos - p1.pos).norm()'),

            # Using ti.dot() instead of .dot()
            (r'ti\.dot\s*\(',
             'Incorrect usage: ti.dot() does not exist',
             'error',
             'Use vector.dot() method instead, e.g., vec1.dot(vec2)'),

            # Missing ti.math namespace
            (r'ti\.pi',
             'Incorrect usage: ti.pi does not exist',
             'error',
             'Import from Python math module: from math import pi'),

            # Division without zero check
            (r'(\w+)\s*/\s*dist(?!\s*\+)',
             'Potential division by zero: dist not checked',
             'warning',
             'Add a check before division: if dist > 0.0: direction = to_other / dist'),
            
            # Unsafe vector normalization without magnitude check
            (r'(\w+)\.normalized\(\)(?!.*?if.*?\.norm\(\)\s*>)',
             'Unsafe vector normalization: no magnitude check',
             'error',
             'Check magnitude before normalizing: if vec.norm() > 0.01: direction = vec.normalized()'),

            # Using Python tuples instead of ti.math.vec2
            (r'force\s*=\s*\([\d.]+\s*,\s*[\d.]+\s*\)',
             'Using Python tuple instead of ti.math.vec2',
             'error',
             'Use ti.math.vec2(x, y) instead of (x, y)'),

            # Using .x/.y instead of indexing
            (r'\.x\b|\\.y\b',
             'Vector component access: .x/.y not supported in Taichi',
             'error',
             'Use indexing: vec[0] for x, vec[1] for y'),

            # Missing return statement in @ti.func
            (r'@ti\.func.*?def\s+\w+[^@]*?(?=@|\Z)(?!.*?\breturn\b)',
             'Missing return statement in Taichi function',
             'error',
             'Taichi functions decorated with @ti.func must return a value'),

            # Using Python math functions instead of Taichi
            (r'math\.(sin|cos|tan|sqrt|atan2)\s*\(',
             'Using Python math functions in Taichi scope',
             'error',
             'Use Taichi math functions: ti.sin(), ti.cos(), etc.'),

            # Incorrect field access patterns
            (r'tv\.p\.p\.field',
             'Double field access: tv.p.p.field is incorrect',
             'error',
             'Use tv.p.field instead'),

            # Incorrect tv subscript access
            (r'tv\[[0-9]\]',
             'Invalid subscript access on Tolvera object',
             'error',
             'Use tv.x for width and tv.y for height instead of tv[0] and tv[1]'),

            # Wrong expert function parameter order - species first
            (r'def\s+expert_\w+\s*\(\s*species\s*:\s*ti\.i32\s*,\s*pos\s*:',
             'Wrong parameter order: species cannot be first parameter',
             'error',
             'Parameter order must be: (pos, vel, mass, species, particle_idx), NOT (species, pos, ...)'),
            
            # Wrong expert function parameter order - vel before pos
            (r'def\s+expert_\w+\s*\(\s*vel\s*:\s*ti\.math\.vec2\s*,\s*pos\s*:',
             'Wrong parameter order: vel cannot come before pos',
             'error',
             'Parameter order must be: (pos, vel, mass, species, particle_idx)'),
            
            # Wrong expert function parameter order - species before mass
            (r'def\s+expert_\w+\s*\(\s*pos\s*:\s*ti\.math\.vec2\s*,\s*vel\s*:\s*ti\.math\.vec2\s*,\s*species\s*:',
             'Wrong parameter order: species must come after mass',
             'error',
             'Parameter order must be: (pos, vel, mass, species, particle_idx)'),
            
            # Detect parameter type mismatch in kernel calls
            (r'expert_\w+\s*\(\s*species\s*,\s*pos\s*,\s*vel\s*,\s*mass\s*,\s*\w+\s*\)',
             'Expert function called with wrong parameter order',
             'error',
             'Call must match definition: expert_name(pos, vel, mass, species, particle_idx)'),

            # Force magnitude too weak
            (r'force\s*[+*]=?\s*[^*]*\*\s*([0-9.]+)(?!\d)',
             'Force magnitude may be too weak',
             'info',
             'Consider using force magnitudes between 50-400 for visible effects'),
            
            # Function accesses states but doesn't compute force
            (r'tv\.s\.llm_\w+\.field.*\n(?!.*force\s*=).*$',
             'Function accesses states but may not compute force',
             'warning',
             'After accessing states, compute a force using those values'),
            
            # State access errors
            (r"AttributeError.*'_IntermediateStruct\d+'.*has no attribute '(\w+)'",
             "Accessing non-existent state: {1}",
             'error',
             "State '{1}' does not exist. Check available states in state_context."),
            
            (r"tv\.s\.llm_\w+\.field\[\w+\]\.(\w+).*#.*(?:ERROR|error|Error)",
             "Attempting to use non-existent state: {1}",
             'error',  
             "State '{1}' not found. Only use states listed in state_context."),
            
            # Common state property mistakes
            (r'tv\.s\.llm_particle\.field\[\w+\]\.position',
             "'position' is not a custom state property",
             'error',
             "Use tv.p.field[i].pos for particle position"),
            
            (r'tv\.s\.llm_particle\.field\[\w+\]\.velocity',
             "'velocity' is not a custom state property",
             'error',
             "Use tv.p.field[i].vel for particle velocity"),
            
            (r'ti\.math\.length\s*\(',
             'ti.math.length does not exist',
             'error',
             'Use .norm() method on vectors, e.g., vec.norm()'),
            
            # AttributeError patterns for position/velocity
            (r"AttributeError.*position",
             "AttributeError: 'position' not found",
             'error',
             "Particle position is accessed via tv.p.field[i].pos, not custom states"),
            
            (r"AttributeError.*velocity",
             "AttributeError: 'velocity' not found",
             'error',
             "Particle velocity is accessed via tv.p.field[i].vel, not custom states"),
        ]

    def _create_error_dict(self,
                           line: int,
                           severity: str,
                           message: str,
                           hint: str,
                           pattern: str,
                           match_text: str) -> Dict[str,
                                                    any]:
        return {
            'line': line,
            'severity': severity,
            'message': message,
            'hint': hint,
            'pattern': pattern,
            'match': match_text
        }

    def _truncate_match(self, match_text: str, max_length: int = 50) -> str:
        if len(match_text) > max_length:
            return match_text[:max_length] + '...'
        return match_text

    def _process_multiline_pattern(self,
                                   pattern: str,
                                   code: str,
                                   message: str,
                                   severity: str,
                                   hint: str) -> List[Dict[str,
                                                           any]]:
        errors = []
        matches = re.finditer(pattern, code, re.DOTALL | re.MULTILINE)

        for match in matches:
            line_num = code[:match.start()].count('\n') + 1
            match_text = self._truncate_match(match.group(0))
            errors.append(self._create_error_dict(
                line_num, severity, message, hint, pattern, match_text
            ))

        return errors

    def _process_force_magnitude(self,
                                 match,
                                 line_num: int,
                                 message: str,
                                 severity: str,
                                 hint: str,
                                 pattern: str) -> Dict[str,
                                                       any]:
        magnitude = float(match.group(1))
        if magnitude < 10:
            return self._create_error_dict(
                line_num, severity,
                f'{message} (found: {magnitude})',
                hint, pattern, match.group(0)
            )
        return None

    def _process_single_line_patterns(self,
                                      lines: List[str],
                                      pattern: str,
                                      message: str,
                                      severity: str,
                                      hint: str) -> List[Dict[str,
                                                              any]]:
        errors = []

        for line_num, line in enumerate(lines, 1):
            matches = re.finditer(pattern, line)
            for match in matches:
                # Sometimes the force is hard to see so this is a check for
                # that
                if 'magnitude may be too weak' in message:
                    error = self._process_force_magnitude(
                        match, line_num, message, severity, hint, pattern
                    )
                    if error:
                        errors.append(error)
                else:
                    errors.append(
                        self._create_error_dict(
                            line_num,
                            severity,
                            message,
                            hint,
                            pattern,
                            match.group(0)))

        return errors

    def _is_multiline_pattern(self, pattern: str) -> bool:
        # Patterns that need multiline matching
        multiline_patterns = [
            r'@ti\\.func\s*\n\s*def\s+\w+\([^)]*\)\s*->\s*ti\\.math\\.vec2:((?!.*\\breturn\\b(.|\n))*$)',
            r'(if|elif|else|for|while)[^\n]*:\s*\n(?:\s*.*\n)*?^\s*return\s+'
        ]
        return pattern in multiline_patterns

    def detect_errors(self, code: str) -> List[Dict[str, any]]:
        errors = []
        lines = code.split('\n')

        for pattern, message, severity, hint in self.error_patterns:
            if self._is_multiline_pattern(pattern):
                errors.extend(self._process_multiline_pattern(
                    pattern, code, message, severity, hint
                ))
            else:
                errors.extend(self._process_single_line_patterns(
                    lines, pattern, message, severity, hint
                ))

        # Sort errors by line number and severity
        severity_order = {'error': 0, 'warning': 1, 'info': 2}
        errors.sort(
            key=lambda x: (
                x['line'],
                severity_order.get(
                    x['severity'],
                    3)))

        return errors

    # Lists errors and the amount of them found in the code
    def get_error_summary(
            self, errors: List[Dict[str, any]]) -> Dict[str, int]:
        summary = {'error': 0, 'warning': 0, 'info': 0}
        for error in errors:
            severity = error.get('severity', 'info')
            summary[severity] = summary.get(severity, 0) + 1
        return summary