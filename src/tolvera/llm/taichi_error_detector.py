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

            # Force magnitude too weak
            (r'force\s*[+*]=?\s*[^*]*\*\s*([0-9.]+)(?!\d)',
             'Force magnitude may be too weak',
             'info',
             'Consider using force magnitudes between 50-400 for visible effects'),
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
        # Currently only the missing return statement pattern is multiline so I
        # check with that
        return pattern == r'@ti\.func\s*\n\s*def\s+\w+.*?(?=@|\Z)(?!.*return)'

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
