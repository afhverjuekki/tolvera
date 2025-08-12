"""
Error Pattern Analyzer for Taichi Code Generation

This module analyzes error patterns from CSV logs to identify common issues
and successful correction strategies, enabling continuous improvement.
"""

import csv
import json
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class ErrorPatternAnalyzer:
    """Analyzes error patterns and corrections from synthesis logs."""
    
    def __init__(self, log_file: str = "poe_llm_interactions.csv"):
        self.log_file = log_file
        self.error_patterns = defaultdict(int)
        self.successful_corrections = []
        self.error_contexts = defaultdict(list)
        
    def analyze_logs(self) -> Dict[str, Any]:
        """
        Analyze the CSV logs to extract error patterns and successful corrections.
        
        Returns:
            Dictionary containing analysis results
        """
        if not os.path.exists(self.log_file):
            logger.warning(f"Log file not found: {self.log_file}")
            return {"error": "Log file not found"}
        
        total_attempts = 0
        correction_attempts = 0
        correction_successes = 0
        error_type_counts = Counter()
        successful_fixes = []
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    total_attempts += 1
                    
                    # Parse detected errors
                    if row.get('detected_errors'):
                        try:
                            detected_errors = json.loads(row['detected_errors'])
                            for error in detected_errors:
                                if error.get('severity') == 'error':
                                    error_type_counts[error['message']] += 1
                                    
                                    # Store context for this error type
                                    self.error_contexts[error['message']].append({
                                        'description': row.get('user_description', ''),
                                        'line': error.get('line'),
                                        'match': error.get('match', ''),
                                        'corrected': row.get('correction_succeeded', '') == 'True'
                                    })
                        except json.JSONDecodeError:
                            pass
                    
                    # Track correction attempts
                    if row.get('correction_attempted', '') == 'True':
                        correction_attempts += 1
                        
                        if row.get('correction_succeeded', '') == 'True':
                            correction_successes += 1
                            
                            # Analyze successful corrections
                            if row.get('correction_history'):
                                try:
                                    history = json.loads(row['correction_history'])
                                    successful_fixes.append({
                                        'description': row.get('user_description', ''),
                                        'original_errors': json.loads(row.get('detected_errors', '[]')),
                                        'correction_steps': history,
                                        'final_code': row.get('final_code', '')
                                    })
                                except json.JSONDecodeError:
                                    pass
            
            # Analyze patterns in successful corrections
            correction_patterns = self._analyze_correction_patterns(successful_fixes)
            
            # Identify most problematic error types
            top_errors = error_type_counts.most_common(10)
            
            # Calculate success rates by error type
            error_fix_rates = {}
            for error_type, contexts in self.error_contexts.items():
                total = len(contexts)
                fixed = sum(1 for c in contexts if c['corrected'])
                error_fix_rates[error_type] = {
                    'total': total,
                    'fixed': fixed,
                    'fix_rate': (fixed / total * 100) if total > 0 else 0
                }
            
            return {
                'total_attempts': total_attempts,
                'correction_attempts': correction_attempts,
                'correction_successes': correction_successes,
                'correction_success_rate': (correction_successes / correction_attempts * 100) if correction_attempts > 0 else 0,
                'top_errors': top_errors,
                'error_fix_rates': error_fix_rates,
                'correction_patterns': correction_patterns,
                'successful_fixes_count': len(successful_fixes)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            return {"error": str(e)}
    
    def _analyze_correction_patterns(self, successful_fixes: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in successful corrections."""
        patterns = {
            'simple_fixes': 0,
            'llm_fixes': 0,
            'multi_attempt_fixes': 0,
            'common_fix_sequences': []
        }
        
        for fix in successful_fixes:
            if not fix.get('correction_steps'):
                continue
                
            steps = fix['correction_steps']
            
            # Count fix types
            has_simple = any(step.get('type') == 'simple' for step in steps)
            has_llm = any(step.get('type') == 'llm' for step in steps)
            
            if has_simple:
                patterns['simple_fixes'] += 1
            if has_llm:
                patterns['llm_fixes'] += 1
            if len(steps) > 1:
                patterns['multi_attempt_fixes'] += 1
        
        return patterns
    
    def generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on error analysis.
        
        Returns:
            List of recommendation strings
        """
        analysis = self.analyze_logs()
        recommendations = []
        
        if 'error' in analysis:
            return [f"Cannot generate recommendations: {analysis['error']}"]
        
        # Recommend new patterns based on frequent errors
        for error_type, count in analysis['top_errors'][:5]:
            fix_rate_data = analysis['error_fix_rates'].get(error_type, {})
            fix_rate = fix_rate_data.get('fix_rate', 0)
            
            if fix_rate < 50:  # Low fix rate
                recommendations.append(
                    f"Error '{error_type}' occurs {count} times with only {fix_rate:.1f}% fix rate. "
                    f"Consider adding specific correction rules."
                )
        
        # Recommend prompt improvements
        if analysis['correction_success_rate'] < 70:
            recommendations.append(
                f"Overall correction success rate is {analysis['correction_success_rate']:.1f}%. "
                f"Consider improving correction prompts or adding more examples."
            )
        
        # Identify patterns that work well
        patterns = analysis.get('correction_patterns', {})
        if patterns.get('simple_fixes', 0) > patterns.get('llm_fixes', 0):
            recommendations.append(
                "Simple pattern-based fixes are more successful. "
                "Consider adding more regex patterns for common errors."
            )
        
        return recommendations
    
    def export_error_knowledge(self) -> Dict[str, List[Dict]]:
        """
        Export error patterns and successful fixes for knowledge base.
        
        Returns:
            Dictionary of error types to successful fix examples
        """
        knowledge = defaultdict(list)
        
        analysis = self.analyze_logs()
        if 'error' in analysis:
            return {}
        
        # Group successful fixes by error type
        for error_type, contexts in self.error_contexts.items():
            successful_contexts = [c for c in contexts if c['corrected']]
            if successful_contexts:
                knowledge[error_type] = successful_contexts[:5]  # Top 5 examples
        
        return dict(knowledge)