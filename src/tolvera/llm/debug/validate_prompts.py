#!/usr/bin/env python3
"""
Comprehensive validation tool for the prompt loading system.
This script validates that all prompts are correctly loaded, formatted, and assembled.
"""

import sys
import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tolvera.llm.core.prompt_loader import PromptLoader, get_prompt_loader
from tolvera.llm.core.prompts import ContextAwarePromptBuilder

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PromptValidator:
    """Validates the prompt loading and assembly system."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.loader = get_prompt_loader()
        self.prompt_builder = ContextAwarePromptBuilder()
        self.results = {
            'files_checked': 0,
            'files_loaded': 0,
            'files_failed': [],
            'placeholders_found': {},
            'unsubstituted_placeholders': {},
            'prompt_sizes': {},
            'assembled_prompts': {},
            'errors': [],
            'warnings': []
        }
        
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def validate_single_file(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate a single prompt file.
        
        Returns:
            Tuple of (success, content, error_message)
        """
        self.results['files_checked'] += 1
        
        try:
            # Try to load the prompt file
            content = self.loader.load_prompt(file_path)
            
            # Check for placeholders
            placeholders = re.findall(r'\{(\w+)\}', content)
            if placeholders:
                self.results['placeholders_found'][file_path] = placeholders
                logger.info(f"✓ File '{file_path}' loaded ({len(content)} chars, {len(placeholders)} placeholders)")
            else:
                logger.info(f"✓ File '{file_path}' loaded ({len(content)} chars, no placeholders)")
            
            self.results['files_loaded'] += 1
            self.results['prompt_sizes'][file_path] = len(content)
            
            return True, content, None
            
        except FileNotFoundError as e:
            error_msg = f"File not found: {file_path}"
            logger.error(f"✗ {error_msg}")
            self.results['files_failed'].append(file_path)
            self.results['errors'].append(error_msg)
            return False, None, error_msg
            
        except Exception as e:
            error_msg = f"Error loading {file_path}: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.results['files_failed'].append(file_path)
            self.results['errors'].append(error_msg)
            return False, None, error_msg
    
    def validate_with_substitution(self, file_path: str, test_vars: Dict[str, str]) -> bool:
        """
        Validate a prompt file with variable substitution.
        """
        try:
            # Load with substitution
            content = self.loader.load_prompt(file_path, **test_vars)
            
            # Check for remaining placeholders
            remaining = re.findall(r'\{(\w+)\}', content)
            if remaining:
                self.results['unsubstituted_placeholders'][file_path] = remaining
                logger.warning(f"⚠ File '{file_path}' has unsubstituted placeholders: {remaining}")
                self.results['warnings'].append(f"Unsubstituted placeholders in {file_path}: {remaining}")
                return False
            
            logger.info(f"✓ File '{file_path}' substitution successful")
            return True
            
        except KeyError as e:
            error_msg = f"Missing required variable in {file_path}: {e}"
            logger.error(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)
            return False
    
    def validate_multi_part_assembly(self, parts: List[str], test_vars: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """
        Validate multi-part prompt assembly.
        """
        try:
            combined = self.loader.load_multi_part_prompt(parts, **test_vars)
            
            # Check for issues
            if not combined:
                error_msg = "Multi-part assembly resulted in empty prompt"
                logger.error(f"✗ {error_msg}")
                self.results['errors'].append(error_msg)
                return False, None
            
            # Check for remaining placeholders
            remaining = re.findall(r'\{(\w+)\}', combined)
            if remaining:
                warning_msg = f"Combined prompt has unsubstituted placeholders: {remaining}"
                logger.warning(f"⚠ {warning_msg}")
                self.results['warnings'].append(warning_msg)
            
            logger.info(f"✓ Multi-part assembly successful ({len(combined)} chars from {len(parts)} parts)")
            return True, combined
            
        except Exception as e:
            error_msg = f"Multi-part assembly failed: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)
            return False, None
    
    def validate_synthesis_prompts(self):
        """Validate synthesis-related prompts."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATING SYNTHESIS PROMPTS")
        logger.info("="*60)
        
        # Test synthesis prompt files
        synthesis_files = [
            "synthesis/expert_synthesis_requirements.txt",
            "synthesis/state_analysis_system.txt",
            "synthesis/state_analysis_examples.txt",
            "synthesis/state_analysis_criteria.txt",
            "synthesis/taichi_critical_rules.txt"
        ]
        
        # Test variables for synthesis prompts
        test_vars = {
            'species_count': '3',
            'expert_type_guidance': 'Test guidance for expert type'
        }
        
        for file_path in synthesis_files:
            success, content, error = self.validate_single_file(file_path)
            if success and self.results['placeholders_found'].get(file_path):
                self.validate_with_substitution(file_path, test_vars)
        
        # Test multi-part assembly for state analysis
        state_analysis_parts = [
            "synthesis/state_analysis_system.txt",
            "synthesis/state_analysis_examples.txt",
            "synthesis/state_analysis_criteria.txt"
        ]
        
        success, combined = self.validate_multi_part_assembly(state_analysis_parts, test_vars)
        if success:
            self.results['assembled_prompts']['state_analysis'] = len(combined)
    
    def validate_decomposition_prompts(self):
        """Validate decomposition-related prompts."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATING DECOMPOSITION PROMPTS")
        logger.info("="*60)
        
        decomposition_files = [
            "decomposition/behavior_decomposition_system.txt",
            "decomposition/behavior_decomposition_examples.txt",
            "decomposition/behavior_decomposition_criteria.txt"
        ]
        
        for file_path in decomposition_files:
            self.validate_single_file(file_path)
        
        # Test multi-part assembly
        success, combined = self.validate_multi_part_assembly(decomposition_files, {})
        if success:
            self.results['assembled_prompts']['decomposition'] = len(combined)
    
    def validate_refinement_prompts(self):
        """Validate refinement-related prompts."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATING REFINEMENT PROMPTS")
        logger.info("="*60)
        
        refinement_files = [
            "refinement/sketch_analysis_system.txt",
            "refinement/sketch_implementation_system.txt",
            "refinement/single_state_refinement_system.txt"
        ]
        
        # Test variables for refinement prompts
        test_vars = {
            'slime_exemplar': '# Test slime exemplar code',
            'boids_exemplar': '# Test boids exemplar code',
            'particle_life_exemplar': '# Test particle life exemplar',
            'taichi_crashes': '# Common Taichi crashes',
            'taichi_fundamentals': '# Taichi fundamentals',
            'movement_patterns': '# Movement patterns',
            'flocking_patterns': '# Flocking patterns'
        }
        
        for file_path in refinement_files:
            success, content, error = self.validate_single_file(file_path)
            if success and self.results['placeholders_found'].get(file_path):
                self.validate_with_substitution(file_path, test_vars)
    
    def validate_drawing_prompts(self):
        """Validate drawing-related prompts."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATING DRAWING PROMPTS")
        logger.info("="*60)
        
        drawing_files = [
            "drawing/drawing_instructions.txt",
            "drawing/drawing_state_access.txt"
        ]
        
        for file_path in drawing_files:
            self.validate_single_file(file_path)
    
    def validate_utility_prompts(self):
        """Validate utility prompts."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATING UTILITY PROMPTS")
        logger.info("="*60)
        
        utility_files = [
            "utilities/color_resolution.txt"
        ]
        
        for file_path in utility_files:
            self.validate_single_file(file_path)
    
    def test_prompt_builder(self):
        """Test the ContextAwarePromptBuilder."""
        logger.info("\n" + "="*60)
        logger.info("TESTING PROMPT BUILDER")
        logger.info("="*60)
        
        try:
            # Test building a synthesis prompt
            test_description = "particles chase each other in a predator-prey simulation"
            test_states = {
                'global': ['time_elapsed', 'day_cycle'],
                'particle': ['energy', 'hunger'],
                'species': ['reproduction_rate']
            }
            
            prompt = self.prompt_builder.build_synthesis_prompt(
                description=test_description,
                available_states=test_states,
                constrained=False
            )
            
            # Check for 5-element structure
            elements = {
                'ROLE': prompt.count('## ROLE'),
                'OBJECTIVE': prompt.count('## OBJECTIVE'),
                'TASK AT HAND': prompt.count('## TASK AT HAND'),
                'KEY EXAMPLES': prompt.count('## KEY EXAMPLES'),
                'SUCCESS VS. FAILURE CRITERIA': prompt.count('## SUCCESS VS. FAILURE CRITERIA')
            }
            
            all_present = all(count == 1 for count in elements.values())
            
            if all_present:
                logger.info(f"✓ Prompt builder generated valid 5-element structure ({len(prompt)} chars)")
                self.results['assembled_prompts']['synthesis_test'] = len(prompt)
            else:
                missing = [elem for elem, count in elements.items() if count != 1]
                error_msg = f"Invalid prompt structure. Issues with: {missing}"
                logger.error(f"✗ {error_msg}")
                self.results['errors'].append(error_msg)
            
            # Check for placeholders
            remaining = re.findall(r'\{(\w+)\}', prompt)
            if remaining:
                warning_msg = f"Generated prompt has unsubstituted placeholders: {remaining}"
                logger.warning(f"⚠ {warning_msg}")
                self.results['warnings'].append(warning_msg)
                
        except Exception as e:
            error_msg = f"Prompt builder test failed: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("\n" + "="*60)
        report.append("PROMPT VALIDATION REPORT")
        report.append("="*60)
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Files checked: {self.results['files_checked']}")
        report.append(f"Files loaded successfully: {self.results['files_loaded']}")
        report.append(f"Files failed: {len(self.results['files_failed'])}")
        report.append(f"Total errors: {len(self.results['errors'])}")
        report.append(f"Total warnings: {len(self.results['warnings'])}")
        report.append("")
        
        # Failed files
        if self.results['files_failed']:
            report.append("FAILED FILES")
            report.append("-" * 40)
            for file_path in self.results['files_failed']:
                report.append(f"  ✗ {file_path}")
            report.append("")
        
        # Files with placeholders
        if self.results['placeholders_found']:
            report.append("FILES WITH PLACEHOLDERS")
            report.append("-" * 40)
            for file_path, placeholders in self.results['placeholders_found'].items():
                report.append(f"  {file_path}: {', '.join(placeholders)}")
            report.append("")
        
        # Unsubstituted placeholders
        if self.results['unsubstituted_placeholders']:
            report.append("UNSUBSTITUTED PLACEHOLDERS")
            report.append("-" * 40)
            for file_path, placeholders in self.results['unsubstituted_placeholders'].items():
                report.append(f"  ⚠ {file_path}: {', '.join(placeholders)}")
            report.append("")
        
        # Assembled prompt sizes
        if self.results['assembled_prompts']:
            report.append("ASSEMBLED PROMPTS")
            report.append("-" * 40)
            for name, size in self.results['assembled_prompts'].items():
                report.append(f"  {name}: {size:,} characters")
            report.append("")
        
        # Errors
        if self.results['errors']:
            report.append("ERRORS")
            report.append("-" * 40)
            for error in self.results['errors']:
                report.append(f"  ✗ {error}")
            report.append("")
        
        # Warnings
        if self.results['warnings']:
            report.append("WARNINGS")
            report.append("-" * 40)
            for warning in self.results['warnings']:
                report.append(f"  ⚠ {warning}")
            report.append("")
        
        # Final status
        report.append("VALIDATION STATUS")
        report.append("-" * 40)
        if self.results['errors']:
            report.append("❌ VALIDATION FAILED - Errors detected in prompt system")
        elif self.results['warnings']:
            report.append("⚠️  VALIDATION PASSED WITH WARNINGS")
        else:
            report.append("✅ VALIDATION PASSED - All prompts loaded successfully")
        
        return "\n".join(report)
    
    def save_results(self, output_path: Optional[str] = None):
        """Save validation results to a JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"prompt_validation_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    def run_validation(self) -> bool:
        """
        Run complete validation suite.
        
        Returns:
            True if validation passed, False otherwise
        """
        logger.info("Starting prompt system validation...")
        
        # Run all validations
        self.validate_synthesis_prompts()
        self.validate_decomposition_prompts()
        self.validate_refinement_prompts()
        self.validate_drawing_prompts()
        self.validate_utility_prompts()
        self.test_prompt_builder()
        
        # Generate and print report
        report = self.generate_report()
        print(report)
        
        # Save results
        self.save_results()
        
        # Return success status
        return len(self.results['errors']) == 0


def main():
    """Main entry point for validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate prompt loading system')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-o', '--output', help='Output file for JSON results')
    parser.add_argument('--dry-run', action='store_true', help='Test mode without LLM calls')
    
    args = parser.parse_args()
    
    # Set up environment for dry run
    if args.dry_run:
        os.environ['PROMPT_VALIDATION_DRY_RUN'] = '1'
        logger.info("DRY RUN MODE - No LLM calls will be made")
    
    # Run validation
    validator = PromptValidator(verbose=args.verbose)
    success = validator.run_validation()
    
    if args.output:
        validator.save_results(args.output)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()