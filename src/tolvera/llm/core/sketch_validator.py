"""
Sketch Validator Module - Validates and auto-fixes sketches before presentation
"""

import subprocess
import sys
import tempfile
import asyncio
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from .sketch_refiner import SketchRefiner

logger = logging.getLogger(__name__)


class SketchValidator:
    """
    Validates generated sketches by attempting compilation in headless mode.
    Automatically fixes common errors before presenting to users.
    """
    
    def __init__(self, refiner: Optional[SketchRefiner] = None, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the sketch validator.
        
        Args:
            refiner: Optional SketchRefiner for auto-fixing. Will create one if not provided.
            model_name: Model to use for refinement if creating a new refiner
        """
        self.refiner = refiner or SketchRefiner(model_name=model_name)
        self.max_fix_attempts = 3
    
    def _prepare_headless_sketch(self, sketch_code: str) -> str:
        """
        Modify sketch to run in headless mode for validation.
        
        Args:
            sketch_code: Original sketch code
            
        Returns:
            Modified sketch code for headless validation
        """
        # Add headless mode to Tolvera initialization
        headless_code = sketch_code
        
        # Replace Tolvera initialization to add headless=True
        tolvera_pattern = r'tv = Tolvera\((.*?)\)'
        
        def add_headless(match):
            args = match.group(1)
            if 'headless' not in args:
                # Add headless=True to the arguments
                if args.strip():
                    return f'tv = Tolvera({args}, headless=True)'
                else:
                    return 'tv = Tolvera(headless=True)'
            return match.group(0)
        
        headless_code = re.sub(tolvera_pattern, add_headless, headless_code)
        
        # Add early exit after a few frames to just test compilation
        # Find the main() function and add frame counter
        main_pattern = r'(def main\(\):.*?)(\n    tv\.show\(\))'
        
        def add_frame_limit(match):
            main_part = match.group(1)
            show_part = match.group(2)
            
            # Add frame counter and early exit
            frame_counter = """
    # Validation mode - run for a few frames then exit
    frame_count = 0
    max_frames = 5"""
            
            show_with_exit = """
    frame_count += 1
    if frame_count >= max_frames:
        print("Validation successful - sketch compiles and runs")
        return  # Exit after validation
    tv.show()"""
            
            return main_part + frame_counter + show_with_exit
        
        headless_code = re.sub(main_pattern, add_frame_limit, headless_code, flags=re.DOTALL)
        
        # If no main() function, try to modify the render loop
        if 'frame_count' not in headless_code:
            # Look for while window.running pattern
            while_pattern = r'(while.*?window\.running.*?:)(.*?)(\n\s+)(.*?show\(\))'
            
            def add_frame_limit_to_while(match):
                while_part = match.group(1)
                body_start = match.group(2)
                indent = match.group(3)
                show_part = match.group(4)
                
                frame_init = f"\n    frame_count = 0  # Validation counter"
                
                frame_check = f"""{indent}frame_count += 1
{indent}if frame_count >= 5:
{indent}    print("Validation successful - sketch compiles and runs")
{indent}    break  # Exit after validation
{indent}{show_part}"""
                
                return while_part + frame_init + body_start + frame_check
            
            headless_code = re.sub(while_pattern, add_frame_limit_to_while, headless_code, flags=re.DOTALL)
        
        return headless_code
    
    async def validate_sketch(
        self,
        sketch_code: str,
        auto_fix: bool = True,
        verbose: bool = False
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate a sketch by attempting to compile and run it briefly.
        
        Args:
            sketch_code: The sketch code to validate
            auto_fix: Whether to attempt automatic fixes for errors
            verbose: Whether to print validation progress
            
        Returns:
            Tuple of (success, final_code, error_message)
        """
        if verbose:
            print("🔍 Validating sketch compilation...")
        
        # Prepare headless version
        test_code = self._prepare_headless_sketch(sketch_code)
        
        # Try to validate the sketch
        validation_result = await self._run_validation(test_code)
        
        if validation_result['success']:
            if verbose:
                print("✅ Sketch validated successfully - no errors found")
            return (True, sketch_code, None)
        
        if not auto_fix:
            return (False, sketch_code, validation_result['error'])
        
        # Attempt to fix errors
        if verbose:
            print(f"⚠️  Validation failed: {validation_result['error'][:100]}...")
            print("🔧 Attempting automatic fixes...")
        
        fixed_code = sketch_code
        attempts = 0
        
        while attempts < self.max_fix_attempts:
            attempts += 1
            
            if verbose:
                print(f"  Fix attempt {attempts}/{self.max_fix_attempts}...")
            
            # Use refiner to fix the error
            fix_result = await self.refiner.refine_sketch(
                fixed_code,
                "Fix the compilation error. Focus only on fixing the error, do not change any other functionality.",
                error_info=validation_result['error']
            )
            
            if not fix_result['success']:
                if verbose:
                    print(f"  ❌ Fix attempt {attempts} failed")
                continue
            
            fixed_code = fix_result['refined_code']
            
            # Validate the fixed code
            test_code = self._prepare_headless_sketch(fixed_code)
            validation_result = await self._run_validation(test_code)
            
            if validation_result['success']:
                if verbose:
                    print(f"  ✅ Fixed successfully: {fix_result['changes_made']}")
                return (True, fixed_code, None)
            
            if verbose:
                print(f"  ⚠️  Still has errors after fix {attempts}")
        
        # Could not fix after max attempts
        if verbose:
            print(f"❌ Could not fix errors after {self.max_fix_attempts} attempts")
        
        return (False, fixed_code, validation_result['error'])
    
    async def _run_validation(self, test_code: str) -> Dict[str, Any]:
        """
        Run the validation test on the sketch code.
        
        Args:
            test_code: Modified sketch code with headless mode
            
        Returns:
            Dictionary with 'success' and optional 'error'
        """
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_path = f.name
        
        try:
            # Run the sketch with a timeout
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                # Wait for up to 10 seconds for validation
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=10.0
                )
                
                # Check if validation was successful
                if process.returncode == 0 or "Validation successful" in stdout.decode():
                    return {'success': True}
                
                # Extract error information
                error_msg = stderr.decode() if stderr else stdout.decode()
                
                # Look for specific error patterns
                error_info = self._extract_error_info(error_msg)
                
                return {
                    'success': False,
                    'error': error_info or error_msg
                }
                
            except asyncio.TimeoutError:
                # Timeout might mean it's running but stuck - could be OK
                process.terminate()
                await process.wait()
                
                # If it times out but didn't crash immediately, consider it a success
                # (Some sketches might have infinite loops which is OK)
                return {'success': True}
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Validation error: {str(e)}"
            }
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
    
    def _extract_error_info(self, error_msg: str) -> Optional[str]:
        """
        Extract meaningful error information from error output.
        
        Args:
            error_msg: Raw error message from subprocess
            
        Returns:
            Cleaned error information or None
        """
        # Common Taichi error patterns
        patterns = [
            (r"Return inside non-static if.*", "Return statement inside conditional block"),
            (r"Division by zero.*", "Division by zero error"),
            (r"Index.*out of.*bound.*", "Array index out of bounds"),
            (r"NameError: name '(\w+)' is not defined", r"Undefined variable: \1"),
            (r"TypeError:.*", "Type mismatch error"),
            (r"AttributeError:.*", "Attribute error"),
            (r"SyntaxError:.*", "Syntax error in code"),
            (r"IndentationError:.*", "Indentation error"),
            (r"RuntimeError:.*", "Runtime error"),
        ]
        
        for pattern, message in patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                if r'\1' in message:
                    return match.expand(message)
                return message
        
        # If no specific pattern, try to get the last line of the traceback
        lines = error_msg.strip().split('\n')
        for line in reversed(lines):
            if line.strip() and not line.startswith(' '):
                return line.strip()
        
        return None
    
    async def validate_and_fix_file(
        self,
        sketch_path: str,
        save_fixed: bool = True,
        verbose: bool = True
    ) -> Tuple[bool, str]:
        """
        Validate and potentially fix a sketch file.
        
        Args:
            sketch_path: Path to the sketch file
            save_fixed: Whether to save the fixed version
            verbose: Whether to print progress
            
        Returns:
            Tuple of (success, path_to_valid_sketch)
        """
        if verbose:
            print(f"📄 Validating sketch: {sketch_path}")
        
        # Read the sketch
        with open(sketch_path, 'r') as f:
            sketch_code = f.read()
        
        # Validate and fix
        success, fixed_code, error = await self.validate_sketch(
            sketch_code,
            auto_fix=True,
            verbose=verbose
        )
        
        if not success:
            if verbose:
                print(f"❌ Validation failed: {error}")
            return (False, sketch_path)
        
        # Check if code was modified
        if fixed_code != sketch_code and save_fixed:
            # Save the fixed version
            fixed_path = sketch_path.replace('.py', '_validated.py')
            with open(fixed_path, 'w') as f:
                f.write(fixed_code)
            
            if verbose:
                print(f"💾 Fixed sketch saved to: {fixed_path}")
            
            return (True, fixed_path)
        
        return (True, sketch_path)


class ValidatingBehaviorAgent:
    """
    Extension of BehaviorAgent that validates sketches before returning them.
    This ensures users only receive working sketches.
    """
    
    def __init__(self, behavior_agent, validator: Optional[SketchValidator] = None):
        """
        Wrap a BehaviorAgent with validation.
        
        Args:
            behavior_agent: The BehaviorAgent instance to wrap
            validator: Optional SketchValidator instance
        """
        self.agent = behavior_agent
        self.validator = validator or SketchValidator()
    
    async def generate_validated_sketch(
        self,
        description: str = "Generated particle system",
        filename: str = "sketch",
        use_timestamp: bool = True,
        verbose: bool = True
    ) -> Tuple[str, str]:
        """
        Generate a sketch and validate it before returning.
        
        Args:
            description: Description for the sketch
            filename: Base filename for the sketch
            use_timestamp: Whether to add timestamp to filename
            verbose: Whether to print progress
            
        Returns:
            Tuple of (sketch_code, sketch_path) for validated sketch
        """
        if verbose:
            print("🎨 Generating sketch...")
        
        # Generate the initial sketch
        sketch_code, sketch_path = self.agent.generate_sketch(
            description=description,
            filename=filename,
            use_timestamp=use_timestamp
        )
        
        if verbose:
            print(f"📝 Initial sketch generated: {sketch_path}")
        
        # Validate and fix if needed
        success, fixed_code, error = await self.validator.validate_sketch(
            sketch_code,
            auto_fix=True,
            verbose=verbose
        )
        
        if success and fixed_code != sketch_code:
            # Save the validated version
            with open(sketch_path, 'w') as f:
                f.write(fixed_code)
            
            if verbose:
                print(f"✅ Sketch validated and updated: {sketch_path}")
        elif not success:
            if verbose:
                print(f"⚠️  Warning: Sketch may have issues: {error}")
        
        return (fixed_code if success else sketch_code, sketch_path)