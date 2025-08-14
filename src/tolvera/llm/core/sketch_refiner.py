"""
Sketch Refiner Module - Iterative refinement of generated Tölvera sketches
"""

import logging
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import os
from ..debug.tracing import get_collector, LLMCallData
from .model_factory import ModelFactory

logger = logging.getLogger(__name__)


class RefinementResponse(BaseModel):
    """Response from the LLM for sketch refinement."""
    refined_code: str = Field(description="The complete refined sketch code")
    changes_made: str = Field(description="Summary of changes applied")
    warnings: Optional[str] = Field(None, description="Any warnings or potential issues")


class SketchRefiner:
    """
    Refines existing Tölvera sketches based on user feedback.
    
    This class enables iterative improvement of generated sketches without
    going through the full synthesis pipeline again.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        """
        Initialize the sketch refiner.
        
        Args:
            model_name: Name of the LLM model to use (can include provider prefix)
            api_key: Optional API key (will use env var if not provided)
        """
        self.model_name = model_name
        
        # Use model factory to create the appropriate model
        self.model = ModelFactory.create_model(model_name, api_key)
        self.provider = ModelFactory.get_provider_for_model(model_name)
        logger.info(f"SketchRefiner using provider '{self.provider}' with model '{model_name}'")
        
        # Initialize the prompt builder - this is what we need!
        from ..core.prompts import ContextAwarePromptBuilder
        self.prompt_builder = ContextAwarePromptBuilder()
        
        # Load context patterns FIRST (for backward compatibility)
        self._load_context_patterns()
        
        # Create the refinement agent AFTER context patterns are loaded
        self.refinement_agent = self._create_refinement_agent()
    
    def _load_context_patterns(self):
        """Load refinement-specific patterns (kept for backward compatibility)."""
        # Try to load refinement-specific patterns if they exist
        try:
            from ..context import refinement_patterns
            self.refinement_patterns = refinement_patterns.REFINEMENT_PATTERNS
        except ImportError:
            self.refinement_patterns = ""
        
        # Note: All other context is now loaded via the ContextAwarePromptBuilder
    
    def _create_refinement_agent(self) -> Agent:
        """Create the pydantic-ai agent for sketch refinement."""
        
        # Build refinement-specific instructions
        refinement_instructions = """You are an expert at refining Tölvera particle simulations.

Your role is to:
1. PRESERVE all existing functionality while applying requested changes
2. Apply targeted refinements based on user feedback
3. ADD NEW FEATURES including artificial life behaviors and drawing operations
4. Fix any errors or issues mentioned
5. Follow Taichi and Tölvera best practices

COMMON REFINEMENTS:

Force Adjustments:
- "Make gravity stronger" → Increase magnitude of downward force (more negative Y)
- "Reduce attraction" → Decrease force multiplier
  - "Add drift" → Add random force component
  
  Speed Changes:
  - "Move faster" → Increase velocity multipliers or speed values
  - "Slow down" → Add more damping (multiply vel by 0.9-0.95)
  - "Stop movement" → Set velocities to 0 or increase damping
  
  Visual Changes:
  - "Make bigger" → Increase tv.p.field[i].size values
  - "Change color" → Modify tv.s.species.field[x].rgba values
  - "Add transparency" → Reduce alpha (4th component) in rgba
  
  Behavioral Additions:
  - "Also repel" → Add new repulsion expert function
  - "Add trails" → Add drawing behavior that samples positions
  - "Make them flock" → Add alignment, cohesion, separation
  
  Bug Fixes:
  - "Not moving" → Check: forces applied? dt multiplied? position updated?
  - "Drawing doesn't appear" → Check: draw() kernel called? correct coordinates?
  - "Crashes" → Look for: return in conditional, division by zero, out of bounds
  
  Temporal/Drawing Adjustments:
  - "Draw more often" → Reduce time thresholds
  - "Change frequency" → Adjust modulo or timing values
  - "Random offset different" → Modify random range multipliers
  
  STRUCTURE TO MAINTAIN:
  1. Configuration section (kwargs setup)
  2. Particle initialization
  3. State initialization (if present)
  4. Expert functions (@ti.func)
  5. Integration kernel (apply_all_experts)
  6. Utility functions (if present)
  7. Drawing functions (if present)
  8. Render loop (@tv.render)
  
  ALWAYS return the COMPLETE refined sketch, not just the changed parts."""
        
        # Use the centralized prompt builder to get all the context
        # For refinement, we want ALL available contexts since we don't know what will be needed
        all_contexts = [
            'core_api', 'taichi', 'taichi_fundamentals', 'taichi_crashes',
            'movement', 'flocking', 'interaction', 'temporal', 'cellular',
            'ecosystem', 'evolution', 'swarm', 'vera_patterns', 'vera_interactions',
            'species_interactions', 'drawing', 'drawing_api', 'alife_patterns',
            'state_access', 'boundaries', 'initialization', 'species_initialization',
            'temporal_updates', 'iml_patterns'
        ]
        
        base_synthesis_prompt = self.prompt_builder.build_synthesis_prompt(
            description="sketch refinement",  # Generic description
            available_states={},  # No specific states for general refinement
            include_contexts=all_contexts,  # Include ALL contexts for refinement
            constrained=False,  # We want full context, not constrained JSON
            context={}  # No specific pattern type
        )
        
        # Combine the synthesis prompt with our refinement-specific instructions
        full_system_prompt = f"{refinement_instructions}\n\n{base_synthesis_prompt}"
        
        # Store the system prompt for debugging/testing
        self._full_system_prompt = full_system_prompt
        
        # Create agent with the complete system prompt
        agent = Agent(
            self.model,
            output_type=RefinementResponse,
            system_prompt=full_system_prompt
        )
        
        return agent
    
    async def refine_sketch(
        self,
        sketch_code: str,
        refinement_request: str,
        error_info: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refine an existing sketch based on user feedback.
        
        Args:
            sketch_code: The current sketch code to refine
            refinement_request: What the user wants to change
            error_info: Optional error message if the sketch crashed
            
        Returns:
            Dictionary with:
                - refined_code: The refined sketch code
                - changes_made: Summary of what was changed
                - warnings: Any potential issues to be aware of
                - success: Whether refinement succeeded
        """
        logger.info(f"Refining sketch with request: {refinement_request}")
        
        collector = get_collector()
        
        # Determine refinement type for tracing
        refinement_type = "error_correction" if error_info else "behavior_modification"
        node_name = f"refine_{refinement_type}"
        
        with collector.trace_node(node_name, "refinement", 
                                  refinement_request=refinement_request,
                                  has_error=bool(error_info)) as node:
            
            # Build the refinement prompt
            prompt = f"""Refine this Tölvera sketch based on the user's request.

CURRENT SKETCH:
```python
{sketch_code}
```

USER REQUEST: {refinement_request}
"""
            
            if error_info:
                prompt += f"""

ERROR INFORMATION:
The sketch crashed with this error:
{error_info}

Fix this error as part of the refinement.
"""
            
            prompt += """

Apply the requested changes while:
1. Preserving all existing behaviors not mentioned in the request
2. Following Taichi best practices (no return in conditionals!)
3. Maintaining the sketch structure
4. Ensuring the sketch will run without errors

Return the COMPLETE refined sketch code.
"""
            
            try:
                # Trace the LLM call
                # Get the full system prompt from the agent
                system_prompt_text = ""
                # Use the full system prompt we stored during agent creation
                # This includes ALL the context from the ContextAwarePromptBuilder
                if hasattr(self, '_full_system_prompt'):
                    system_prompt_text = self._full_system_prompt
                else:
                    # Fallback if somehow the prompt wasn't stored
                    system_prompt_text = "Refining Tölvera particle simulations"
                
                with collector.trace_node("refinement_llm_call", "llm_call",
                                        model=self.model_name) as llm_node:
                    
                    # Run the refinement agent
                    result = await self.refinement_agent.run(prompt)
                    
                    # Extract the response
                    refined_code = result.output.refined_code
                    changes_made = result.output.changes_made
                    warnings = result.output.warnings
                    
                    # Sanitize refined code to fix common LLM formatting issues
                    refined_code = self._sanitize_refined_code(refined_code)
                    
                    # Create LLM call data with full prompt
                    full_prompt_text = f"{system_prompt_text}\n\n{prompt}"
                    llm_data = LLMCallData(
                        model=self.model_name,
                        provider=self.provider,
                        user_prompt=prompt,
                        system_prompt=system_prompt_text,
                        full_prompt=full_prompt_text,  # Add the combined prompt
                        parsed_response={
                            'refined_code': refined_code[:500] + '...' if len(refined_code) > 500 else refined_code,
                            'changes_made': changes_made,
                            'warnings': warnings
                        }
                    )
                    llm_node.llm_call = llm_data
                
                # Quick validation of the refined code
                validation_issues = self._validate_refined_code(refined_code)
                if validation_issues:
                    warnings = (warnings or "") + f"\nValidation warnings: {validation_issues}"
                
                # Update main node with results
                if node:
                    node.output_data = {
                        'changes_made': changes_made,
                        'warnings': warnings,
                        'validation_issues': validation_issues,
                        'code_length': len(refined_code)
                    }
                
                return {
                    'success': True,
                    'refined_code': refined_code,
                    'changes_made': changes_made,
                    'warnings': warnings
                }
                
            except Exception as e:
                logger.error(f"Refinement failed: {e}")
                
                if node:
                    node.set_error(str(e))
                
                return {
                    'success': False,
                    'error': str(e),
                    'refined_code': sketch_code,  # Return original on failure
                    'changes_made': "No changes applied due to error",
                    'warnings': f"Refinement failed: {str(e)}"
                }
    
    def _sanitize_refined_code(self, code: str) -> str:
        """
        Sanitize refined code to fix common LLM formatting issues.
        
        Args:
            code: The refined sketch code from LLM
            
        Returns:
            Cleaned code with formatting issues fixed
        """
        # Remove trailing whitespace
        code = code.rstrip()
        
        # Remove trailing triple quotes that LLMs sometimes add
        if code.endswith('"""'):
            code = code[:-3].rstrip()
        elif code.endswith("'''"):
            code = code[:-3].rstrip()
        
        return code
    
    def _validate_refined_code(self, code: str) -> Optional[str]:
        """
        Perform basic validation on refined code.
        
        Args:
            code: The refined sketch code
            
        Returns:
            Warning message if issues found, None otherwise
        """
        warnings = []
        
        # Check for return statements inside conditionals
        if_return_pattern = r'if\s+.*:\s*\n\s*return\s+'
        if re.search(if_return_pattern, code):
            warnings.append("Potential 'return in conditional' detected - may crash")
        
        # Check for division without safety check
        unsafe_div_pattern = r'\/\s*(?:dist|norm|length)(?!\s*if)'
        if re.search(unsafe_div_pattern, code):
            warnings.append("Division by distance without safety check detected")
        
        # Check if main components are present
        if 'def main(' not in code:
            warnings.append("Main function not found")
        if '@tv.render' not in code:
            warnings.append("Render function not found")
        if 'tv = Tolvera(' not in code:
            warnings.append("Tölvera initialization not found")
        
        return "; ".join(warnings) if warnings else None
    
    async def repair_sketch(
        self,
        sketch_code: str,
        error_logs: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Repair a sketch that failed to execute based on error logs.
        
        Args:
            sketch_code: The current sketch code that failed
            error_logs: The error messages and logs from the failed execution
            additional_context: Optional additional context from the user
            
        Returns:
            Dictionary with:
                - refined_code: The repaired sketch code
                - changes_made: Summary of what was fixed
                - warnings: Any potential issues to be aware of
                - success: Whether repair succeeded
        """
        logger.info(f"Repairing sketch based on error logs")
        
        # Build a repair-focused request with expert context
        repair_request = f"""You are a world-class expert in Python, Taichi, and the Tölvera particle simulation ecosystem. 
You have deep, specialized knowledge of:

1. Taichi Programming: GPU-accelerated computing, Taichi kernels, fields, and functions
2. Tölvera Framework: Particle systems, species management, custom states, and integration kernels
3. Python: Advanced Python programming, debugging, and error resolution
4. Particle Simulations: Physics simulations, force calculations, emergent behaviors

CRITICAL TAICHI RULES TO FOLLOW:
- NEVER use return statements inside conditional blocks (causes "Return inside non-static if" crash)
- Always declare result variables at the start of functions and modify them in conditionals
- Use ti.math.vec2() or ti.Vector([x, y]) for force calculations
- Handle division by zero explicitly with safety checks
- Use Taichi math functions (ti.sin, ti.cos) not Python's math module inside kernels
- Vector components accessed via indexing: vec[0], vec[1] (not .x, .y for ti.Vector)

The sketch crashed with the following errors:

{error_logs}

COMMON ERROR PATTERNS AND FIXES:
- "Return inside non-static if": Move return statement outside conditionals
- "Division by zero": Add safety checks like "if dist > 0.001:"
- "AttributeError on vec.x": Use vec[0] instead for ti.Vector types
- "NameError": Check variable definitions and scope
- "IndexError": Verify array bounds and particle counts
- "TypeError": Ensure correct Taichi types (ti.f32, ti.i32, etc.)

Please analyze the error logs carefully and fix ALL issues to ensure the sketch runs without crashing.
Focus on:
1. Identifying the exact error type and location
2. Applying Taichi-specific fixes (especially for return statements)
3. Preserving all existing functionality while fixing the errors
4. Following Tölvera and Taichi best practices"""
        
        if additional_context:
            repair_request += f"\n\nAdditional context from user: {additional_context}"
        
        repair_request += "\n\nReturn the COMPLETE repaired sketch code with all errors fixed."
        
        # Use the existing refine_sketch method with the error info
        return await self.refine_sketch(
            sketch_code=sketch_code,
            refinement_request=repair_request,
            error_info=error_logs
        )
    
    def extract_expert_functions(self, sketch_code: str) -> Dict[str, str]:
        """
        Extract expert functions from sketch for analysis.
        
        Args:
            sketch_code: The sketch code to analyze
            
        Returns:
            Dictionary mapping expert names to their code
        """
        experts = {}
        
        # Pattern to match @ti.func definitions
        func_pattern = r'@ti\.func\s*\n\s*def\s+(\w+)\s*\([^)]*\)[^:]*:\s*\n((?:.*\n)*?)(?=\n@|\ndef|\n\s*#\s*===|\Z)'
        
        matches = re.finditer(func_pattern, sketch_code, re.MULTILINE)
        for match in matches:
            name = match.group(1)
            # Extract the full function definition
            experts[name] = f"@ti.func\ndef {name}{match.group(0)[match.group(0).find('('):]}"
        
        return experts
    
    def analyze_sketch_structure(self, sketch_code: str) -> Dict[str, Any]:
        """
        Analyze the structure of a sketch for better refinement.
        
        Args:
            sketch_code: The sketch code to analyze
            
        Returns:
            Dictionary with structural information
        """
        structure = {
            'has_states': 'tv.s.set(' in sketch_code or 'llm_' in sketch_code,
            'has_drawing': '@ti.kernel\ndef draw(' in sketch_code,
            'has_utility': 'utility' in sketch_code.lower(),
            'num_species': 1,
            'experts': self.extract_expert_functions(sketch_code),
            'has_temporal': 'temporal' in sketch_code.lower()
        }
        
        # Try to detect number of species
        species_match = re.search(r"kwargs\['species'\]\s*=\s*(\d+)", sketch_code)
        if species_match:
            structure['num_species'] = int(species_match.group(1))
        
        return structure