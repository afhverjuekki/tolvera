"""
Sketch Refiner Module - Two-stage architectural refactoring of generated Tölvera sketches
"""

import logging
import re
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from .prompt_loader import get_prompt_loader

from ..debug.tracing import get_collector, LLMCallData
from .model_factory import ModelFactory
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)


class AnalysisResponse(BaseModel):
    """Response from Stage 1: Analysis and planning."""
    implementation_plan: str = Field(description="Detailed implementation plan for refactoring")
    errors_found: str = Field(description="List of errors and issues found in the sketch")
    architectural_needs: str = Field(description="Architectural patterns that need to be added")
    
    
class RefinementResponse(BaseModel):
    """Response from Stage 2: Complete refactored sketch."""
    refined_code: str = Field(description="The complete refactored sketch code")
    changes_summary: str = Field(description="Summary of changes applied")


class SingleStageRefinementResponse(BaseModel):
    """Response from single-stage refinement/repair operations."""
    refined_code: str = Field(description="The complete refined sketch code")
    changes_made: str = Field(description="Summary of changes applied")
    warnings: Optional[str] = Field(None, description="Any warnings or potential issues")


class SketchRefiner:
    """
    Two-stage refiner for Tölvera sketches.
    
    Stage 1: Analyze sketch and create implementation plan
    Stage 2: Execute plan to generate refactored sketch
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        """
        Initialize the sketch refiner.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: Optional API key
        """
        self.model_name = model_name
        
        # Use model factory to create the appropriate model
        self.model = ModelFactory.create_model(model_name, api_key)
        self.provider = ModelFactory.get_provider_for_model(model_name)
        logger.info(f"SketchRefiner using provider '{self.provider}' with model '{model_name}'")
        
        # Load exemplar sketches
        self._load_exemplars()
        
        # Load context patterns for syntax guidance
        self._load_context_patterns()
        
        # Create agents for two-stage refinement
        self.analysis_agent = self._create_analysis_agent()
        self.implementation_agent = self._create_implementation_agent()
        
        # Create single-stage refinement agent for backward compatibility
        self.refinement_agent = self._create_single_stage_refinement_agent()
        
        # Initialize conversation manager for contextual memory
        self.conversation_manager = ConversationManager(
            max_history_length=10,
            max_context_tokens=2000
        )
    
    def _load_exemplars(self):
        """Load full exemplar sketches as reference."""
        exemplar_dir = Path(__file__).parent.parent.parent.parent.parent / "examples/generated_sketches/exemplars"
        
        exemplars = {}
        exemplar_files = {
            'slime': 'slime.py',
            'boids': 'boids.py', 
            'particle_life': 'particle-life.py'
        }
        
        for name, filename in exemplar_files.items():
            try:
                path = exemplar_dir / filename
                if path.exists():
                    with open(path, 'r') as f:
                        exemplars[name] = f.read()
                    logger.info(f"Loaded exemplar: {name}")
                else:
                    logger.warning(f"Exemplar not found: {path}")
                    exemplars[name] = f"# Exemplar {name} not found"
            except Exception as e:
                logger.error(f"Failed to load exemplar {name}: {e}")
                exemplars[name] = f"# Failed to load exemplar {name}"
        
        self.exemplars = exemplars
    
    def _load_context_patterns(self):
        """Load Taichi and behavior patterns for syntax guidance."""
        try:
            from ..context import taichi_patterns, patterns
            
            self.taichi_fundamentals = taichi_patterns.TAICHI_FUNDAMENTALS
            self.taichi_crashes = taichi_patterns.TAICHI_CRASH_FIXES
            self.movement_patterns = patterns.MOVEMENT_PATTERNS
            self.flocking_patterns = patterns.FLOCKING_PATTERNS
            
            # Try to load additional patterns
            try:
                from ..context import alife_patterns, drawing_patterns, temporal_patterns_extended
                self.ecosystem_patterns = alife_patterns.ECOSYSTEM_PATTERNS
                self.evolution_patterns = alife_patterns.EVOLUTION_PATTERNS
                self.drawing_patterns = drawing_patterns.DRAWING_PATTERNS if hasattr(drawing_patterns, 'DRAWING_PATTERNS') else ""
                self.temporal_patterns = temporal_patterns_extended.TEMPORAL_UPDATES if hasattr(temporal_patterns_extended, 'TEMPORAL_UPDATES') else ""
            except ImportError:
                self.ecosystem_patterns = ""
                self.evolution_patterns = ""
                self.drawing_patterns = ""
                self.temporal_patterns = ""
                
        except ImportError as e:
            logger.warning(f"Could not load context patterns: {e}")
            self.taichi_fundamentals = ""
            self.taichi_crashes = ""
            self.movement_patterns = ""
            self.flocking_patterns = ""
            self.ecosystem_patterns = ""
            self.evolution_patterns = ""
            self.drawing_patterns = ""
            self.temporal_patterns = ""
    
    def _create_analysis_agent(self) -> Agent:
        """Create the Stage 1 analysis agent."""
        
        # Load analysis prompt using PromptLoader
        loader = get_prompt_loader()
        system_prompt = loader.load_prompt("refinement/sketch_analysis_system.txt",
                                          slime_exemplar=self.exemplars.get('slime', '# Slime exemplar not loaded'),
                                          boids_exemplar=self.exemplars.get('boids', '# Pheromone exemplar not loaded'),
                                          particle_life_exemplar=self.exemplars.get('particle_life', '# Particle life exemplar not loaded'),
                                          taichi_crashes=self.taichi_crashes)
        
        # Log the assembled system prompt
        logger.info(f"[SKETCH_REFINER] Analysis Agent System Prompt assembled: {len(system_prompt)} chars")
        logger.debug(f"[SKETCH_REFINER] System prompt preview (first 500 chars): {system_prompt[:500]}")
        if logger.isEnabledFor(logging.DEBUG):
            # In debug mode, log the full prompt
            logger.debug(f"[SKETCH_REFINER] Full system prompt:\n{system_prompt}")
        
        agent = Agent(
            self.model,
            output_type=AnalysisResponse,
            system_prompt=system_prompt
        )
        
        return agent
    
    def _create_implementation_agent(self) -> Agent:
        """Create the Stage 2 implementation agent."""
        
        # Load implementation prompt using PromptLoader
        loader = get_prompt_loader()
        system_prompt = loader.load_prompt("refinement/sketch_implementation_system.txt",
                                          taichi_fundamentals=self.taichi_fundamentals,
                                          movement_patterns=self.movement_patterns,
                                          flocking_patterns=self.flocking_patterns)
        
        # Log the assembled system prompt
        logger.info(f"[SKETCH_REFINER] Implementation Agent System Prompt assembled: {len(system_prompt)} chars")
        logger.debug(f"[SKETCH_REFINER] System prompt preview (first 500 chars): {system_prompt[:500]}")
        if logger.isEnabledFor(logging.DEBUG):
            # In debug mode, log the full prompt
            logger.debug(f"[SKETCH_REFINER] Full system prompt:\n{system_prompt}")
        
        agent = Agent(
            self.model,
            output_type=RefinementResponse,
            system_prompt=system_prompt
        )
        
        return agent
    
    def _create_single_stage_refinement_agent(self) -> Agent:
        """Create the single-stage refinement agent for quick fixes and repairs."""
        
        # Load single-stage refinement prompt using PromptLoader
        loader = get_prompt_loader()
        system_prompt = loader.load_prompt("refinement/single_state_refinement_system.txt",
                                          taichi_crashes=self.taichi_crashes if hasattr(self, 'taichi_crashes') else '',
                                          taichi_fundamentals=self.taichi_fundamentals if hasattr(self, 'taichi_fundamentals') else '')
        
        # Log the assembled system prompt
        logger.info(f"[SKETCH_REFINER] Single-Stage Agent System Prompt assembled: {len(system_prompt)} chars")
        logger.debug(f"[SKETCH_REFINER] System prompt preview (first 500 chars): {system_prompt[:500]}")
        if logger.isEnabledFor(logging.DEBUG):
            # In debug mode, log the full prompt
            logger.debug(f"[SKETCH_REFINER] Full system prompt:\n{system_prompt}")
        
        agent = Agent(
            self.model,
            output_type=SingleStageRefinementResponse,
            system_prompt=system_prompt
        )
        
        return agent
    
    async def analyze_sketch(
        self,
        sketch_code: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Stage 1: Analyze the sketch and create an implementation plan.
        
        Args:
            sketch_code: The initial sketch to analyze
            description: User's description of desired behavior
            
        Returns:
            Dictionary with analysis results and plan
        """
        logger.info("Stage 1: Analyzing sketch and creating implementation plan")
        
        collector = get_collector()
        
        with collector.trace_node("analyze_sketch", "analysis", description=description) as node:
            
            prompt = f"""Analyze this Tölvera sketch and create a detailed implementation plan.

USER'S GOAL: {description}

DRAFT SKETCH:
```python
{sketch_code}
```

Analyze the sketch against the user's goal and the architectural patterns in the exemplars. Create a comprehensive implementation plan that addresses ALL issues and missing features."""
            
            try:
                with collector.trace_node("analysis_llm_call", "llm_call", model=self.model_name) as llm_node:
                    # Log the user prompt being sent
                    logger.info(f"[SKETCH_REFINER] Running analysis with user prompt: {len(prompt)} chars")
                    logger.debug(f"[SKETCH_REFINER] User prompt: {prompt}")
                    
                    result = await self.analysis_agent.run(prompt)
                    
                    analysis = {
                        'success': True,
                        'implementation_plan': result.output.implementation_plan,
                        'errors_found': result.output.errors_found,
                        'architectural_needs': result.output.architectural_needs
                    }
                    
                    # Create LLM call data for tracing
                    llm_data = LLMCallData(
                        model=self.model_name,
                        provider=self.provider,
                        user_prompt=prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        system_prompt="Analysis agent system prompt",
                        full_prompt=prompt,  # Store the full prompt for complete trace viewing
                        parsed_response=analysis
                    )
                    if llm_node:
                        llm_node.llm_call = llm_data
                    
                    logger.info("Analysis complete - plan created")
                    return analysis
                    
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                if node:
                    node.set_error(str(e))
                return {
                    'success': False,
                    'error': str(e)
                }
    
    async def implement_refinement(
        self,
        sketch_code: str,
        plan: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Stage 2: Implement the refactoring plan.
        
        Args:
            sketch_code: The original sketch
            plan: Implementation plan from Stage 1
            description: User's description
            
        Returns:
            Dictionary with refactored code
        """
        logger.info("Stage 2: Implementing refactoring based on plan")
        
        collector = get_collector()
        
        with collector.trace_node("implement_refinement", "implementation", description=description) as node:
            
            prompt = f"""Implement this refactoring plan to create a complete, working Tölvera sketch.

USER'S GOAL: {description}

IMPLEMENTATION PLAN:
{plan}

ORIGINAL SKETCH TO REFACTOR:
```python
{sketch_code}
```

Following the plan exactly, provide the COMPLETE refactored sketch with all errors fixed and architectural enhancements added."""
            
            try:
                with collector.trace_node("implementation_llm_call", "llm_call", model=self.model_name) as llm_node:
                    # Log the user prompt being sent
                    logger.info(f"[SKETCH_REFINER] Running implementation with user prompt: {len(prompt)} chars")
                    logger.debug(f"[SKETCH_REFINER] User prompt: {prompt}")
                    
                    result = await self.implementation_agent.run(prompt)
                    
                    refinement = {
                        'success': True,
                        'refined_code': result.output.refined_code,
                        'changes_summary': result.output.changes_summary
                    }
                    
                    # Create LLM call data
                    llm_data = LLMCallData(
                        model=self.model_name,
                        provider=self.provider,
                        user_prompt=prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        system_prompt="Implementation agent system prompt",
                        full_prompt=prompt,  # Store the full prompt for complete trace viewing
                        parsed_response={'changes_summary': result.output.changes_summary}
                    )
                    if llm_node:
                        llm_node.llm_call = llm_data
                    
                    logger.info("Implementation complete - sketch refactored")
                    return refinement
                    
            except Exception as e:
                logger.error(f"Implementation failed: {e}")
                if node:
                    node.set_error(str(e))
                return {
                    'success': False,
                    'error': str(e),
                    'refined_code': sketch_code  # Return original on failure
                }
    
    async def refine_sketch_two_stage(
        self,
        sketch_code: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Complete two-stage refinement process.
        
        Args:
            sketch_code: Initial sketch to refine
            description: User's description of desired behavior
            
        Returns:
            Dictionary with refined code and process details
        """
        logger.info(f"Starting two-stage refinement for: {description}")
        
        collector = get_collector()
        
        # Wrap the entire two-stage process in a parent trace node
        with collector.trace_node("two_stage_refinement", "refinement", 
                                  description=description,
                                  refinement_type="architectural") as parent_node:
            
            # Stage 1: Analyze and plan
            analysis = await self.analyze_sketch(sketch_code, description)
            
            if not analysis['success']:
                logger.error(f"Stage 1 failed: {analysis.get('error')}")
                if parent_node:
                    parent_node.set_error(f"Analysis failed: {analysis.get('error')}")
                return {
                    'success': False,
                    'error': f"Analysis failed: {analysis.get('error')}",
                    'refined_code': sketch_code
                }
            
            # Add analysis metadata to parent node
            if parent_node:
                parent_node.metadata = {
                    'stage1_complete': True,
                    'errors_found': analysis.get('errors_found', ''),
                    'architectural_needs': analysis.get('architectural_needs', '')
                }
            
            # Stage 2: Implement the plan
            implementation = await self.implement_refinement(
                sketch_code=sketch_code,
                plan=analysis['implementation_plan'],
                description=description
            )
            
            if not implementation['success']:
                logger.error(f"Stage 2 failed: {implementation.get('error')}")
                if parent_node:
                    parent_node.set_error(f"Implementation failed: {implementation.get('error')}")
                return {
                    'success': False,
                    'error': f"Implementation failed: {implementation.get('error')}",
                    'refined_code': sketch_code
                }
            
            # Success! Update parent node with complete information
            if parent_node:
                parent_node.output_data = {
                    'success': True,
                    'changes_summary': implementation['changes_summary'],
                    'stages_completed': 2,
                    'refined': True
                }
            
            # Return the refactored sketch
            return {
                'success': True,
                'refined_code': implementation['refined_code'],
                'analysis': analysis,
                'changes_summary': implementation['changes_summary']
            }
    
    # Compatibility method for existing integration
    async def refine_to_architecture(
        self,
        sketch_code: str,
        description: str,
        pattern: Optional[str] = None  # Ignored in new approach
    ) -> Dict[str, Any]:
        """
        Compatibility wrapper for existing integration.
        Uses the two-stage refinement process.
        """
        result = await self.refine_sketch_two_stage(sketch_code, description)
        
        # Map to expected format
        return {
            'success': result['success'],
            'refined_code': result.get('refined_code', sketch_code),
            'changes_made': result.get('changes_summary', 'No changes'),
            'warnings': result.get('error') if not result['success'] else None
        }
    
    # Simplified pattern detection for backwards compatibility
    def detect_architectural_pattern(self, description: str, sketch_code: str) -> Dict[str, Any]:
        """
        Simplified detection - always returns that refinement is needed.
        The two-stage process will handle all patterns.
        """
        return {
            'primary_pattern': 'general',  # Generic pattern
            'scores': {},
            'confidence': 1.0,  # Always confident we can improve
            'needs_refinement': True  # Always try to refine
        }
    
    # Single-stage refinement methods for backward compatibility
    async def refine_sketch(
        self,
        sketch_code: str,
        refinement_request: str,
        error_info: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Single-stage sketch refinement for quick fixes and feature additions.
        Used by the textual UI for incremental refinements.
        
        Args:
            sketch_code: The current sketch code to refine
            refinement_request: What the user wants to change
            error_info: Optional error message if the sketch crashed
            
        Returns:
            Dictionary with refined_code, changes_made, warnings, and success flag
        """
        logger.info(f"Single-stage refinement: {refinement_request}")
        
        collector = get_collector()
        refinement_type = "error_correction" if error_info else "feature_addition"
        
        with collector.trace_node(f"refine_{refinement_type}", "refinement",
                                  request=refinement_request,
                                  has_error=bool(error_info)) as node:
            
            # Check if this looks like a signature mismatch error
            signature_error_patterns = [
                "expects", "arguments", "got", "TypeError", "missing", 
                "required positional", "takes", "given", "parameter"
            ]
            has_signature_issue = error_info and any(pattern in error_info.lower() for pattern in signature_error_patterns)
            
            # Build conversation context
            conversation_context = self.conversation_manager.get_conversation_context()
            
            prompt = f"""Refine this Tölvera sketch based on the user's request.

{conversation_context}

## CURRENT REQUEST
{refinement_request}

## CURRENT SKETCH
```python
{sketch_code}
```

IMPORTANT: Review the conversation history above to understand the user's evolving requirements and avoid repeating previous mistakes or undoing previous improvements."""
            
            if error_info:
                prompt += f"""

ERROR INFORMATION:
The sketch crashed with this error:
{error_info}

Fix this error as part of the refinement.
"""
                
                # Add specialized context for signature mismatches
                if has_signature_issue:
                    prompt += f"""

DETECTED: Function signature mismatch error!
{self._build_signature_mismatch_context()}
"""
            
            prompt += """

Apply the requested changes while:
1. Preserving all existing behaviors not mentioned in the request
2. Following Taichi best practices (no return in conditionals!)
3. Maintaining the sketch structure
4. Ensuring the sketch will run without errors
5. Verifying ALL function calls match their definitions exactly

Return the COMPLETE refined sketch code.
"""
            
            try:
                with collector.trace_node("refinement_llm_call", "llm_call",
                                        model=self.model_name) as llm_node:
                    # Log the user prompt being sent
                    logger.info(f"[SKETCH_REFINER] Running single-stage refinement with user prompt: {len(prompt)} chars")
                    logger.debug(f"[SKETCH_REFINER] User prompt: {prompt}")
                    
                    result = await self.refinement_agent.run(prompt)
                    
                    refined_code = result.output.refined_code
                    changes_made = result.output.changes_made
                    warnings = result.output.warnings
                    
                    # Sanitize the code
                    refined_code = self._sanitize_refined_code(refined_code)
                    
                    # Validate the code
                    validation_issues = self._validate_refined_code(refined_code)
                    if validation_issues:
                        warnings = (warnings or "") + f"\nValidation: {validation_issues}"
                    
                    # Create LLM call data
                    llm_data = LLMCallData(
                        model=self.model_name,
                        provider=self.provider,
                        user_prompt=prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        system_prompt="Single-stage refinement agent",
                        full_prompt=prompt,  # Store the full prompt for complete trace viewing
                        parsed_response={
                            'changes_made': changes_made,
                            'warnings': warnings
                        }
                    )
                    if llm_node:
                        llm_node.llm_call = llm_data
                    
                    # Store conversation entry for context
                    interaction_type = "repair" if error_info else "refinement"
                    self.conversation_manager.add_conversation_entry(
                        user_request=refinement_request,
                        agent_response_summary=changes_made,
                        interaction_type=interaction_type,
                        success=True,
                        pydantic_messages=result.new_messages() if hasattr(result, 'new_messages') else None
                    )
                    
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
                
                # Store failed conversation entry for context
                interaction_type = "repair" if error_info else "refinement"
                self.conversation_manager.add_conversation_entry(
                    user_request=refinement_request,
                    agent_response_summary=f"Failed: {str(e)}",
                    interaction_type=interaction_type,
                    success=False,
                    error_info=str(e)
                )
                    
                return {
                    'success': False,
                    'error': str(e),
                    'refined_code': sketch_code,
                    'changes_made': "No changes due to error",
                    'warnings': f"Refinement failed: {str(e)}"
                }
    
    async def repair_sketch(
        self,
        sketch_code: str,
        error_logs: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Repair a sketch that failed to execute based on error logs.
        Specialized method for error recovery used by the textual UI.
        
        Args:
            sketch_code: The sketch code that failed
            error_logs: Error messages from failed execution
            additional_context: Optional additional context
            
        Returns:
            Dictionary with refined_code, changes_made, warnings, and success flag
        """
        logger.info("Repairing sketch based on error logs")
        
        # Build specialized repair request
        repair_request = f"""Fix the following errors in this Tölvera sketch:

ERROR LOGS:
{error_logs}

CRITICAL FIXES TO APPLY:
1. If "Return inside non-static if" - restructure ALL expert functions to use single return
2. If division errors - add safety checks for zero distances
3. If undefined variables - ensure all variables declared before use
4. If type errors - verify Taichi types (ti.f32, ti.i32, ti.math.vec2)
5. If index errors - check particle bounds and array sizes
6. If "function expects X arguments but got Y" or similar - THIS IS A FUNCTION SIGNATURE MISMATCH:
   - Find the function definition (the @ti.func or def line)
   - Count the parameters it expects
   - Find ALL calls to that function
   - Fix each call to pass the correct number and type of arguments
   - CRITICAL: Modify the CALL sites, NOT the function definition
   - Example: If particle_life_interactions expects (pos, vel, mass, species, idx), 
     don't pass tv.p.field[i] - instead extract and pass pos, vel, mass, species, i separately

FUNCTION SIGNATURE VERIFICATION PROTOCOL:
- For EVERY @ti.func and @ti.kernel in the sketch:
  1. Note its signature (parameter names, types, count)
  2. Find ALL places where it's called
  3. Verify each call matches the signature EXACTLY
  4. If mismatch found, fix the CALL, not the definition

Remember: You are an expert at fixing Taichi and Tölvera errors. Apply precise fixes, especially for function signature mismatches."""
        
        if additional_context:
            repair_request += f"\n\nAdditional context: {additional_context}"
        
        # Use refine_sketch with error info
        return await self.refine_sketch(
            sketch_code=sketch_code,
            refinement_request=repair_request,
            error_info=error_logs
        )
    
    def _build_signature_mismatch_context(self) -> str:
        """
        Build specialized context for function signature mismatch errors.
        
        Returns:
            String containing detailed guidance for fixing signature mismatches
        """
        return """
## FUNCTION SIGNATURE MISMATCH REPAIR GUIDE

### Common Signature Mismatch Patterns

1. Passing Structs Instead of Fields
   WRONG: particle_interactions(tv.p.field[i], tv.p.field[j])
   RIGHT: particle_interactions(tv.p.field[i].pos, tv.p.field[i].vel, tv.p.field[i].mass, tv.p.field[i].species, i)

2. Wrong Parameter Order
   WRONG: calculate_force(mass, vel, pos, species)
   RIGHT: calculate_force(pos, vel, mass, species)  # Match the definition order

3. Missing Parameters
   WRONG: apply_behavior(pos, vel)
   RIGHT: apply_behavior(pos, vel, mass, species, particle_idx)

4. Extra Parameters
   WRONG: simple_gravity(pos, vel, mass, species, extra_param)
   RIGHT: simple_gravity(pos, vel, mass)  # Only pass what's expected

### Verification Checklist
- [ ] Count parameters in function definition
- [ ] Check parameter types in definition
- [ ] Find all function calls
- [ ] Verify each call matches exactly
- [ ] Extract struct fields if needed
- [ ] Pass scalars/vectors, not structs

### Example Fix Pattern
```python
# If you see this pattern:
force = expert_func(tv.p.field[i])  # Passing whole particle

# Check the function definition:
@ti.func
def expert_func(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, idx: ti.i32):
    # Expects 5 specific parameters

# Fix the call:
pos_i = tv.p.field[i].pos
vel_i = tv.p.field[i].vel  
mass_i = tv.p.field[i].mass
species_i = tv.p.field[i].species
force = expert_func(pos_i, vel_i, mass_i, species_i, i)
```
"""
    
    def _sanitize_refined_code(self, code: str) -> str:
        """
        Sanitize refined code to fix common LLM formatting issues.
        
        Args:
            code: The refined sketch code from LLM
            
        Returns:
            Cleaned code
        """
        # Remove trailing whitespace
        code = code.rstrip()
        
        # Remove markdown code blocks if present
        if code.startswith('```python'):
            code = code[9:]  # Remove ```python
        if code.startswith('```'):
            code = code[3:]  # Remove ```
        if code.endswith('```'):
            code = code[:-3]  # Remove trailing ```
            
        # Remove trailing quotes
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
        
        # Check for return in conditionals
        if_return_pattern = r'if\s+.*:\s*\n\s*return\s+'
        if re.search(if_return_pattern, code):
            warnings.append("Potential 'return in conditional' detected")
        
        # Check for division without safety
        unsafe_div_pattern = r'\/\s*(?:dist|norm|length)(?!\s*if)'
        if re.search(unsafe_div_pattern, code):
            warnings.append("Division without safety check")
        
        # Check main components
        if 'def main(' not in code:
            warnings.append("Main function not found")
        if '@tv.render' not in code:
            warnings.append("Render decorator not found")
        if 'tv = Tolvera(' not in code:
            warnings.append("Tölvera initialization not found")
        
        return "; ".join(warnings) if warnings else None
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history. Used when resetting the UI."""
        self.conversation_manager.clear_history()
        logger.info("SketchRefiner conversation history cleared")