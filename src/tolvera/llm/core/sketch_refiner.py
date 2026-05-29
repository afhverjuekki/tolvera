import os
import re
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from ..prompts.prompt_loader import get_prompt_loader

from ..debug.tracing import get_collector, LLMCallData
from .llm_factory import ModelFactory
from .conversation_manager import ConversationManager


class AnalysisResponse(BaseModel):
    """Response from Stage 1: Analysis and planning."""
    implementation_plan: str = Field(description="Detailed implementation plan for refactoring")
    errors_found: str = Field(description="List of errors and issues found in the sketch")
    architectural_needs: str = Field(description="Architectural patterns that need to be added")
    
    
# Refinement/repair returns a COMPLETE sketch (~25-40 KB ≈ 8-12k output tokens)
# in a single response. The provider default max output is far lower, so the
# response gets truncated mid-sketch: in structured-output mode the model spends
# its budget on reasoning text and then emits an empty tool call ({}), failing
# validation with "Exceeded maximum retries for output validation". Opus 4.8 hits
# this where smaller-output models (per-expert generation) do not. Give the
# whole-sketch refinement calls enough room to emit the full file.
REFINEMENT_MAX_TOKENS = 20000


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
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the sketch refiner.

        Args:
            model_name: Name of the LLM model to use
            api_key: Optional API key
        """
        if model_name is None:
            model_name = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
        self.model_name = model_name
        
        # Use model factory to create the appropriate model
        self.model = ModelFactory.create_model(model_name, api_key)
        self.provider = ModelFactory.get_provider_for_model(model_name)
        
        # Load exemplar sketches
        self._load_exemplars()
        
        # Load context patterns for syntax guidance
        self._load_context_patterns()
        
        # Defer agent creation until needed to avoid premature context selection
        self.analysis_agent = None
        self.implementation_agent = None
        self.refinement_agent = None
        
        # Initialize conversation manager for contextual memory
        self.conversation_manager = ConversationManager(
            max_history_length=10,
            max_context_tokens=2000
        )
    
    def _load_exemplars(self):
        """Load full exemplar sketches as reference."""
        exemplar_dir = Path(__file__).parent.parent.parent.parent.parent / "examples/exemplars"
        
        exemplars = {}
        exemplar_files = {
            'slime': 'slime.py',
            'boids': 'boids.py', 
            'particle_life': 'particle_life.py'
        }
        
        for name, filename in exemplar_files.items():
            try:
                path = exemplar_dir / filename
                if path.exists():
                    with open(path, 'r') as f:
                        exemplars[name] = f.read()
                else:
                    exemplars[name] = f"# Exemplar {name} not found"
            except Exception:
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
                
        except ImportError:
            self.taichi_fundamentals = ""
            self.taichi_crashes = ""
            self.movement_patterns = ""
            self.flocking_patterns = ""
            self.ecosystem_patterns = ""
            self.evolution_patterns = ""
            self.drawing_patterns = ""
            self.temporal_patterns = ""
    
    async def _create_analysis_agent(self) -> Agent:
        """Create the Stage 1 analysis agent."""
        
        # Load analysis prompt using dynamic context selection
        loader = get_prompt_loader()
        system_prompt = await loader.load_prompt_with_dynamic_context_async(
            "refinement/sketch_analysis_system.txt",
            description="analysis agent for sketch improvement",
            refinement_type="analysis",
            slime_exemplar=self.exemplars.get('slime', '# Slime exemplar not loaded'),
            boids_exemplar=self.exemplars.get('boids', '# Pheromone exemplar not loaded'),
            particle_life_exemplar=self.exemplars.get('particle_life', '# Particle life exemplar not loaded'),
            taichi_crashes=self.taichi_crashes
        )
        
        
        agent = Agent(
            self.model,
            output_type=AnalysisResponse,
            output_retries=2,
            system_prompt=system_prompt
        )
        
        return agent
    
    async def _create_implementation_agent(self) -> Agent:
        """Create the Stage 2 implementation agent."""
        
        # Load implementation prompt using dynamic context selection
        loader = get_prompt_loader()
        system_prompt = await loader.load_prompt_with_dynamic_context_async(
            "refinement/sketch_implementation_system.txt",
            description="implementation agent for architectural refinement",
            refinement_type="implementation"
        )
        
        
        agent = Agent(
            self.model,
            output_type=RefinementResponse,
            output_retries=2,
            system_prompt=system_prompt,
            model_settings={"max_tokens": REFINEMENT_MAX_TOKENS},
        )

        return agent

    async def _create_single_stage_refinement_agent(self) -> Agent:
        """Create the single-stage refinement agent for quick fixes and repairs."""
        
        # Load single-stage refinement prompt using dynamic context selection
        loader = get_prompt_loader()
        system_prompt = await loader.load_prompt_with_dynamic_context_async(
            "refinement/single_state_refinement_system.txt",
            description="single stage refinement for error correction and features",
            refinement_type="error_correction"
        )
        
        
        agent = Agent(
            self.model,
            output_type=SingleStageRefinementResponse,
            output_retries=2,
            system_prompt=system_prompt,
            model_settings={"max_tokens": REFINEMENT_MAX_TOKENS},
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
        
        # Lazily create analysis agent if needed
        if self.analysis_agent is None:
            self.analysis_agent = await self._create_analysis_agent()
        
        collector = get_collector()
        
        with collector.trace_node("analyze_sketch", "analysis", description=description) as node:
            
            # Load the analysis prompt template
            from ..prompts.prompt_loader import get_prompt_loader
            loader = get_prompt_loader()
            prompt = loader.load_prompt(
                "refinement/analysis_user.txt",
                description=description,
                sketch_code=sketch_code
            )
            
            try:
                with collector.trace_node("analysis_llm_call", "llm_call", model=self.model_name) as llm_node:
                    
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
                        user_prompt=prompt,
                        system_prompt="Analysis agent system prompt",
                        full_prompt=prompt,  # Store the full prompt for complete trace viewing
                        parsed_response=analysis
                    )
                    if llm_node:
                        llm_node.llm_call = llm_data
                    
                    return analysis
                    
            except Exception as e:
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
        
        # Lazily create implementation agent if needed
        if self.implementation_agent is None:
            self.implementation_agent = await self._create_implementation_agent()
        
        collector = get_collector()
        
        with collector.trace_node("implement_refinement", "implementation", description=description) as node:
            
            # Load BASE_CONTEXT for library documentation
            from ..context.library_docs import BASE_CONTEXT
            
            # Load the implementation prompt template
            from ..prompts.prompt_loader import get_prompt_loader
            loader = get_prompt_loader()
            prompt = loader.load_prompt(
                "refinement/implementation_user.txt",
                description=description,
                plan=plan,
                sketch_code=sketch_code,
                base_context=BASE_CONTEXT
            )
            
            try:
                with collector.trace_node("implementation_llm_call", "llm_call", model=self.model_name) as llm_node:
                    
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
                        user_prompt=prompt,
                        system_prompt="Implementation agent system prompt",
                        full_prompt=prompt,  # Store the full prompt for complete trace viewing
                        parsed_response={'changes_summary': result.output.changes_summary}
                    )
                    if llm_node:
                        llm_node.llm_call = llm_data
                    
                    return refinement
                    
            except Exception as e:
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
        
        collector = get_collector()
        
        # Wrap the entire two-stage process in a parent trace node
        with collector.trace_node("two_stage_refinement", "refinement", 
                                  description=description,
                                  refinement_type="architectural") as parent_node:
            
            # Stage 1: Analyze and plan
            analysis = await self.analyze_sketch(sketch_code, description)
            
            if not analysis['success']:
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
        
        # Lazily create refinement agent if needed
        if self.refinement_agent is None:
            self.refinement_agent = await self._create_single_stage_refinement_agent()
        
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
            
            # Build error and signature sections if needed
            error_section = ""
            if error_info:
                error_section = f"""

ERROR INFORMATION:
The sketch crashed with this error:
{error_info}

Fix this error as part of the refinement.
"""
                
            signature_section = ""
            if error_info and has_signature_issue:
                signature_section = f"""

DETECTED: Function signature mismatch error!
{self._build_signature_mismatch_context()}
"""
            
            # Load the single-stage refinement prompt template
            from ..prompts.prompt_loader import get_prompt_loader
            loader = get_prompt_loader()
            prompt = loader.load_prompt(
                "refinement/single_stage_user.txt",
                conversation_context=conversation_context,
                refinement_request=refinement_request,
                sketch_code=sketch_code,
                error_section=error_section,
                signature_section=signature_section
            )
            
            try:
                with collector.trace_node("refinement_llm_call", "llm_call",
                                        model=self.model_name) as llm_node:
                    
                    result = await self.refinement_agent.run(prompt)
                    
                    refined_code = result.output.refined_code
                    changes_made = result.output.changes_made
                    warnings = result.output.warnings
                    
                    # Sanitize the code
                    refined_code = self._sanitize_refined_code(refined_code)

                    # Re-ensure OSC render-loop wiring. A full-sketch rewrite
                    # (especially error-repair) often keeps the OSC sender-block
                    # definitions but drops the per-frame calls, notably
                    # `_draw_tracked_rings()`, which renders the tracked-particle
                    # highlight. ensure_osc_senders is idempotent and re-adds any
                    # missing calls when the sender block is present.
                    from ..sc.emitter import ensure_osc_senders
                    refined_code = ensure_osc_senders(refined_code, "")

                    # Validate the code
                    validation_issues = self._validate_refined_code(refined_code)
                    if validation_issues:
                        warnings = (warnings or "") + f"\nValidation: {validation_issues}"
                    
                    # Create LLM call data
                    llm_data = LLMCallData(
                        model=self.model_name,
                        provider=self.provider,
                        user_prompt=prompt,
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
        
        # Load BASE_CONTEXT for library documentation
        from ..context.library_docs import BASE_CONTEXT
        
        # Load the repair request prompt template
        from ..prompts.prompt_loader import get_prompt_loader
        loader = get_prompt_loader()
        
        # Build additional context section if needed
        additional_context_section = ""
        if additional_context:
            additional_context_section = f"\n\nAdditional context: {additional_context}\n\n{BASE_CONTEXT}"
        else:
            additional_context_section = f"\n\n{BASE_CONTEXT}"
        
        repair_request = loader.load_prompt(
            "refinement/repair_request.txt",
            error_logs=error_logs,
            additional_context_section=additional_context_section
        )
        
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
        # Load the signature mismatch guide from external file
        from ..prompts.prompt_loader import get_prompt_loader
        loader = get_prompt_loader()
        return loader.load_prompt("refinement/signature_mismatch_guide.txt")
    
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