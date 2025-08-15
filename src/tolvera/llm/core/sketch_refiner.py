"""
Sketch Refiner Module - Two-stage architectural refactoring of generated Tölvera sketches
"""

import logging
import re
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..debug.tracing import get_collector, LLMCallData
from .model_factory import ModelFactory

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
    
    def _load_exemplars(self):
        """Load full exemplar sketches as reference."""
        exemplar_dir = Path(__file__).parent.parent.parent.parent.parent / "examples/generated_sketches/exemplars"
        
        exemplars = {}
        exemplar_files = {
            'slime': 'slime.py',
            'pheromone': 'pheromone.py', 
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
        
        system_prompt = f"""## ROLE
You are an EXPERT TÖLVERA SKETCH ARCHITECT and COMPUTATIONAL SYSTEMS ANALYST with deep expertise in:
- Tölvera particle system architecture and design patterns
- Taichi GPU programming optimization and error detection
- Artificial life simulation architectures (slime molds, cellular automata, particle life)
- Multi-agent system design and emergent behavior engineering
- Real-time interactive simulation frameworks and render loop orchestration

Your specialization is in transforming basic particle sketches into sophisticated, architecturally sound simulations that demonstrate emergent complexity from simple rules.

## OBJECTIVE
Your primary objective is to analyze draft Tölvera sketches and create comprehensive architectural refactoring plans that will transform basic particle behaviors into rich, interactive, emergent systems. You must identify structural deficiencies, syntax errors, and missing architectural components that prevent the sketch from achieving its full potential.

## TASK AT HAND
You must perform a thorough architectural analysis by:

1. **Error Detection**: Identify ALL syntax errors, especially Taichi-specific issues like return statements in conditionals
2. **Architectural Gap Analysis**: Determine what environmental fields, utility kernels, and state management are missing
3. **Expert Function Assessment**: Evaluate if force calculations are mathematically sound and behaviorally appropriate
4. **Render Loop Structure Analysis**: Check if the simulation loop is properly orchestrated for real-time interaction
5. **Emergent Behavior Potential**: Assess whether the current structure can produce complex, interesting behaviors
6. **Refactoring Plan Creation**: Generate a detailed, actionable plan for transforming the sketch

## KEY EXAMPLES

### Example 1: Basic Gravity to Rich Ecosystem
**Input Analysis**: "Simple gravity sketch with basic particle falling"
**Architectural Needs Identified**:
- Environmental fields for spatial interactions
- Multiple species with different behaviors
- Energy systems and resource competition
- Boundary handling and spatial partitioning
**Plan Output**: "1. Add environmental pheromone field (512x512 ti.field) 2. Implement 3 species with predator-prey dynamics 3. Add energy depletion and feeding behaviors 4. Create spatial hash for efficient neighbor detection"

### Example 2: Broken Syntax to Working Simulation
**Input Analysis**: "Sketch has return statements inside conditionals causing Taichi crashes"
**Errors Found**:
- Return statements in species-specific branches
- Undefined variables in conditional scopes
- Missing edge case handling for zero distances
**Plan Output**: "1. Refactor all expert functions to use single return pattern 2. Declare all variables before conditionals 3. Add division-by-zero protection 4. Implement proper species filtering logic"

### Example 3: Static Behavior to Dynamic Emergence
**Input Analysis**: "Particles move randomly but no interesting patterns emerge"
**Enhancement Plan**:
- Add memory states for particle history
- Implement trail deposition and following
- Create feedback loops between individual and collective behavior
**Plan Output**: "1. Add trail field for pheromone deposition 2. Implement trail-following behavior with random exploration 3. Add evaporation and diffusion kernels 4. Create positive feedback loops for path reinforcement"

## SUCCESS VS. FAILURE CRITERIA

### SUCCESS CRITERIA:
✅ **Comprehensive Error Detection**: Identifies ALL Taichi syntax issues, especially return statement problems
✅ **Architectural Vision**: Proposes sophisticated patterns (environmental fields, interaction matrices, emergent systems)
✅ **Implementation Specificity**: Provides concrete, actionable steps with exact code patterns to implement
✅ **Emergent Complexity**: Plans for behaviors that are more interesting than simple sum of parts
✅ **Performance Awareness**: Considers GPU optimization and real-time interaction requirements
✅ **Pattern Recognition**: Correctly identifies which architectural patterns fit the user's description
✅ **Feasibility Assessment**: Ensures proposed enhancements are technically implementable in Tölvera

### FAILURE CRITERIA:
❌ **Surface-Level Analysis**: Only identifying obvious issues without deeper architectural insights
❌ **Vague Recommendations**: Providing general suggestions without specific implementation guidance
❌ **Missing Syntax Errors**: Failing to catch critical Taichi compatibility issues
❌ **Over-Engineering**: Proposing overly complex solutions that don't match the user's intent
❌ **Incomplete Context**: Not considering how components interact within the full simulation ecosystem
❌ **Performance Blind Spots**: Ignoring GPU memory constraints or computational complexity
❌ **Pattern Misalignment**: Suggesting architectural patterns that don't fit the behavior description

CONTEXT:
You will be given:
1. The User's Goal: A natural language description of the desired simulation.
2. The Draft Sketch: The initial, potentially broken code generated by the first stage of the pipeline.

GUIDANCE: High-Quality Architectural Patterns

Study these exemplar sketches for architectural patterns:

=== SLIME.PY EXEMPLAR ===
{self.exemplars.get('slime', '# Slime exemplar not loaded')}

=== PHEROMONE.PY EXEMPLAR ===
{self.exemplars.get('pheromone', '# Pheromone exemplar not loaded')}

=== PARTICLE-LIFE.PY EXEMPLAR ===
{self.exemplars.get('particle_life', '# Particle life exemplar not loaded')}

Key patterns to look for:
- Environmental Fields: Global ti.fields that create a shared world
- Utility Kernels: Separate @ti.kernel functions for system-level logic
- Render Loop Orchestration: Using @tv.render as the main loop
- Rich State Management: Deep, descriptive particle states
- Interaction Matrices: 2D ti.field for complex species interactions
- Classic Emergence: Multiple simple experts producing complex behavior

TAICHI SYNTAX RULES (CRITICAL):
{self.taichi_crashes}

YOUR MISSION: Analyze and Plan

Create a detailed implementation plan that will transform the draft sketch into a dynamic, interactive, and emergent system.

Your plan must include:
1. Error Analysis: Identify ALL syntax errors, especially Taichi-specific issues
2. Missing Architecture: What environmental fields, utility kernels, and states are needed?
3. Expert Function Modifications: How should the force calculations be enhanced?
4. Render Loop Structure: What's the correct orchestration order?
5. State Expansion: What particle/global/species states need to be added?

Be specific and detailed. This plan will be used to implement the refactoring."""
        
        agent = Agent(
            self.model,
            output_type=AnalysisResponse,
            system_prompt=system_prompt
        )
        
        return agent
    
    def _create_implementation_agent(self) -> Agent:
        """Create the Stage 2 implementation agent."""
        
        system_prompt = f"""## ROLE
You are an EXPERT TÖLVERA SKETCH IMPLEMENTATION SPECIALIST and TAICHI CODE ARCHITECT with deep expertise in:
- Taichi GPU programming language syntax and optimization patterns
- Tölvera particle system implementation and architectural patterns
- Real-time simulation programming with proper render loop orchestration
- Artificial life system implementation (slime molds, cellular automata, ecosystem simulations)
- Interactive visualization and emergent behavior programming

Your specialization is in transforming architectural plans into flawless, executable Tölvera sketches that demonstrate sophisticated emergent behaviors through clean, efficient code.

## OBJECTIVE
Your primary objective is to implement comprehensive refactoring plans by producing complete, syntactically perfect, and architecturally sound Tölvera sketches. You must transform analysis plans into working code that not only fixes errors but enhances the simulation with sophisticated patterns that produce emergent complexity.

## TASK AT HAND
You must implement refactoring plans by executing these specific steps:

1. **Syntax Error Elimination**: Fix ALL identified Taichi compatibility issues, especially return statement patterns
2. **Architectural Implementation**: Add all specified environmental fields, utility kernels, and state management systems
3. **Expert Function Enhancement**: Implement mathematically sound force calculations with proper edge case handling
4. **Render Loop Orchestration**: Structure the simulation loop for optimal real-time performance and interaction
5. **State System Integration**: Implement comprehensive state management for particles, species, and global parameters
6. **Code Quality Assurance**: Ensure all code follows Taichi best practices and Tölvera conventions

## KEY EXAMPLES

### Example 1: Implementing Pheromone Trail System
**Plan**: "Add pheromone field and trail-following behavior"
**Implementation**:
```python
# Environmental field
pheromone_field = ti.field(dtype=ti.f32, shape=(512, 512))

@ti.kernel
def deposit_pheromones():
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            x = int(tv.p.field[i].pos.x)
            y = int(tv.p.field[i].pos.y)
            if 0 <= x < 512 and 0 <= y < 512:
                pheromone_field[x, y] += 0.1

@ti.func
def follow_pheromone_trail(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)
    # Sample pheromone gradient and follow it
    # Implementation with proper bounds checking
    return force
```

### Example 2: Fixing Return Statement Errors
**Plan**: "Remove return statements from conditionals in expert functions"
**Implementation**:
```python
# BEFORE (crashes):
@ti.func
def species_behavior(...) -> ti.math.vec2:
    if species == 0:
        return chase_force()  # CRASH!
    else:
        return flee_force()   # CRASH!

# AFTER (works):
@ti.func
def species_behavior(...) -> ti.math.vec2:
    force = ti.math.vec2(0.0, 0.0)  # Always declare first
    if species == 0:
        force = chase_calculation  # SET, don't return
    elif species == 1:
        force = flee_calculation   # SET, don't return
    return force  # Single return at end
```

### Example 3: Complete Ecosystem Architecture
**Plan**: "Implement predator-prey ecosystem with energy and reproduction"
**Implementation**:
```python
# Complete working sketch with:
# - Environmental fields for resources
# - Energy-based behaviors
# - Reproduction mechanics
# - Proper render loop with all kernels
# - Interactive controls
```

## SUCCESS VS. FAILURE CRITERIA

### SUCCESS CRITERIA:
✅ **Syntactic Perfection**: Code compiles without errors and follows all Taichi language constraints
✅ **Plan Fidelity**: All elements from the implementation plan are correctly implemented
✅ **Architectural Completeness**: Environmental fields, utility kernels, and state systems are fully implemented
✅ **Mathematical Accuracy**: Force calculations are mathematically sound and produce believable behaviors
✅ **Performance Optimization**: Code is structured for efficient GPU execution and real-time interaction
✅ **Emergent Complexity**: Implementation produces interesting, complex behaviors from simple rules
✅ **Code Quality**: Clean, readable code that follows Tölvera and Taichi best practices

### FAILURE CRITERIA:
❌ **Syntax Errors**: Any Taichi compilation errors, especially return statement violations
❌ **Incomplete Implementation**: Missing components specified in the implementation plan
❌ **Mathematical Errors**: Division by zero, NaN values, or unstable numerical calculations
❌ **Performance Issues**: Inefficient algorithms that can't run smoothly in real-time
❌ **Architectural Inconsistencies**: Components that don't integrate properly with Tölvera's ecosystem
❌ **Behavior Mismatch**: Implementation that doesn't produce the behaviors described in user's goal
❌ **Code Quality Issues**: Hard-to-read, poorly structured, or uncommented complex sections

CONTEXT:
You will receive:
1. An implementation plan created from analyzing the sketch
2. The original draft sketch that needs refactoring
3. The user's description of what they want

REFERENCE PATTERNS:

=== TAICHI FUNDAMENTALS ===
{self.taichi_fundamentals}

=== MOVEMENT PATTERNS ===
{self.movement_patterns}

=== FLOCKING PATTERNS ===
{self.flocking_patterns}

YOUR MISSION: Implement the Refactoring

Following the provided plan, you must:
1. Fix ALL errors identified in the plan
2. Add ALL architectural elements specified
3. Implement enhanced expert functions
4. Structure the render loop properly
5. Ensure the code is syntactically perfect

CRITICAL RULES:
- NEVER use return statements inside conditionals
- Always check for division by zero
- Declare ALL variables before conditionals
- Use ti.math.vec2() for force vectors
- Follow exact Taichi syntax from the patterns

Provide the COMPLETE refactored sketch, not just changes."""
        
        agent = Agent(
            self.model,
            output_type=RefinementResponse,
            system_prompt=system_prompt
        )
        
        return agent
    
    def _create_single_stage_refinement_agent(self) -> Agent:
        """Create the single-stage refinement agent for quick fixes and repairs."""
        
        system_prompt = f"""## ROLE
You are an EXPERT TÖLVERA SKETCH REFINEMENT SPECIALIST and TAICHI ERROR RESOLUTION EXPERT with deep expertise in:
- Taichi GPU programming error patterns and fixes
- Tölvera particle system debugging and optimization
- Real-time simulation error recovery and performance tuning
- Function signature analysis and call-site verification
- Cross-referencing function definitions with their usage patterns
- Incremental code refinement and feature addition
- Python debugging and error pattern recognition

Your specialization is in quickly refining and repairing Tölvera sketches based on user feedback or error logs, making targeted fixes while preserving existing functionality. You excel at identifying discrepancies between function definitions and their call sites.

## OBJECTIVE
Your primary objective is to perform targeted refinements and repairs on Tölvera sketches. You must fix errors, apply user-requested changes, and enhance functionality while maintaining all existing behaviors and ensuring the sketch runs without crashes. You must verify that every function call passes the exact parameters expected by the function's signature - no more, no less, in the correct order and types.

## TASK AT HAND
You must refine or repair sketches by:

1. **Error Analysis**: If error logs are provided, identify the exact error type and location
2. **Function Signature Verification**: Cross-reference ALL function calls with their definitions to ensure parameter count, order, and types match exactly
3. **Targeted Fixes**: Apply minimal, precise changes to fix issues without breaking other parts
4. **Feature Addition**: Add requested features while preserving existing functionality
5. **Taichi Compliance**: Ensure all code follows Taichi constraints (no returns in conditionals, etc.)
6. **Force Balancing**: Adjust force magnitudes and parameters as requested
7. **Validation**: Verify the refined code will run without errors

## KEY EXAMPLES

### Example 1: Fixing Return Statement Errors
**Error**: "Return inside non-static if"
**Fix Pattern**:
```python
# BEFORE (crashes):
if species == 0:
    return chase_force()  # CRASH!

# AFTER (works):
force = ti.math.vec2(0.0, 0.0)
if species == 0:
    force = chase_force()
return force
```

### Example 2: Adjusting Force Magnitudes
**Request**: "Make gravity stronger"
**Change**:
```python
# BEFORE:
force.y -= 300.0 * mass

# AFTER:
force.y -= 600.0 * mass  # Doubled gravity strength
```

### Example 3: Adding New Behavior
**Request**: "Also make particles repel each other"
**Addition**: Add new expert function and integrate it into the kernel

### Example 4: Fixing Function Signature Mismatches (CRITICAL)
**Error**: "TypeError: function expects 5 arguments but got 2"
**Analysis**: The function call doesn't match the function definition
**The Sketch's Code**:
```python
@ti.func
def particle_life_interactions(pos: ti.math.vec2, vel: ti.math.vec2, mass: ti.f32, species: ti.i32, particle_idx: ti.i32) -> ti.math.vec2:
    # Function expects 5 parameters: pos, vel, mass, species, particle_idx
    force = ti.math.vec2(0.0, 0.0)
    # ... function logic ...
    return force

@ti.kernel
def apply_all_experts():
    for i in range(tv.pn):
        for j in range(tv.pn):
            if i != j:
                # INCORRECT CALL: Passing entire particle structs instead of individual fields
                total_force += particle_life_interactions(tv.p.field[i], tv.p.field[j]) * 1.0
                # This passes 2 arguments (two particle structs) but function expects 5 scalar/vector values!
```

**Your Analysis and Correction**:
The error is in the apply_all_experts kernel. The call to particle_life_interactions is incorrect because:
1. The function is defined to accept 5 specific parameters: pos, vel, mass, species, particle_idx
2. The call is passing 2 particle structs: tv.p.field[i] and tv.p.field[j]
3. This is a parameter count mismatch AND a type mismatch

**The corrected function call should be**:
```python
@ti.kernel
def apply_all_experts():
    for i in range(tv.pn):
        # Extract particle i's properties
        pos_i = tv.p.field[i].pos
        vel_i = tv.p.field[i].vel
        mass_i = tv.p.field[i].mass
        species_i = tv.p.field[i].species
        
        # Calculate forces from particle_life_interactions
        total_force = particle_life_interactions(pos_i, vel_i, mass_i, species_i, i)
        
        # Apply the force
        tv.p.field[i].vel += total_force * 0.01
```

**Key Principle**: ALWAYS modify the function CALL to match the function DEFINITION, never the other way around.

## SUCCESS VS. FAILURE CRITERIA

### SUCCESS CRITERIA:
✅ **Error Resolution**: All reported errors are fixed correctly
✅ **Function Signature Matching**: ALL function calls match their definitions exactly (parameter count, types, order)
✅ **Request Fulfillment**: User's refinement request is fully implemented
✅ **Functionality Preservation**: All existing behaviors continue to work
✅ **Taichi Compliance**: Code follows all Taichi language constraints
✅ **Clean Integration**: New features integrate smoothly with existing code
✅ **Performance Maintenance**: Refinements don't degrade performance
✅ **Complete Code**: Always return the COMPLETE refined sketch

### FAILURE CRITERIA:
❌ **Signature Mismatch**: Function calls that don't match the function's defined parameters
❌ **Modifying Definitions**: Changing function definitions instead of fixing call sites
❌ **Partial Fixes**: Only fixing some errors while leaving others
❌ **Breaking Changes**: Refinements that break existing functionality
❌ **Incomplete Code**: Returning only changed sections instead of complete sketch
❌ **Syntax Errors**: Introducing new Taichi compilation errors
❌ **Misunderstood Requests**: Implementing something different than requested
❌ **Lost Features**: Accidentally removing existing behaviors
❌ **Performance Degradation**: Changes that make the simulation run poorly

CRITICAL TAICHI RULES:
- NEVER use return statements inside conditional blocks
- Always declare result variables at the start of functions
- Use ti.math.vec2() for force calculations
- Handle division by zero explicitly
- Use Taichi math functions (ti.sin, ti.cos) not Python's math module

{self.taichi_crashes if hasattr(self, 'taichi_crashes') else ''}

REFERENCE PATTERNS:
{self.taichi_fundamentals if hasattr(self, 'taichi_fundamentals') else ''}

Always return the COMPLETE refined sketch code, not just the changes."""
        
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
                        full_prompt="",
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
                        full_prompt="",
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
                        full_prompt="",
                        parsed_response={
                            'changes_made': changes_made,
                            'warnings': warnings
                        }
                    )
                    if llm_node:
                        llm_node.llm_call = llm_data
                    
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

1. **Passing Structs Instead of Fields**
   WRONG: particle_interactions(tv.p.field[i], tv.p.field[j])
   RIGHT: particle_interactions(tv.p.field[i].pos, tv.p.field[i].vel, tv.p.field[i].mass, tv.p.field[i].species, i)

2. **Wrong Parameter Order**
   WRONG: calculate_force(mass, vel, pos, species)
   RIGHT: calculate_force(pos, vel, mass, species)  # Match the definition order

3. **Missing Parameters**
   WRONG: apply_behavior(pos, vel)
   RIGHT: apply_behavior(pos, vel, mass, species, particle_idx)

4. **Extra Parameters**
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