"""
Pure LLM-based expert synthesis for the PoE behavior system.

This module provides functionality to generate behavior experts from natural language descriptions.
"""

import logging
import re
from typing import Optional, List, Dict, Any, Tuple
import asyncio
import time

from .poe_core import SimpleProgrammaticExpert
from .poe_ollama import OllamaClient
from .prompt_loader import load_prompt
from .poe_logger import get_logger
from .taichi_error_detector import TaichiErrorDetector
from .taichi_error_corrector import TaichiErrorCorrector
from .kernel_accumulator import KernelAccumulator
from .behavior_decomposer import BehaviorDecomposer, SubBehavior
from .boundary_manager import BoundaryManager, BoundaryMode
from .state_synthesizer import StateSynthesizer
from .dynamic_state_manager import DynamicStateManager

logger = logging.getLogger(__name__)
csv_logger = get_logger()


class PoEExpertSynthesizer:

    def __init__(self, model_name: Optional[str] = None, auto_correct: bool = True, 
                 accumulator_path: str = "generated_kernels/kernels_repository.py",
                 enable_decomposition: bool = True, tolvera_instance=None):
        self.client = OllamaClient(model_name)
        self.error_detector = TaichiErrorDetector()
        self.error_corrector = TaichiErrorCorrector(model_name)
        self.auto_correct = auto_correct
        self.kernel_accumulator = KernelAccumulator(accumulator_path)
        self.decomposer = BehaviorDecomposer(model_name) if enable_decomposition else None
        self.enable_decomposition = enable_decomposition
        self.boundary_manager = BoundaryManager()
        self.state_synthesizer = StateSynthesizer(model_name)
        self.state_manager = DynamicStateManager(tolvera_instance) if tolvera_instance else None
        
        # Determine if we're using a small model
        self.is_small_model = self._is_small_model(model_name)

    def _is_small_model(self, model_name: Optional[str]) -> bool:
        """Determine if the model is a small model that needs simplified prompts."""
        if not model_name:
            return False
        small_model_patterns = ['3b', '4b', '7b', '1b', '2b', 'small', 'mini', 'tiny']
        return any(pattern in model_name.lower() for pattern in small_model_patterns)
    
    def _fix_parameter_order(self, code: str) -> str:
        """Fix parameter order in expert functions if they're incorrect."""
        import re
        
        # Pattern to match expert function definition
        func_pattern = r'(@ti\.func\s*\n\s*def\s+expert_\w+\s*\()([^)]+)(\)\s*->\s*ti\.math\.vec2\s*:)'
        
        match = re.search(func_pattern, code, re.MULTILINE | re.DOTALL)
        if not match:
            return code
        
        decorator_and_def = match.group(1)
        params = match.group(2)
        return_type = match.group(3)
        
        # Check if parameters are in wrong order (species first)
        if 'species:' in params and params.strip().startswith('species:'):
            logger.warning("Detected incorrect parameter order (species first), fixing...")
            
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
                logger.info("Successfully fixed parameter order")
                return fixed_code
        
        return code

    def analyze_boundary_requirements(self, description: str) -> BoundaryMode:
        mode, confidence = self.boundary_manager.analyze_boundary_requirements(description)
        logger.info(f"Boundary analysis for '{description}': {mode.value} (confidence: {confidence})")
        return mode
    
    def extract_code(self, response: str) -> str:
        # Try to find code between triple backticks
        code_match = re.search(
            r'```(?:python)?\n(.*?)```',
            response,
            re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try to find @ti.func definition directly
        func_match = re.search(
            r'(@ti\.func.*?)(?=\n@|\n\n|\Z)',
            response,
            re.DOTALL)
        if func_match:
            return func_match.group(1).strip()

        # Return cleaned response
        return response.strip()

    def validate_expert_code(self, code: str) -> Tuple[bool, List[str]]:
        errors = []

        # Check for required structure
        if "@ti.func" not in code:
            errors.append("Missing @ti.func decorator")

        # Check for return statement
        if "return" not in code:
            errors.append("Missing return statement for force vector")
        else:
            # Check if return statement returns a vector (more flexible)
            # Allow any variable name that ends with 'force' or direct vec2 construction
            if not re.search(r'return\s+\w*force\b', code) and not re.search(r'return\s+ti\.math\.vec2\s*\(', code):
                # Still might be valid - check if it returns any variable at all
                if not re.search(r'return\s+\w+', code):
                    errors.append("Return statement must return a force vector")
        
        # Check if function accesses states but doesn't compute force
        if re.search(r'tv\.s\.llm_\w+\.field', code):
            # Function accesses states
            if not re.search(r'force\s*=', code):
                errors.append("Function accesses states but doesn't compute any force")
            else:
                # Check if force is computed beyond just initialization
                # Look for force assignments other than the initial zero
                force_assignments = re.findall(r'force\s*=\s*(.+)', code)
                if len(force_assignments) <= 1:
                    # Only one assignment (likely just initialization)
                    if force_assignments and 'ti.math.vec2(0.0, 0.0)' in force_assignments[0]:
                        errors.append("Force is only initialized to zero - must compute actual force using states")
                # If there are multiple force assignments, assume it's being computed properly

        # Check for unsafe operations
        unsafe_patterns = [
            (r'exec\s*\(', "exec() is not allowed"),
            (r'eval\s*\(', "eval() is not allowed"),
            (r'__import__', "__import__ is not allowed"),
            (r'open\s*\(', "file operations not allowed"),
            (r'subprocess', "subprocess operations not allowed")
        ]

        for pattern, message in unsafe_patterns:
            if re.search(pattern, code):
                errors.append(message)

        # Check for Taichi-specific patterns
        if re.search(r'(?<!ti\.)math\.(sqrt|sin|cos|tan)', code):
            errors.append("Use ti.sqrt, ti.sin, ti.cos instead of Python math")

        return len(errors) == 0, errors
    
    def _validate_state_usage(self, code: str, state_context: Dict[str, Any]) -> List[str]:
        """
        Validate that the code only uses states that exist in the state context.
        
        Returns a list of invalid state references found.
        """
        # Patterns to find state accesses
        patterns = [
            r'tv\.s\.llm_global\.field\[\d+\]\.(\w+)',     # Global states
            r'tv\.s\.llm_particle\.field\[[\w_]+\]\.(\w+)', # Particle states  
            r'tv\.s\.llm_species\.field\[[\w_]+\]\.(\w+)',  # Species states
            r'tv\.s\.llm_pixel\.field\[[\w,\s]+\]\.(\w+)'   # Pixel states
        ]
        
        invalid_references = []
        
        # Extract available states from context
        available_states = {
            'global': set(),
            'particle': set(), 
            'species': set(),
            'pixel': set()
        }
        
        if state_context and 'available_states' in state_context:
            states_dict = state_context['available_states']
            for state_name, state_info in states_dict.items():
                if 'llm_global' in state_info.get('name', ''):
                    for prop in state_info.get('properties', {}):
                        available_states['global'].add(prop)
                elif 'llm_particle' in state_info.get('name', ''):
                    for prop in state_info.get('properties', {}):
                        available_states['particle'].add(prop)
                elif 'llm_species' in state_info.get('name', ''):
                    for prop in state_info.get('properties', {}):
                        available_states['species'].add(prop)
                elif 'llm_pixel' in state_info.get('name', ''):
                    for prop in state_info.get('properties', {}):
                        available_states['pixel'].add(prop)
        
        # Check each pattern
        for i, pattern in enumerate(patterns):
            state_type = ['global', 'particle', 'species', 'pixel'][i]
            matches = re.findall(pattern, code)
            
            for state_property in matches:
                # Check if this is a valid state property
                if state_property not in available_states[state_type]:
                    # Check if the entire state type is missing
                    if not available_states[state_type]:
                        invalid_references.append(f"tv.s.llm_{state_type} (no {state_type} states exist)")
                    else:
                        invalid_references.append(f"tv.s.llm_{state_type}.field[...].{state_property}")
        
        return list(set(invalid_references))  # Remove duplicates
    
    async def _fix_invalid_state_references(self, code: str, invalid_refs: List[str], state_context: Dict[str, Any]) -> str:
        """
        Use LLM to fix code that references invalid states.
        """
        # Build a clear list of available states
        available_states_text = self._format_available_states_for_fix(state_context)
        
        fix_prompt = f"""The following Taichi function references states that don't exist:

{code}

INVALID STATE REFERENCES FOUND:
{chr(10).join(f"- {ref}" for ref in invalid_refs)}

AVAILABLE STATES YOU CAN USE:
{available_states_text}

Please fix this code to:
1. ONLY use the states listed above
2. Remove or replace any references to non-existent states
3. Maintain the same behavioral intent but work with available states
4. If a required state doesn't exist, find a creative workaround

Return ONLY the corrected @ti.func code, no explanations."""

        messages = [
            {'role': 'system', 'content': 'You are a Taichi expert fixing invalid state references.'},
            {'role': 'user', 'content': fix_prompt}
        ]
        
        response = await self.client.chat(messages, temperature=0.1)
        return self.extract_code(response)
    
    def _format_available_states_for_fix(self, state_context: Dict[str, Any]) -> str:
        """Format available states in a clear way for the fixing prompt."""
        if not state_context or 'available_states' not in state_context:
            return "No custom states available - use only particle properties: pos, vel, mass, species"
        
        lines = []
        states_dict = state_context['available_states']
        
        for state_name, state_info in states_dict.items():
            if state_info.get('properties'):
                lines.append(f"\n{state_info['category'].upper()} STATES ({state_info['name']}):")
                for prop, details in state_info['properties'].items():
                    access_pattern = details.get('access_pattern', '')
                    lines.append(f"  - {prop}: {access_pattern}")
        
        if not lines:
            return "No custom states available - use only particle properties: pos, vel, mass, species"
        
        return "\n".join(lines)

    async def synthesize_expert_with_states(self, description: str) -> Dict[str, Any]:
        """
        Synthesize expert with automatic state analysis and creation.
        
        Args:
            description: Natural language behavior description
            
        Returns:
            Expert synthesis result with state information
        """
        logger.info(f"Synthesizing expert with state analysis for: '{description}'")
        
        # Step 1: Analyze state requirements
        state_spec = await self.state_synthesizer.analyze_state_requirements(description)
        logger.info(f"State analysis result for '{description}': {state_spec}")
        state_context = None
        
        # Step 2: Create states if needed and manager is available
        if state_spec and self.state_manager:
            any_states = any(state_spec.get(cat, {}) for cat in ['global_states', 'particle_states', 'species_states'])
            logger.info(f"State spec check: state_spec={bool(state_spec)}, manager={bool(self.state_manager)}, any_states={any_states}")
            if any_states:
                logger.info(f"Creating dynamic states based on analysis: {state_spec}")
                created_states = self.state_manager.create_states_from_spec(state_spec)
                logger.info(f"Created states: {created_states}")
                state_context = self.state_manager.get_synthesis_context()
                logger.info(f"State context: {state_context}")
            else:
                logger.warning(f"No states found in spec: {state_spec}")
        else:
            logger.warning(f"Skipping state creation: state_spec={bool(state_spec)}, manager={bool(self.state_manager)}")
        
        # Step 3: Synthesize expert with state context, routing to appropriate type
        classification = await self.classify_behavior(description)
        
        if classification == "INTERACTION":
            result = await self.synthesize_interaction_expert(description, state_context)
        elif classification == "STATE_TRANSITION":
            result = await self.synthesize_state_transition_expert(description, state_context)
        elif classification == "SENSOR":
            result = await self.synthesize_sensor_expert(description, state_context)
        elif classification == "DEPOSIT":
            result = await self.synthesize_deposit_expert(description, state_context)
        else:  # SINGLE or default
            result = await self.synthesize_expert(description, state_context)
        
        # Add state information to result
        if result['success']:
            result['state_spec'] = state_spec
            result['state_context'] = state_context
        
        return result

    async def synthesize_expert(self, description: str, state_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        logger.info(f"Synthesizing expert for: '{description}'")
        
        species_info = await self.extract_species_info(description)

        system_prompt = load_prompt("expert_synthesis_system")
        
        # Choose appropriate user prompt based on model size
        user_prompt_file = "expert_synthesis_user_small_model" if self.is_small_model else "expert_synthesis_user"
        
        # Add state context to user prompt if available
        if state_context:
            # Generate state access examples from the state_spec in context
            state_examples = ""
            if state_context.get('state_spec'):
                # Log available states for debugging
                available_states = state_context.get('available_states', {})
                if available_states:
                    state_names = []
                    for state_name, state_info in available_states.items():
                        for prop_name in state_info.get('properties', {}):
                            state_names.append(f"{state_name}.{prop_name}")
                    logger.info(f"Available states for synthesis: {', '.join(state_names)}")
                else:
                    logger.warning("No custom states available for synthesis - using only standard properties")
                
                # Use the same approach as integration kernel for consistency
                if self.is_small_model:
                    typed_context = self.state_synthesizer.generate_simplified_state_context(state_context.get('state_spec', {}))
                    state_examples = typed_context.get('state_summary', '')
                else:
                    # For full models, use the detailed access examples
                    state_examples = state_context.get('access_examples', '')
                    if not state_examples and self.state_manager:
                        state_examples = self.state_manager.generate_state_access_examples()
            else:
                logger.warning("No state_spec in state context - using only standard properties")
            
            # Format user prompt with state context
            user_prompt_template = load_prompt(user_prompt_file)
            user_prompt = user_prompt_template.format(
                description=description,
                state_context=state_examples if state_examples else "No custom states available"
            )
        else:
            logger.warning("No state context provided - using only standard properties")
            # Format with empty state context
            user_prompt = load_prompt(user_prompt_file).format(
                description=description,
                state_context="No custom states available")

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        logger.debug(f"LLM Messages: {messages}")

        try:
            response = await self.client.chat(messages, temperature=0.5, think=False)

            logger.info(f"Raw LLM Response:\n{response}")

            code = self.extract_code(response)

            logger.info(f"Extracted Code:\n{code}")
            
            # Fix parameter order if needed
            original_code = code
            code = self._fix_parameter_order(code)
            if code != original_code:
                logger.info("Fixed parameter order in generated code")
            
            # Validate state usage
            invalid_state_refs = self._validate_state_usage(code, state_context)
            if invalid_state_refs:
                logger.warning(f"Found invalid state references: {invalid_state_refs}")
                logger.info("Attempting to fix invalid state references...")
                code = await self._fix_invalid_state_references(code, invalid_state_refs, state_context)
                logger.info(f"Fixed code:\n{code}")
                
                # Re-validate after fix
                invalid_state_refs = self._validate_state_usage(code, state_context)
                if invalid_state_refs:
                    logger.error(f"Still has invalid state references after fix: {invalid_state_refs}")

            is_valid, errors = self.validate_expert_code(code)
            
            # Run error detector for additional checks
            detected_errors = self.error_detector.detect_errors(code)
            if detected_errors:
                error_summary = self.error_detector.get_error_summary(detected_errors)
                logger.warning(f"Detected {len(detected_errors)} potential issues in generated code")
                logger.warning(f"Error summary: {error_summary}")
                
                # Add detected errors to the response for logging
                detected_error_messages = [
                    f"Line {e['line']}: {e['severity']} - {e['message']}" 
                    for e in detected_errors if e['severity'] == 'error'
                ]
                errors.extend(detected_error_messages)
            
            # Attempt auto-correction if enabled and there are errors
            correction_result = None
            if self.auto_correct and (not is_valid or (detected_errors and any(e['severity'] == 'error' for e in detected_errors))):
                logger.info("Attempting automatic error correction...")
                correction_result = await self.error_corrector.correct_errors(code)
                
                if correction_result['success']:
                    logger.info(f"Successfully corrected code after {correction_result['attempts']} attempts")
                    code = correction_result['corrected_code']
                    # Re-validate the corrected code
                    is_valid, errors = self.validate_expert_code(code)
                    detected_errors = correction_result['final_errors']
                else:
                    logger.warning("Auto-correction did not resolve all errors")
            
            if not is_valid:
                logger.warning(f"Generated invalid code: {errors}")
                return {
                    "success": False,
                    "code": code,
                    "errors": errors,
                    "raw_response": response,
                    "detected_errors": detected_errors,
                    "correction_attempted": correction_result is not None,
                    "correction_result": correction_result}

            name_match = re.search(r'def\s+expert_(\w+)', code)
            if not name_match:
                # Try fallback for functions without expert_ prefix
                name_match = re.search(r'def\s+(\w+)', code)
                if not name_match:
                    return {
                        "success": False,
                        "code": code,
                        "errors": ["Could not extract function name"],
                        "raw_response": response}
                name = name_match.group(1)
            else:
                name = name_match.group(1)
                
            # Check if LLM forgot to replace NAME placeholder
            if name == "NAME":
                logger.warning("LLM did not replace NAME placeholder in function signature")
                # Ask LLM to provide a proper name
                name_prompt = f"""The following expert function uses 'NAME' as a placeholder:

{code}

Based on the behavior description: "{description}"

Please provide ONLY a suitable function name (lowercase with underscores, no 'expert_' prefix).
Examples: center_attraction, upward_drift, species_repulsion"""
                
                messages = [{'role': 'user', 'content': name_prompt}]
                try:
                    name_response = await self.client.chat(messages, temperature=0.1, max_tokens=20)
                    suggested_name = name_response.strip().lower().replace(' ', '_')
                    # Clean up the name
                    suggested_name = re.sub(r'[^a-z0-9_]', '', suggested_name)
                    if suggested_name:
                        name = suggested_name
                    else:
                        name = "behavior"  # Fallback
                except:
                    name = "behavior"  # Fallback
                
                # Fix the code to use the generated name
                code = code.replace("expert_NAME", f"expert_{name}")

            # Save successful kernel to accumulator
            kernel_uuid = ""
            try:
                additional_metadata = {
                    "description": description,
                    "corrected": correction_result is not None
                }
                kernel_uuid = self.kernel_accumulator.save_kernel(
                    code=code,
                    prompt=description,
                    model=self.client.model_name,
                    kernel_type="expert",
                    additional_metadata=additional_metadata
                )
                logger.info(f"Saved expert kernel with UUID: {kernel_uuid}")
            except Exception as e:
                logger.warning(f"Failed to save kernel to accumulator: {e}")
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "description": description,
                "metadata": {
                    "type": "force",
                    "returns": "ti.math.vec2",
                    "description": description
                },
                "errors": [],
                "raw_response": response,
                "detected_errors": detected_errors if detected_errors else [],
                "correction_attempted": correction_result is not None,
                "correction_result": correction_result,
                "kernel_uuid": kernel_uuid,
                "species_info": species_info}

        except Exception as e:
            logger.error(f"Expert synthesis failed: {e}")
            return {
                "success": False,
                "code": "",
                "errors": [
                    str(e)],
                "raw_response": ""}


    async def synthesize_integration_kernel(self, expert_info: list, boundary_mode: Optional[BoundaryMode] = None, state_context: Optional[Dict[str, Any]] = None) -> dict:
        """
        Template-based integration kernel synthesis using typed holes approach.
        This is now the primary method for kernel generation.
        """
        logger.info(f"Synthesizing integration kernel using template approach for {len(expert_info)} experts")
        
        # Step 1: Ask LLM for JSON configuration only
        config = await self._get_kernel_configuration(expert_info, state_context)
        if not config or not config.get('success'):
            logger.error("Failed to get kernel configuration")
            return {
                "success": False,
                "errors": ["Failed to generate kernel configuration"],
                "code": "",
                "config": config
            }
        
        # Step 2: Use SpeciesManager logic to generate expert calls
        # This preserves all the sophisticated species filtering
        single_experts = config['single_particle_experts']
        interaction_experts = config.get('interaction_experts', [])
        
        # Generate expert call strings using existing logic from SpeciesManager
        single_expert_calls = self._generate_single_expert_calls(single_experts)
        interaction_expert_calls = self._generate_interaction_expert_calls(interaction_experts)
        
        # Step 3: Generate state access code if needed
        state_access_code = ""
        if state_context:
            if self.is_small_model:
                typed_context = self.state_synthesizer.generate_simplified_state_context(state_context.get('state_spec', {}))
            else:
                typed_context = self.state_synthesizer.generate_typed_state_context(state_context.get('state_spec', {}))
            state_access_code = typed_context.get('declarations', '')
        
        # Step 4: Get boundary code
        if boundary_mode is None:
            boundary_mode = BoundaryMode.NONE
        boundary_code = self.boundary_manager.get_boundary_code(boundary_mode, use_new_pos=True)
        
        # Step 5: Select appropriate template based on expert types
        template_name = self._select_kernel_template(expert_info)
        
        # Step 6: Categorize experts if using multi-modal template
        if template_name == 'integration_kernel_multimodal.j2':
            categorized = self._categorize_experts_by_type(expert_info)
            
            # Generate calls for each type
            sensor_experts = self._prepare_sensor_experts(categorized['sensor'])
            state_transition_experts = self._prepare_state_transition_experts(categorized['state_transition'])
            deposit_experts = self._prepare_deposit_experts(categorized['deposit'])
            
            # Force experts are from the categorized results
            single_experts = categorized['force_single']
            interaction_experts = categorized['force_interaction']
            
            # Re-generate expert calls for force experts only
            single_expert_calls = self._generate_single_expert_calls(single_experts)
            interaction_expert_calls = self._generate_interaction_expert_calls(interaction_experts)
            
            template_data = {
                'single_particle_experts': single_experts,
                'single_expert_calls': single_expert_calls,
                'interaction_experts': interaction_experts,
                'interaction_expert_calls': interaction_expert_calls,
                'sensor_experts': sensor_experts,
                'state_transition_experts': state_transition_experts,
                'deposit_experts': deposit_experts,
                'state_access_code': state_access_code,
                'boundary_code': boundary_code
            }
        else:
            # Standard force-only template
            template_data = {
                'single_particle_experts': single_experts,
                'single_expert_calls': single_expert_calls,
                'interaction_experts': interaction_experts,
                'interaction_expert_calls': interaction_expert_calls,
                'state_access_code': state_access_code,
                'boundary_code': boundary_code
            }
        
        # Step 7: Load and render template
        import jinja2
        import os
        
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        template = env.get_template(template_name)
        
        kernel_code = template.render(**template_data)
        
        # Step 8: Validate state usage in kernel
        invalid_state_refs = self._validate_state_usage(kernel_code, state_context)
        if invalid_state_refs:
            logger.warning(f"Found invalid state references in integration kernel: {invalid_state_refs}")
            logger.info("Attempting to fix invalid state references...")
            kernel_code = await self._fix_invalid_state_references(kernel_code, invalid_state_refs, state_context)
            logger.info(f"Fixed kernel code")
            
            # Re-validate after fix
            invalid_state_refs = self._validate_state_usage(kernel_code, state_context)
            if invalid_state_refs:
                logger.error(f"Still has invalid state references after fix: {invalid_state_refs}")
        
        # Step 9: Validate generated code
        is_valid, errors = self._validate_kernel_code(kernel_code)
        if not is_valid:
            logger.error(f"Template generated invalid kernel: {errors}")
            return {
                "success": False,
                "errors": errors,
                "code": kernel_code,
                "config": config
            }
        
        # Save to accumulator if successful
        kernel_uuid = ""
        try:
            expert_names = [e['name'] for e in expert_info]
            additional_metadata = {
                "expert_count": len(expert_info),
                "experts": expert_names,
                "has_interactions": len(interaction_experts) > 0,
                "template_based": True
            }
            kernel_uuid = self.kernel_accumulator.save_kernel(
                code=kernel_code,
                prompt=f"Template-based integration kernel for experts: {', '.join(expert_names)}",
                model=self.client.model_name,
                kernel_type="integration",
                additional_metadata=additional_metadata
            )
            logger.info(f"Saved template-based integration kernel with UUID: {kernel_uuid}")
        except Exception as e:
            logger.warning(f"Failed to save kernel to accumulator: {e}")
        
        return {
            "success": True,
            "name": "apply_all_experts",
            "code": kernel_code,
            "errors": [],
            "config": config,
            "kernel_uuid": kernel_uuid,
            "template_based": True
        }
    
    async def _get_kernel_configuration(self, expert_info: list, state_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get JSON configuration for kernel integration from LLM."""
        logger.info("Getting kernel configuration from LLM")
        
        from .state_models import KernelConfiguration, ExpertConfig
        
        # Pre-classify experts based on metadata to avoid LLM classification errors
        single_particle_experts = []
        interaction_experts = []
        
        for expert in expert_info:
            expert_config = ExpertConfig(
                name=expert['name'],
                weight=expert.get('weight', 1.0),
                species_filter=None,
                species_pairs=None
            )
            
            # Use metadata to determine classification
            if expert.get('is_interaction', False):
                interaction_experts.append(expert_config)
                logger.debug(f"Pre-classified {expert['name']} as interaction expert based on metadata")
            else:
                single_particle_experts.append(expert_config)
                logger.debug(f"Pre-classified {expert['name']} as single-particle expert based on metadata")
        
        # Create initial configuration from metadata
        config = KernelConfiguration(
            single_particle_experts=single_particle_experts,
            interaction_experts=interaction_experts,
            success=True
        )
        
        # If we have a simple case, skip LLM altogether
        if len(expert_info) <= 2:
            logger.info("Using metadata-based configuration for simple case")
            return config.model_dump()
        
        # For complex cases, ask LLM to adjust weights and species filters only
        system_prompt = load_prompt("kernel_configuration_system")
        
        # Build expert descriptions
        expert_descriptions = []
        for expert in expert_info:
            desc = f"- {expert['name']}: "
            if expert.get('metadata', {}).get('description'):
                desc += expert['metadata']['description']
            else:
                desc += expert['name'].replace('_', ' ')
            if expert.get('is_interaction'):
                desc += " (interaction expert)"
            expert_descriptions.append(desc)
        
        # Add species context if available
        species_context = ""
        if any(expert.get('species_info') for expert in expert_info):
            species_mentioned = set()
            for expert in expert_info:
                if expert.get('species_info', {}).get('species_mentioned'):
                    species_mentioned.update(expert['species_info']['species_mentioned'])
            if species_mentioned:
                species_context = f"\nSpecies in use: {sorted(list(species_mentioned))}"
        
        user_prompt = load_prompt("kernel_configuration_user").format(
            expert_descriptions='\n'.join(expert_descriptions),
            species_context=species_context
        )
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        try:
            # Use structured output for reliable JSON parsing
            llm_config = await self.client.chat_structured(
                messages, 
                KernelConfiguration,
                temperature=0.1
            )
            
            logger.info("Successfully got structured kernel configuration from LLM")
            
            # Validate and fix any classification errors
            single_names = {e.name for e in llm_config.single_particle_experts}
            interaction_names = {e.name for e in llm_config.interaction_experts}
            duplicates = single_names & interaction_names
            
            if duplicates:
                logger.warning(f"LLM placed experts in both categories: {duplicates}. Using metadata to fix.")
                
                # Remove duplicates based on metadata
                for expert in expert_info:
                    if expert['name'] in duplicates:
                        if expert.get('is_interaction', False):
                            # Remove from single particle list
                            llm_config.single_particle_experts = [
                                e for e in llm_config.single_particle_experts 
                                if e.name != expert['name']
                            ]
                            logger.info(f"Removed {expert['name']} from single_particle_experts (is interaction)")
                        else:
                            # Remove from interaction list
                            llm_config.interaction_experts = [
                                e for e in llm_config.interaction_experts 
                                if e.name != expert['name']
                            ]
                            logger.info(f"Removed {expert['name']} from interaction_experts (is single)")
                
            # Convert to dictionary for compatibility with rest of code
            return llm_config.model_dump()
            
        except Exception as e:
            logger.error(f"Failed to get kernel configuration: {e}")
            # Try to extract the expert names from the malformed JSON for a simpler retry
            if len(expert_info) > 3:
                logger.warning("Too many experts may be causing JSON issues, trying simpler format")
                # Create a simple valid configuration
                config = {
                    'single_particle_experts': [],
                    'interaction_experts': []
                }
                for expert in expert_info:
                    if expert.get('is_interaction'):
                        config['interaction_experts'].append({
                            'name': expert['name'],
                            'weight': expert.get('weight', 1.0),
                            'species_pairs': None
                        })
                    else:
                        config['single_particle_experts'].append({
                            'name': expert['name'],
                            'weight': expert.get('weight', 1.0),
                            'species_filter': None
                        })
                config['success'] = True
                logger.info("Created fallback configuration from expert_info")
                return config
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"Failed to get kernel configuration: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_single_expert_calls(self, experts: List[Dict]) -> str:
        """Generate single-particle expert calls preserving SpeciesManager logic."""
        if not experts:
            return "# No single-particle experts"
        
        calls = []
        for expert in experts:
            # Safety check: verify this is not an interaction expert
            if expert.get('is_interaction', False):
                logger.warning(f"Expert {expert['name']} is marked as interaction but found in single-particle list. Skipping.")
                continue
            
            species_info = expert.get('species_info', {})
            species_filter = expert.get('species_filter')
            weight = expert.get('weight', 1.5)
            
            # Add comment about expected signature
            calls.append(f"# Expert {expert['name']} expects: (pos, vel, mass, species, particle_idx) -> ti.math.vec2")
            
            # Use species filter if provided, otherwise check species_info
            if species_filter:
                conditions = " or ".join([f"species == {s}" for s in species_filter])
                calls.append(
                    f"if {conditions}:\n"
                    f"    total_force += expert_{expert['name']}(pos, vel, mass, species, i) * {weight:.2f}"
                )
            elif species_info.get('species_mentioned') and not species_info.get('requires_all_species'):
                # Use species mentioned from analysis
                species_mentioned = species_info['species_mentioned']
                conditions = " or ".join([f"species == {s}" for s in species_mentioned])
                calls.append(
                    f"if {conditions}:\n"
                    f"    total_force += expert_{expert['name']}(pos, vel, mass, species, i) * {weight:.2f}"
                )
            else:
                # Apply to all species
                calls.append(
                    f"total_force += expert_{expert['name']}(pos, vel, mass, species, i) * {weight:.2f}"
                )
        
        return "\n".join(calls)
    
    def _select_kernel_template(self, experts: List[Dict]) -> str:
        """Select the appropriate kernel template based on expert types."""
        expert_types = set()
        for expert in experts:
            # Check metadata first, then expert_type field, then is_interaction
            if expert.get('metadata', {}).get('type'):
                expert_types.add(expert['metadata']['type'])
            elif expert.get('expert_type'):
                expert_types.add(expert['expert_type'])
            elif expert.get('is_interaction'):
                expert_types.add('force')  # Interaction experts are force-type
            else:
                expert_types.add('force')  # Default to force
        
        logger.info(f"Expert types detected: {expert_types}")
        
        # If we have any non-force experts, use multi-modal template
        if expert_types - {'force'}:
            logger.info("Using multi-modal integration kernel template")
            return 'integration_kernel_multimodal.j2'
        else:
            logger.info("Using standard force-only integration kernel template")
            return 'integration_kernel.j2'
    
    def _categorize_experts_by_type(self, expert_info: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize experts by their type for multi-modal kernel."""
        categories = {
            'force_single': [],
            'force_interaction': [],
            'sensor': [],
            'state_transition': [],
            'deposit': []
        }
        
        for expert in expert_info:
            # CRITICAL: Check is_interaction flag FIRST before expert type
            # This ensures interaction experts are never placed in single-particle category
            is_interaction = expert.get('is_interaction', False) or expert.get('metadata', {}).get('is_interaction', False)
            
            # ADDITIONAL CHECK: Look at function signature to detect interaction experts
            # If function has (p1, p2) parameters, it's definitely an interaction expert
            code_preview = expert.get('code_preview', '')
            if ('p1: ti.template(), p2: ti.template()' in code_preview or 
                'p1:ti.template(),p2:ti.template()' in code_preview or
                'p1 : ti.template(), p2 : ti.template()' in code_preview):
                is_interaction = True
                logger.info(f"Force-detected interaction expert by signature: {expert['name']}")
            
            # Get expert type from metadata or expert_type field
            expert_type = expert.get('metadata', {}).get('type') or expert.get('expert_type', 'force')
            
            logger.debug(f"Categorizing {expert['name']}: type={expert_type}, is_interaction={is_interaction}")
            
            if expert_type == 'sensor':
                categories['sensor'].append(expert)
            elif expert_type == 'state_transition':
                categories['state_transition'].append(expert)
            elif expert_type == 'deposit':
                categories['deposit'].append(expert)
            elif expert_type == 'force':
                # Use is_interaction flag to determine category
                if is_interaction:
                    categories['force_interaction'].append(expert)
                    logger.info(f"Categorized {expert['name']} as interaction expert")
                else:
                    categories['force_single'].append(expert)
                    logger.info(f"Categorized {expert['name']} as single-particle expert")
            else:
                # Default to force if type is unknown
                logger.warning(f"Unknown expert type '{expert_type}' for {expert['name']}, defaulting to force")
                if is_interaction:
                    categories['force_interaction'].append(expert)
                else:
                    categories['force_single'].append(expert)
        
        return categories
    
    def _prepare_sensor_experts(self, experts: List[Dict]) -> List[Dict]:
        """Prepare sensor experts for multi-modal template."""
        prepared = []
        for expert in experts:
            prepared_expert = {
                'name': expert['name'],
                'description': expert.get('metadata', {}).get('description', ''),
                'call': f"expert_{expert['name']}(pos, vel, species, i, tv)",
                'store_result': f"# TODO: Store sensor reading for {expert['name']}"
            }
            prepared.append(prepared_expert)
        return prepared
    
    def _prepare_state_transition_experts(self, experts: List[Dict]) -> List[Dict]:
        """Prepare state transition experts for multi-modal template."""
        prepared = []
        for expert in experts:
            prepared_expert = {
                'name': expert['name'],
                'description': expert.get('metadata', {}).get('description', ''),
                'get_current_state': f"tv.p.field[i].state",  # Placeholder
                'call': f"expert_{expert['name']}(pos, vel, mass, species, i)",
                'update_state': f"tv.p.field[i].state = new_state"
            }
            prepared.append(prepared_expert)
        return prepared
    
    def _prepare_deposit_experts(self, experts: List[Dict]) -> List[Dict]:
        """Prepare deposit experts for multi-modal template."""
        prepared = []
        for expert in experts:
            prepared_expert = {
                'name': expert['name'],
                'description': expert.get('metadata', {}).get('description', ''),
                'call': f"expert_{expert['name']}(pos, vel, species, i, tv)"
            }
            prepared.append(prepared_expert)
        return prepared
    
    def extract_function_name(self, code: str) -> Optional[str]:
        """Extract function name from expert code."""
        match = re.search(r'def\s+expert_(\w+)', code)
        if match:
            return match.group(1)
        # Try fallback for functions without expert_ prefix
        match = re.search(r'def\s+(\w+)', code)
        if match:
            return match.group(1)
        return None
    
    def _generate_expert_name(self, description: str) -> str:
        """Generate a valid function name from description."""
        # Remove common words and clean up
        words = description.lower().split()
        # Remove articles, prepositions, etc.
        stop_words = {'the', 'a', 'an', 'to', 'from', 'with', 'by', 'for', 'of', 'in', 'on', 'at', 'and', 'or', 'but'}
        words = [w for w in words if w not in stop_words]
        
        # Take first 3 meaningful words
        name_parts = []
        for word in words[:3]:
            # Remove non-alphanumeric characters
            clean_word = re.sub(r'[^a-zA-Z0-9]', '', word)
            if clean_word:
                name_parts.append(clean_word)
        
        # Join with underscores
        name = '_'.join(name_parts) if name_parts else 'expert'
        
        # Ensure it starts with a letter
        if name and name[0].isdigit():
            name = 'expert_' + name
            
        return name
    
    def _generate_interaction_expert_calls(self, experts: List[Dict]) -> str:
        """Generate interaction expert calls preserving SpeciesManager logic."""
        if not experts:
            return "# No interaction experts"
        
        calls = []
        for expert in experts:
            # Safety check: log if this doesn't look like an interaction expert
            if not expert.get('is_interaction') and 'is_interaction' in expert:
                logger.warning(f"Expert {expert['name']} is not marked as interaction but found in interaction list.")
            
            species_pairs = expert.get('species_pairs')
            species_info = expert.get('species_info', {})
            weight = expert.get('weight', 1.5)
            
            # Add comment about expected signature
            calls.append(f"# Expert {expert['name']} expects: (p1: Particle, p2: Particle) -> ti.math.vec2")
            
            if species_pairs:
                # Specific pairs provided
                conditions = []
                for pair in species_pairs:
                    if len(pair) == 2:
                        conditions.append(f"(p1.species == {pair[0]} and p2.species == {pair[1]})")
                        if pair[0] != pair[1]:
                            conditions.append(f"(p1.species == {pair[1]} and p2.species == {pair[0]})")
                
                if conditions:
                    condition_str = " or ".join(conditions)
                    calls.append(
                        f"if {condition_str}:\n"
                        f"    total_force += expert_{expert['name']}(p1, p2) * {weight:.2f}"
                    )
            elif species_info.get('interaction_pairs'):
                # Use pairs from species analysis
                pairs = species_info['interaction_pairs']
                conditions = []
                for pair in pairs:
                    if len(pair) == 2:
                        conditions.append(f"(p1.species == {pair[0]} and p2.species == {pair[1]})")
                        if pair[0] != pair[1]:
                            conditions.append(f"(p1.species == {pair[1]} and p2.species == {pair[0]})")
                
                if conditions:
                    condition_str = " or ".join(conditions)
                    calls.append(
                        f"if {condition_str}:\n"
                        f"    total_force += expert_{expert['name']}(p1, p2) * {weight:.2f}"
                    )
                else:
                    calls.append(f"total_force += expert_{expert['name']}(p1, p2) * {weight:.2f}")
            else:
                # Apply to all interactions
                calls.append(f"total_force += expert_{expert['name']}(p1, p2) * {weight:.2f}")
        
        return "\n".join(calls)
    
    def _validate_kernel_code(self, code: str) -> tuple:
        errors = []

        if "@ti.kernel" not in code:
            errors.append("Missing @ti.kernel decorator")

        if "def " not in code:
            errors.append("Missing function definition")
            
        # Check for common errors
        if "tv.p.p.field" in code:
            errors.append("Found tv.p.p.field - should be tv.p.field")
            
        if "tv.px.field" in code and "tv.px.particles" not in code:
            errors.append("Found tv.px.field - particles are accessed via tv.p.field, not tv.px.field")

        unsafe_patterns = [
            (r'exec\s*\(', "exec() is not allowed"),
            (r'eval\s*\(', "eval() is not allowed"),
            (r'__import__', "__import__ is not allowed"),
            (r'open\s*\(', "file operations not allowed"),
            (r'subprocess', "subprocess operations not allowed")
        ]

        for pattern, message in unsafe_patterns:
            if re.search(pattern, code):
                errors.append(message)

        return len(errors) == 0, errors
    
    def _validate_expert_code(self, code: str) -> tuple:
        """Validate expert function code"""
        errors = []
        
        if "@ti.func" not in code:
            errors.append("Missing @ti.func decorator")
        
        if "def expert_" not in code:
            errors.append("Function must start with 'expert_'")
        
        if "return" not in code:
            errors.append("Missing return statement")
        
        # Check for unsafe patterns
        unsafe_patterns = [
            (r'exec\s*\(', "exec() is not allowed"),
            (r'eval\s*\(', "eval() is not allowed"),
            (r'__import__', "__import__ is not allowed"),
            (r'open\s*\(', "file operations not allowed"),
            (r'subprocess', "subprocess operations not allowed")
        ]
        
        for pattern, message in unsafe_patterns:
            if re.search(pattern, code):
                errors.append(message)
        
        return len(errors) == 0, errors
    
    def _keyword_classify_behavior(self, description: str, loose: bool = False) -> str:
        """Keyword-based behavior classification as fallback for small models."""
        desc_lower = description.lower()
        
        # Check for Boids/flocking behaviors FIRST (before sensor)
        boids_keywords = ['align', 'cohesion', 'separation', 'flock', 'swarm', 'boid']
        if any(keyword in desc_lower for keyword in boids_keywords):
            if 'neighbor' in desc_lower or 'nearby' in desc_lower or 'together' in desc_lower:
                return "INTERACTION"
        
        # Check for sensor behaviors
        sensor_keywords = ['sense', 'detect', 'measure', 'count', 'read', 'scan', 'check', 'monitor', 'observe']
        if any(keyword in desc_lower for keyword in sensor_keywords):
            # Additional context check
            if any(word in desc_lower for word in ['pheromone', 'concentration', 'neighbor', 'nearby', 'around', 'environment']):
                return "SENSOR"
        
        # Check for deposit behaviors
        deposit_keywords = ['deposit', 'leave', 'mark', 'trail', 'drop', 'place', 'emit', 'release', 'secrete']
        if any(keyword in desc_lower for keyword in deposit_keywords):
            # Additional context check
            if any(word in desc_lower for word in ['pheromone', 'trail', 'marker', 'substance', 'chemical']):
                return "DEPOSIT"
        
        # Check for state transition behaviors
        state_keywords = ['become', 'change state', 'transition', 'switch', 'die', 'birth', 'alive', 'dead', 
                         'activate', 'deactivate', 'turn on', 'turn off', 'transform']
        rule_keywords = ['rule', 'if', 'when', 'condition', 'threshold']
        if any(keyword in desc_lower for keyword in state_keywords):
            return "STATE_TRANSITION"
        if any(keyword in desc_lower for keyword in rule_keywords) and 'neighbors' in desc_lower:
            return "STATE_TRANSITION"
        
        # Check for interaction behaviors (force between particles)
        interaction_keywords = ['chase', 'flee', 'follow', 'avoid', 'repel each other', 'attract each other',
                               'interact', 'between', 'towards', 'away from',
                               'align with neighbor', 'flock', 'swarm', 'cohesion', 'separation']
        if any(keyword in desc_lower for keyword in interaction_keywords):
            # Check if it mentions multiple particles or species
            if 'species' in desc_lower and re.search(r'species \d+.*species \d+', desc_lower):
                return "INTERACTION"
            if any(word in desc_lower for word in ['each other', 'one another', 'between particles']):
                return "INTERACTION"
            # Boids behaviors are interactions
            if any(word in desc_lower for word in ['align', 'cohesion', 'separation', 'flock', 'swarm']) and 'neighbor' in desc_lower:
                return "INTERACTION"
        
        # Check for single particle forces
        force_keywords = ['move', 'drift', 'fall', 'gravity', 'push', 'pull', 'attract', 'repel', 
                         'accelerate', 'velocity', 'force', 'forward', 'backward', 'upward', 'downward']
        if any(keyword in desc_lower for keyword in force_keywords):
            # Make sure it's not an interaction
            if not any(word in desc_lower for word in ['each other', 'species 0', 'species 1']):
                return "SINGLE"
        
        # If loose matching is enabled, make educated guesses
        if loose:
            # Default based on common patterns
            if 'turn' in desc_lower or 'rotate' in desc_lower:
                return "SINGLE"  # Turning is usually a force behavior
            if 'highest' in desc_lower or 'lowest' in desc_lower:
                return "SENSOR"  # Comparing values suggests sensing
            
        return "SINGLE" if loose else "UNKNOWN"
    
    async def classify_behavior(self, description: str) -> str:
        """Use LLM to classify behavior type including new expert types."""
        logger.info(f"Classifying behavior: '{description}'")
        
        # First try keyword-based classification for better reliability
        keyword_classification = self._keyword_classify_behavior(description)
        if keyword_classification != "UNKNOWN":
            logger.info(f"Using keyword classification: {keyword_classification}")
            return keyword_classification
        
        from .state_models import ExpertClassification, ExpertType, BehaviorType
        
        # Use structured output for reliable classification
        system_prompt = """You are an expert at classifying particle behavior descriptions for physics simulations.

Classify behaviors into one of these expert types:

1. FORCE: Continuous forces affecting movement (gravity, attraction, repulsion, chase, flee)
2. STATE_TRANSITION: Discrete state changes (alive/dead, on/off, mode switching)
3. SENSOR: Reading environment/neighbors (counting, detecting, measuring)
4. DEPOSIT: Writing to environment (trails, marks, pheromones)

For FORCE behaviors, also determine if it's:
- SINGLE: Forces on individual particles (gravity, drift)
- INTERACTION: Forces between particle pairs (chase, repel each other)

Examples:
- "particles fall with gravity" → FORCE (SINGLE)
- "species 0 chases species 1" → FORCE (INTERACTION)
- "cells become alive or dead based on neighbors" → STATE_TRANSITION
- "count live neighbors" → SENSOR
- "deposit pheromone trail" → DEPOSIT
- "apply birth rule - dead cells with 3 neighbors become alive" → STATE_TRANSITION"""
        
        user_prompt = f"Classify this behavior: {description}"
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        try:
            # Use structured output
            result = await self.client.chat_structured(
                messages, 
                ExpertClassification,
                temperature=0.1
            )
            
            logger.info(f"LLM classification result: {result.expert_type.value}, reasoning: {result.reasoning}")
            
            # Convert to string format expected by rest of code
            if result.expert_type == ExpertType.STATE_TRANSITION:
                return "STATE_TRANSITION"
            elif result.expert_type == ExpertType.SENSOR:
                return "SENSOR"
            elif result.expert_type == ExpertType.DEPOSIT:
                return "DEPOSIT"
            elif result.expert_type == ExpertType.FORCE:
                # Return SINGLE or INTERACTION for force behaviors
                if result.force_subtype == BehaviorType.INTERACTION:
                    return "INTERACTION"
                else:
                    return "SINGLE"
            else:
                logger.warning(f"Unexpected expert type: {result.expert_type}. Using keyword fallback.")
                return self._keyword_classify_behavior(description, loose=True)
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}. Using keyword fallback.")
            fallback = self._keyword_classify_behavior(description, loose=True)
            logger.info(f"Keyword fallback classification: {fallback}")
            return fallback
    
    async def extract_species_info(self, description: str) -> dict:
        logger.info(f"Extracting species info from: '{description}'")
        
        prompt = f"""Analyze this particle behavior description and extract species information.\n\nDescription: "{description}"\n\nLook for:\n1. Specific species mentioned by number (e.g., "species 0", "species 1", "species 2")\n2. Phrases indicating multiple species (e.g., "all species", "each species", "different species")\n3. Interactions between species (e.g., "species 0 chases species 1")\n\nIMPORTANT: Extract ONLY the exact species numbers that are explicitly mentioned. Do not assume species 0 exists unless it's explicitly mentioned.\n\nRespond with ONLY a JSON object in this format:\n{{\n    "max_species": <highest species number + 1, or null if unclear>,\n    "species_mentioned": [<list of species numbers explicitly mentioned>],\n    "requires_all_species": <true if behavior applies to all species, false otherwise>,\n    "interaction_pairs": [[0, 1], ...] // pairs of species that interact\n}}\n\nExamples:\n- "species 0 chases species 1" -> {{"max_species": 2, "species_mentioned": [0, 1], "requires_all_species": false, "interaction_pairs": [[0, 1]]}}\n- "species 2 avoids species 0" -> {{"max_species": 3, "species_mentioned": [0, 2], "requires_all_species": false, "interaction_pairs": [[2, 0]]}}  \n- "species 1 and species 2 repel each other" -> {{"max_species": 3, "species_mentioned": [1, 2], "requires_all_species": false, "interaction_pairs": [[1, 2]]}}\n- "species 4 moves upward" -> {{"max_species": 5, "species_mentioned": [4], "requires_all_species": false, "interaction_pairs": []}}\n- "all species attract each other" -> {{"max_species": null, "species_mentioned": [], "requires_all_species": true, "interaction_pairs": []}}\n- "particles fall downward" -> {{"max_species": 1, "species_mentioned": [], "requires_all_species": false, "interaction_pairs": []}}\n"""
        
        messages = [
            {'role': 'system', 'content': 'You are a species detection expert. Extract species information accurately.'},
            {'role': 'user', 'content': prompt}
        ]
        
        response = await self.client.chat(messages, temperature=0.1)
        logger.debug(f"Species extraction response: {response}")
        
        # Clean response and parse JSON
        import json
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        
        species_info = json.loads(response_clean)
        logger.info(f"Extracted species info: {species_info}")
        return species_info
    
    async def _synthesize_single_behavior(self, description: str) -> Dict[str, Any]:
        # If we have a state manager, use the state-aware synthesis
        if self.state_manager:
            return await self.synthesize_expert_with_states(description)
        
        # Otherwise, fall back to regular synthesis
        behavior_type = await self.classify_behavior(description)
        if behavior_type == "INTERACTION":
            return await self.synthesize_interaction_expert(description, state_context=None)
        elif behavior_type == "STATE_TRANSITION":
            return await self.synthesize_state_transition_expert(description, state_context=None)
        elif behavior_type == "SENSOR":
            return await self.synthesize_sensor_expert(description, state_context=None)
        elif behavior_type == "DEPOSIT":
            return await self.synthesize_deposit_expert(description, state_context=None)
        return await self.synthesize_expert(description, state_context=None)
    
    def _add_decomposition_metadata(self, result: Dict[str, Any], 
                                   original_description: str, 
                                   sub_behavior: SubBehavior,
                                   index: int, 
                                   total_count: int) -> None:
        result['is_decomposed'] = True
        result['original_description'] = original_description
        result['sub_behavior_index'] = index
        result['sub_behavior_count'] = total_count
        result['relationship'] = sub_behavior.relationship
        result['adjusted_weight'] = sub_behavior.weight
        result['weight'] = sub_behavior.weight
    
    async def _process_sub_behaviors(self, sub_behaviors: List[SubBehavior], 
                                   original_description: str,
                                   original_state_spec: Optional[Dict[str, Any]] = None,
                                   original_state_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Process and synthesize each sub-behavior."""
        results = []
        
        for i, sub_behavior in enumerate(sub_behaviors):
            logger.info(f"Synthesizing sub-behavior {i+1}/{len(sub_behaviors)}: "
                       f"'{sub_behavior.description}' (weight: {sub_behavior.weight})")
            
            try:
                # Use original state context if available, otherwise analyze per sub-behavior
                if original_state_spec and original_state_context:
                    logger.info(f"Using original state context for sub-behavior: {sub_behavior.description}")
                    # Enhance state context with explicit warnings about state names
                    enhanced_context = original_state_context.copy()
                    if 'state_summary' in enhanced_context:
                        enhanced_context['state_summary'] += "\n\nIMPORTANT: Only use the exact state names listed above. Do not invent new state names."
                    result = await self._synthesize_with_state_context(sub_behavior.description, original_state_spec, enhanced_context)
                else:
                    result = await self._synthesize_single_behavior(sub_behavior.description)
                
                self._add_decomposition_metadata(
                    result, original_description, sub_behavior, i, len(sub_behaviors)
                )
                
                # Add original state spec to the result for proper state initialization
                if original_state_spec:
                    result['state_spec'] = original_state_spec
                    result['state_context'] = original_state_context
                
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to synthesize sub-behavior '{sub_behavior.description}': {e}")
                # Continue with other sub-behaviors
        
        return results
    
    async def _synthesize_with_state_context(self, description: str, state_spec: Dict[str, Any], state_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize expert code with a given state context."""
        # Use the existing synthesize_expert method with the provided state context
        # The state_spec is included in state_context, so we don't need it as a separate parameter
        classification = await self.classify_behavior(description)
        
        if classification == "INTERACTION":
            return await self.synthesize_interaction_expert(description, state_context)
        elif classification == "STATE_TRANSITION":
            return await self.synthesize_state_transition_expert(description, state_context)
        elif classification == "SENSOR":
            return await self.synthesize_sensor_expert(description, state_context)
        elif classification == "DEPOSIT":
            return await self.synthesize_deposit_expert(description, state_context)
        else:
            return await self.synthesize_expert(description, state_context)

    async def _handle_no_decomposition(self, description: str) -> List[Dict[str, Any]]:
        result = await self._synthesize_single_behavior(description)
        return [result]
    
    async def synthesize_with_decomposition(self, description: str) -> List[Dict[str, Any]]:
        # Check if decomposition is enabled
        if not self.enable_decomposition or not self.decomposer:
            return await self._handle_no_decomposition(description)
        
        sub_behaviors = await self.decomposer.decompose_behavior(description)
        
        # If only one sub-behavior, it wasn't really decomposed
        if len(sub_behaviors) == 1:
            logger.info(f"No decomposition needed for: '{description}'")
            return await self._handle_no_decomposition(description)
        
        logger.info(f"Decomposed into {len(sub_behaviors)} sub-behaviors")
        
        sub_behaviors = self.decomposer.adjust_weights_for_balance(sub_behaviors, description)
        
        # IMPORTANT: Analyze state requirements for the ORIGINAL behavior before decomposition
        # This ensures that complex behaviors requiring states get proper state analysis
        original_state_spec = None
        original_state_context = None
        if self.state_manager:
            logger.info(f"Analyzing state requirements for original behavior: '{description}'")
            original_state_spec = await self.state_synthesizer.analyze_state_requirements(description)
            
            # Create states if needed
            if original_state_spec:
                any_states = any(original_state_spec.get(cat, {}) for cat in ['global_states', 'particle_states', 'species_states'])
                if any_states:
                    logger.info(f"Creating states for original behavior: {original_state_spec}")
                    self.state_manager.create_states_from_spec(original_state_spec)
                    original_state_context = self.state_manager.get_synthesis_context()
        
        results = await self._process_sub_behaviors(sub_behaviors, description, original_state_spec, original_state_context)
        
        if not results:
            logger.warning("All sub-behavior synthesis failed, attempting original")
            return await self._handle_no_decomposition(description)
        
        return results
    
    async def synthesize_interaction_expert(self, description: str, state_context: Optional[Dict[str, Any]] = None) -> dict:
        logger.info(f"Synthesizing expert for: '{description}'")
        
        # Use the router to classify the behavior
        classification = await self.classify_behavior(description)
        
        # Extract species information
        species_info = await self.extract_species_info(description)
        
        if classification == "SINGLE":
            logger.info("Behavior classified as SINGLE-PARTICLE, using standard synthesis")
            result = await self.synthesize_expert(description, state_context)
            # Add species info to result
            if result["success"]:
                result["species_info"] = species_info
            return result
        
        logger.info("Behavior classified as INTERACTION, using interaction synthesis")
        
        system_prompt = load_prompt("expert_interaction_synthesis_system")
        
        # Add state context to user prompt if available
        if state_context:
            # Generate state access examples from the state_spec in context
            state_examples = ""
            if state_context.get('state_spec'):
                # Use the same approach as integration kernel for consistency
                if self.is_small_model:
                    typed_context = self.state_synthesizer.generate_simplified_state_context(state_context.get('state_spec', {}))
                    state_examples = typed_context.get('state_summary', '')
                else:
                    # For full models, use the detailed access examples
                    state_examples = state_context.get('access_examples', '')
                    if not state_examples and self.state_manager:
                        state_examples = self.state_manager.generate_state_access_examples()
            
            # Format user prompt with state context
            user_prompt_template = load_prompt("expert_interaction_synthesis_user")
            user_prompt = user_prompt_template.format(
                description=description,
                state_context=state_examples if state_examples else "No custom states available"
            )
        else:
            # Format with empty state context
            user_prompt = load_prompt("expert_interaction_synthesis_user").format(
                description=description,
                state_context="No custom states available")

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        try:
            response = await self.client.chat(messages, temperature=0.5, think=False)
            
            logger.info(f"Raw interaction response:\n{response}")
            
            code = self.extract_code(response)
            
            logger.info(f"Extracted interaction code:\n{code}")
            
            # Validate state usage
            invalid_state_refs = self._validate_state_usage(code, state_context)
            if invalid_state_refs:
                logger.warning(f"Found invalid state references in interaction expert: {invalid_state_refs}")
                logger.info("Attempting to fix invalid state references...")
                code = await self._fix_invalid_state_references(code, invalid_state_refs, state_context)
                logger.info(f"Fixed interaction code:\n{code}")
                
                # Re-validate after fix
                invalid_state_refs = self._validate_state_usage(code, state_context)
                if invalid_state_refs:
                    logger.error(f"Still has invalid state references after fix: {invalid_state_refs}")
            
            # Validate the interaction expert code
            is_valid, errors = self._validate_expert_code(code)
            
            # Run error detector for additional checks
            detected_errors = self.error_detector.detect_errors(code)
            if detected_errors:
                error_summary = self.error_detector.get_error_summary(detected_errors)
                logger.warning(f"Detected {len(detected_errors)} potential issues in interaction expert code")
                logger.warning(f"Error summary: {error_summary}")
                
                # Add detected errors to the response for logging
                detected_error_messages = [
                    f"Line {e['line']}: {e['severity']} - {e['message']}" 
                    for e in detected_errors if e['severity'] == 'error'
                ]
                errors.extend(detected_error_messages)
            
            if not is_valid:
                logger.warning(f"Generated invalid interaction expert: {errors}")
                return {
                    "success": False,
                    "code": code,
                    "description": description,
                    "errors": errors,
                    "raw_response": response,
                    "detected_errors": detected_errors
                }
            
            # Extract function name
            name_match = re.search(r'def\s+expert_(\w+)', code)
            if not name_match:
                return {
                    "success": False,
                    "code": code,
                    "description": description,
                    "errors": ["Could not extract function name"],
                    "raw_response": response
                }
            
            name = name_match.group(1)
            
            # Save successful interaction kernel to accumulator
            kernel_uuid = ""
            try:
                additional_metadata = {
                    "description": description,
                    "is_interaction": True
                }
                kernel_uuid = self.kernel_accumulator.save_kernel(
                    code=code,
                    prompt=description,
                    model=self.client.model_name,
                    kernel_type="expert",
                    additional_metadata=additional_metadata
                )
                logger.info(f"Saved interaction expert kernel with UUID: {kernel_uuid}")
            except Exception as e:
                logger.warning(f"Failed to save interaction kernel to accumulator: {e}")
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "description": description,
                "metadata": {
                    "type": "force",
                    "returns": "ti.math.vec2",
                    "description": description
                },
                "errors": [],
                "raw_response": response,
                "is_interaction": True,
                "detected_errors": detected_errors if detected_errors else [],
                "kernel_uuid": kernel_uuid,
                "species_info": species_info
            }
            
        except Exception as e:
            logger.error(f"Interaction expert synthesis failed: {e}")
            return {
                "success": False,
                "code": "",
                "description": description,
                "errors": [str(e)],
                "raw_response": ""
            }

    async def synthesize_state_transition_expert(self, description: str, state_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synthesize a state transition expert for discrete state changes like cellular automata."""
        logger.info(f"Synthesizing state transition expert for: '{description}'")
        
        try:
            # Load state transition specific prompts
            system_prompt = load_prompt("expert_state_transition_system")
            user_prompt_template = load_prompt("expert_state_transition_user")
            
            # Add state context to user prompt if available
            state_examples = ""
            if state_context and state_context.get('state_spec'):
                if self.is_small_model:
                    typed_context = self.state_synthesizer.generate_simplified_state_context(state_context.get('state_spec', {}))
                    state_examples = typed_context.get('state_summary', '')
                else:
                    state_examples = state_context.get('access_examples', '')
                    if not state_examples and self.state_manager:
                        state_examples = self.state_manager.generate_state_access_examples()
            
            # Generate a name from the description
            name = self._generate_expert_name(description)
            
            user_prompt = user_prompt_template.format(
                description=description,
                state_context=state_examples if state_examples else "No custom states available",
                name=name,
                parameters="pos: ti.math.vec2, vel: ti.math.vec2, state: ti.i32, species: ti.i32"
            )
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            
            response = await self.client.chat(messages, temperature=0.3)
            code = self.extract_code(response)
            name = self.extract_function_name(code) or "state_transition"
            
            # Validate the function returns ti.i32
            if "-> ti.i32" not in code:
                logger.warning("State transition expert doesn't return ti.i32, fixing...")
                code = code.replace("-> ti.math.vec2", "-> ti.i32")
            
            logger.info(f"Successfully synthesized state transition expert: {name}")
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "description": description,
                "expert_type": "state_transition",
                "metadata": {
                    "type": "state_transition",
                    "returns": "ti.i32",
                    "description": description,
                    "state_context": state_context
                },
                "errors": [],
                "prompt": user_prompt,
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"State transition expert synthesis failed: {e}", exc_info=True)
            return {
                "success": False,
                "errors": [str(e)],
                "raw_response": ""
            }
    
    async def synthesize_sensor_expert(self, description: str, state_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synthesize a sensor expert for reading environment values."""
        logger.info(f"Synthesizing sensor expert for: '{description}'")
        
        try:
            # Load sensor specific prompts
            system_prompt = load_prompt("expert_sensor_system")
            user_prompt_template = load_prompt("expert_sensor_user")
            
            # Add state context
            state_examples = ""
            if state_context and state_context.get('state_spec'):
                if self.is_small_model:
                    typed_context = self.state_synthesizer.generate_simplified_state_context(state_context.get('state_spec', {}))
                    state_examples = typed_context.get('state_summary', '')
                else:
                    state_examples = state_context.get('access_examples', '')
            
            # Add explicit list of available states to prevent hallucination
            if state_context and state_context.get('state_summary'):
                state_examples = state_context.get('state_summary', '') + "\n\n" + (state_examples or "")
                logger.info(f"Sensor synthesis state context: {state_examples}")
            
            # Generate a name from the description
            name = self._generate_expert_name(description)
            
            user_prompt = user_prompt_template.format(
                description=description,
                state_context=state_examples if state_examples else "No custom states available",
                name=name,
                parameters="pos: ti.math.vec2, vel: ti.math.vec2, species: ti.i32, particle_idx: ti.i32, tv: ti.template()"
            )
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            
            response = await self.client.chat(messages, temperature=0.3)
            code = self.extract_code(response)
            name = self.extract_function_name(code) or "sensor"
            
            # Validate the function returns ti.f32
            if "-> ti.f32" not in code:
                logger.warning("Sensor expert doesn't return ti.f32, fixing...")
                code = code.replace("-> ti.math.vec2", "-> ti.f32")
            
            # Ensure sensor has return statement
            if "return" not in code:
                logger.warning("Sensor expert missing return statement, adding default return")
                # Find the end of the function and add return before it
                lines = code.split('\n')
                # Find last non-empty line that's still part of function (not starting a new function)
                insert_index = len(lines) - 1
                while insert_index > 0 and (not lines[insert_index].strip() or lines[insert_index].startswith('@')):
                    insert_index -= 1
                # Add return statement with safe default
                lines.insert(insert_index + 1, "    return 0.0  # Default return added - no sensor reading")
                code = '\n'.join(lines)
            
            logger.info(f"Successfully synthesized sensor expert: {name}")
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "description": description,
                "expert_type": "sensor",
                "metadata": {
                    "type": "sensor",
                    "returns": "ti.f32",
                    "description": description,
                    "state_context": state_context
                },
                "errors": [],
                "prompt": user_prompt,
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"Sensor expert synthesis failed: {e}", exc_info=True)
            return {
                "success": False,
                "errors": [str(e)],
                "raw_response": ""
            }
    
    async def synthesize_deposit_expert(self, description: str, state_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synthesize a deposit expert for writing to environment."""
        logger.info(f"Synthesizing deposit expert for: '{description}'")
        
        try:
            # Load deposit specific prompts
            system_prompt = load_prompt("expert_deposit_system")
            user_prompt_template = load_prompt("expert_deposit_user")
            
            # Add state context
            state_examples = ""
            if state_context and state_context.get('state_spec'):
                if self.is_small_model:
                    typed_context = self.state_synthesizer.generate_simplified_state_context(state_context.get('state_spec', {}))
                    state_examples = typed_context.get('state_summary', '')
                else:
                    state_examples = state_context.get('access_examples', '')
            
            # Generate a name from the description
            name = self._generate_expert_name(description)
            
            user_prompt = user_prompt_template.format(
                description=description,
                state_context=state_examples if state_examples else "No custom states available",
                name=name,
                parameters="pos: ti.math.vec2, vel: ti.math.vec2, species: ti.i32, particle_idx: ti.i32, tv: ti.template()"
            )
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            
            response = await self.client.chat(messages, temperature=0.3)
            code = self.extract_code(response)
            name = self.extract_function_name(code) or "deposit"
            
            # Validate the function has no return type (void)
            if "->" in code and "def expert" in code:
                logger.warning("Deposit expert has return type, removing...")
                # Remove return type annotation
                import re
                code = re.sub(r'\s*->\s*[^:]+:', ':', code)
            
            logger.info(f"Successfully synthesized deposit expert: {name}")
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "description": description,
                "expert_type": "deposit",
                "metadata": {
                    "type": "deposit",
                    "returns": "void",
                    "description": description,
                    "state_context": state_context
                },
                "errors": [],
                "prompt": user_prompt,
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"Deposit expert synthesis failed: {e}", exc_info=True)
            return {
                "success": False,
                "errors": [str(e)],
                "raw_response": ""
            }


class PureLLMSynthesizer:

    def __init__(self, model_name: Optional[str] = None, auto_correct: bool = True, enable_decomposition: bool = True, tolvera_instance=None):
        self.synthesizer = PoEExpertSynthesizer(model_name=model_name, auto_correct=auto_correct, enable_decomposition=enable_decomposition, tolvera_instance=tolvera_instance)
        logger.info(
            f"Initialized PureLLMSynthesizer with model: {model_name or 'default'}, auto_correct: {auto_correct}, enable_decomposition={enable_decomposition}")

    async def synthesize_expert(
            self,
            description: str,
            weight: float = 1.0) -> SimpleProgrammaticExpert:

        logger.info(f"Synthesizing expert for: '{description}'")

        start_time = time.time()
        result = await self.synthesizer.synthesize_expert(description)
        synthesis_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Raw LLM response: {result.get('raw_response', 'No response')}")

        logger.debug(
            f"Extracted code: {result.get('code', 'No code extracted')}")

        # Extract correction info
        correction_result = result.get("correction_result", None)
        
        csv_logger.log_synthesis_attempt(
            user_description=description,
            llm_prompt=result.get("prompt", ""),
            raw_response=result.get("raw_response", ""),
            extracted_code=result.get("code", ""),
            success=result["success"],
            errors=result.get("errors", []),
            model_name=self.synthesizer.client.model_name,
            expert_name=result.get("name", None),
            synthesis_time_ms=synthesis_time_ms,
            detected_errors=result.get("detected_errors", []),
            correction_attempted=result.get("correction_attempted", False),
            correction_succeeded=correction_result['success'] if correction_result else False,
            correction_history=correction_result['correction_history'] if correction_result else None,
            final_code=correction_result['corrected_code'] if correction_result and correction_result['success'] else result.get("code", "")
        )

        if result["success"]:
            expert = SimpleProgrammaticExpert(
                name=result["name"],
                code=result["code"],
                weight=weight
            )
            expert.metadata["description"] = description
            expert.metadata["synthesis_method"] = "pure_llm"
            expert.metadata["raw_llm_response"] = result.get(
                "raw_response", "")

            logger.info(f"Successfully created expert '{result['name']}'")

            return expert
        else:
            # Log errors
            logger.error(
                f"Synthesis failed: {result.get('errors', ['Unknown error'])}")

            raise ValueError(
                f"Failed to synthesize expert: {result.get('errors', ['Unknown error'])}")


async def create_expert_from_description(
        description: str,
        model_name: Optional[str] = None,
        weight: float = 1.0) -> SimpleProgrammaticExpert:
    synthesizer = PureLLMSynthesizer(model_name)
    return await synthesizer.synthesize_expert(description, weight)


# Was trying to see if we wanted async or sync functionality here.
def create_expert_from_description_sync(
        description: str,
        model_name: Optional[str] = None,
        weight: float = 1.0) -> SimpleProgrammaticExpert:
    return asyncio.run(
        create_expert_from_description(
            description,
            model_name,
            weight))
