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

logger = logging.getLogger(__name__)
csv_logger = get_logger()


class PoEExpertSynthesizer:

    def __init__(self, model_name: Optional[str] = None, auto_correct: bool = True, 
                 accumulator_path: str = "generated_kernels/kernels_repository.py"):
        self.client = OllamaClient(model_name)
        self.error_detector = TaichiErrorDetector()
        self.error_corrector = TaichiErrorCorrector(model_name)
        self.auto_correct = auto_correct
        self.kernel_accumulator = KernelAccumulator(accumulator_path)

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

    async def synthesize_expert(self, description: str) -> Dict[str, Any]:

        logger.info(f"Synthesizing expert for: '{description}'")

        system_prompt = load_prompt("expert_synthesis_system")
        user_prompt = load_prompt("expert_synthesis_user").format(
            description=description)

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
                "errors": [],
                "raw_response": response,
                "detected_errors": detected_errors if detected_errors else [],
                "correction_attempted": correction_result is not None,
                "correction_result": correction_result,
                "kernel_uuid": kernel_uuid}

        except Exception as e:
            logger.error(f"Expert synthesis failed: {e}")
            return {
                "success": False,
                "code": "",
                "errors": [
                    str(e)],
                "raw_response": ""}

    async def synthesize_integration_kernel(self, expert_info: list) -> dict:
        logger.info(
            f"Synthesizing integration kernel for {len(expert_info)} experts")

        # Separate single-particle and interaction experts
        single_experts = [e for e in expert_info if not e.get('is_interaction', False)]
        interaction_experts = [e for e in expert_info if e.get('is_interaction', False)]
        
        # Check if we have interaction experts
        has_interactions = len(interaction_experts) > 0
        
        if has_interactions:
            # Use the new interaction kernel template
            single_expert_calls = [
                f"            total_force += expert_{expert['name']}(pos, vel, mass, species) * {expert['weight']:.2f}" 
                for expert in single_experts
            ]
            single_expert_calls_str = "\n".join(single_expert_calls) if single_experts else "            # No single-particle experts"
            
            interaction_expert_calls = [
                f"                    total_force += expert_{expert['name']}(p1, p2) * {expert['weight']:.2f}"
                for expert in interaction_experts
            ]
            interaction_expert_calls_str = "\n".join(interaction_expert_calls)
            
            system_prompt = load_prompt("kernel_integration_interaction_system")
            user_prompt = load_prompt("kernel_integration_interaction_user").format(
                single_expert_calls_str=single_expert_calls_str,
                interaction_expert_calls_str=interaction_expert_calls_str
            )
        else:
            # Use the original single-particle kernel template
            expert_calls = [
                f"            total_force += expert_{expert['name']}(pos, vel, mass, species) * {expert['weight']:.2f}" 
                for expert in expert_info
            ]
            expert_calls_str = "\n".join(expert_calls)
            
            system_prompt = load_prompt("kernel_integration_system")
            user_prompt = load_prompt("kernel_integration_user").format(
                expert_calls_str=expert_calls_str
            )
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        try:
            response = await self.client.chat(messages, temperature=0.0, think=False)

            logger.info(f"Raw kernel response:\n{response}")

            code = self.extract_code(response)
            
            # Fix common LLM errors
            if "tv.p.p.field" in code:
                logger.warning("Found tv.p.p.field error, fixing...")
                code = code.replace("tv.p.p.field", "tv.p.field")

            logger.info(f"Extracted kernel code:\n{code}")

            is_valid, errors = self._validate_kernel_code(code)
            
            # Run error detector for additional checks
            detected_errors = self.error_detector.detect_errors(code)
            if detected_errors:
                error_summary = self.error_detector.get_error_summary(detected_errors)
                logger.warning(f"Detected {len(detected_errors)} potential issues in kernel code")
                logger.warning(f"Error summary: {error_summary}")
                
                # Add detected errors to the response for logging
                detected_error_messages = [
                    f"Line {e['line']}: {e['severity']} - {e['message']}" 
                    for e in detected_errors if e['severity'] == 'error'
                ]
                errors.extend(detected_error_messages)
            
            if not is_valid:
                logger.warning(f"Generated invalid kernel: {errors}")
                return {
                    "success": False,
                    "code": code,
                    "errors": errors,
                    "raw_response": response,
                    "detected_errors": detected_errors}

            name_match = re.search(r'def\s+(\w+)', code)
            if not name_match:
                return {
                    "success": False,
                    "code": code,
                    "errors": ["Could not extract kernel function name"],
                    "raw_response": response}

            name = name_match.group(1)

            # Save successful integration kernel to accumulator
            kernel_uuid = ""
            try:
                # Build expert list for metadata
                expert_names = [e['name'] for e in expert_info]
                additional_metadata = {
                    "expert_count": len(expert_info),
                    "experts": expert_names,
                    "has_interactions": any(e.get('is_interaction', False) for e in expert_info)
                }
                kernel_uuid = self.kernel_accumulator.save_kernel(
                    code=code,
                    prompt=f"Integration kernel for experts: {', '.join(expert_names)}",
                    model=self.client.model_name,
                    kernel_type="integration",
                    additional_metadata=additional_metadata
                )
                logger.info(f"Saved integration kernel with UUID: {kernel_uuid}")
            except Exception as e:
                logger.warning(f"Failed to save integration kernel to accumulator: {e}")
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "errors": [],
                "raw_response": response,
                "detected_errors": detected_errors if detected_errors else [],
                "kernel_uuid": kernel_uuid}

        except Exception as e:
            logger.error(f"Kernel synthesis failed: {e}")
            return {
                "success": False,
                "code": "",
                "errors": [
                    str(e)],
                "raw_response": ""}

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
    
    async def classify_behavior(self, description: str) -> str:
        """Use LLM to classify whether a behavior is single-particle or interaction based."""
        logger.info(f"Classifying behavior: '{description}'")
        
        system_prompt = load_prompt("behavior_router_system")
        user_prompt = load_prompt("behavior_router_user").format(description=description)
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        try:
            response = await self.client.chat(messages, temperature=0.1, max_tokens=10)
            classification = response.strip().upper()
            
            if classification not in ["SINGLE", "INTERACTION"]:
                logger.warning(f"Invalid classification response: {response}. Defaulting to SINGLE.")
                classification = "SINGLE"
            
            logger.info(f"Behavior classified as: {classification}")
            return classification
            
        except Exception as e:
            logger.error(f"Classification failed: {e}. Defaulting to SINGLE.")
            return "SINGLE"
    
    async def synthesize_interaction_expert(self, description: str) -> dict:
        logger.info(f"Synthesizing expert for: '{description}'")
        
        # Use the router to classify the behavior
        classification = await self.classify_behavior(description)
        
        if classification == "SINGLE":
            logger.info("Behavior classified as SINGLE-PARTICLE, using standard synthesis")
            return await self.synthesize_expert(description)
        
        logger.info("Behavior classified as INTERACTION, using interaction synthesis")
        
        system_prompt = load_prompt("expert_interaction_synthesis_system")
        user_prompt = load_prompt("expert_interaction_synthesis_user").format(description=description)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        try:
            response = await self.client.chat(messages, temperature=0.5, think=False)
            
            logger.info(f"Raw interaction response:\n{response}")
            
            code = self.extract_code(response)
            
            logger.info(f"Extracted interaction code:\n{code}")
            
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
                "errors": [],
                "raw_response": response,
                "is_interaction": True,
                "detected_errors": detected_errors if detected_errors else [],
                "kernel_uuid": kernel_uuid
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


class PureLLMSynthesizer:

    def __init__(self, model_name: Optional[str] = None, auto_correct: bool = True):
        self.synthesizer = PoEExpertSynthesizer(model_name, auto_correct)
        logger.info(
            f"Initialized PureLLMSynthesizer with model: {model_name or 'default'}, auto_correct: {auto_correct}")

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
