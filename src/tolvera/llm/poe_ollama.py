
"""
Ollama integration for PoE expert synthesis.

This module provides the LLM integration for generating Taichi-compatible expert functions from natural language descriptions.
"""

import logging
import re
from typing import Optional, List, Dict, Any, Tuple
import ollama
from .prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class OllamaModelManager:

    def __init__(self):
        self.compatible_models = [
            "qwen2.5:3b", "qwen2.5:7b", "qwen2.5-coder:7b",
            "gemma2:2b", "gemma2:9b",
            "qwen2.5:3b", "llama3.2:1b",
            "mistral:7b", "mistral-nemo:12b"
        ]
        self.default_model = "qwen2.5:3b"
        try:
            self.client = ollama.Client()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Ollama client. Is Ollama running? Error: {e}")

    def check_ollama_running(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def list_available_models(self) -> List[str]:
        try:
            models_info = self.client.list()
            return [model['name'] for model in models_info.get('models', [])]
        except Exception:
            return []

    def ensure_compatible_model(
            self, requested_model: Optional[str] = None) -> str:
        available = self.list_available_models()

        if requested_model and requested_model in available:
            logger.info(f"Using requested model: {requested_model}")
            return requested_model

        for model in self.compatible_models:
            if model in available:
                logger.info(f"Using compatible model: {model}")
                return model

        logger.warning(f"No compatible model found. Available: {available}")
        logger.info(f"Pulling default model: {self.default_model}")

        try:
            ollama.pull(self.default_model)
            return self.default_model
        except Exception as e:
            raise RuntimeError(f"Failed to pull default model: {e}")


class OllamaClient:

    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        self.model_manager = OllamaModelManager()

        if not self.model_manager.check_ollama_running():
            raise RuntimeError(
                "Ollama is not running. Please start it with 'ollama serve'")

        self.model_name = self.model_manager.ensure_compatible_model(
            model_name)
        self.client = ollama.AsyncClient()
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")

    async def chat(self,
                   messages: List[Dict[str,
                                       str]],
                   temperature: float = 0.7,
                   max_tokens: int = 2000,
                   think: bool = False) -> str:
        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                },
                # think=think
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


class PoEExpertSynthesizer:

    def __init__(self, model_name: Optional[str] = None):
        self.client = OllamaClient(model_name)

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
            if not is_valid:
                logger.warning(f"Generated invalid code: {errors}")
                return {
                    "success": False,
                    "code": code,
                    "errors": errors,
                    "raw_response": response}

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

            return {
                "success": True,
                "name": name,
                "code": code,
                "description": description,
                "errors": [],
                "raw_response": response}

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

            logger.info(f"Extracted kernel code:\n{code}")

            is_valid, errors = self._validate_kernel_code(code)
            if not is_valid:
                logger.warning(f"Generated invalid kernel: {errors}")
                return {
                    "success": False,
                    "code": code,
                    "errors": errors,
                    "raw_response": response}

            name_match = re.search(r'def\s+(\w+)', code)
            if not name_match:
                return {
                    "success": False,
                    "code": code,
                    "errors": ["Could not extract kernel function name"],
                    "raw_response": response}

            name = name_match.group(1)

            return {
                "success": True,
                "name": name,
                "code": code,
                "errors": [],
                "raw_response": response}

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
    
    async def synthesize_interaction_expert(self, description: str) -> dict:
        logger.info(f"Synthesizing interaction expert for: '{description}'")
        
        # TODO: Fix this up to a router...not just something like this keyword search because this doesn't work well.
        # Detect if this is an interaction description
        interaction_keywords = ['interact', 'between', 'chase', 'avoid', 'attract', 'repel', 'flock', 'hunt', 'flee']
        is_interaction = any(keyword in description.lower() for keyword in interaction_keywords)
        
        if not is_interaction:
            logger.info("No interaction keywords detected, using single-particle synthesis")
            return await self.synthesize_expert(description)
        
        logger.info("Detected interaction description, using interaction synthesis")
        
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
            if not is_valid:
                logger.warning(f"Generated invalid interaction expert: {errors}")
                return {
                    "success": False,
                    "code": code,
                    "description": description,
                    "errors": errors,
                    "raw_response": response
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
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "description": description,
                "errors": [],
                "raw_response": response,
                "is_interaction": True
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
