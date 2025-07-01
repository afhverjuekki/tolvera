
"""
Ollama integration for PoE expert synthesis.

This module provides LLM integration for generating Taichi-compatible
expert functions from natural language descriptions.
"""

import logging
import re
from typing import Optional, List, Dict, Any, Tuple
import ollama

logger = logging.getLogger(__name__)


class OllamaModelManager:
    """Manages Ollama models and ensures compatibility."""
    
    def __init__(self):
        self.compatible_models = [
            "llama3.2:3b", "qwen2.5:7b", "qwen2.5-coder:7b",
            "gemma2:2b", "gemma2:9b",
            "llama3.2:3b", "llama3.2:1b",
            "mistral:7b", "mistral-nemo:12b"
        ]
        self.default_model = "llama3.2:3b"
        try:
            self.client = ollama.Client()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama client. Is Ollama running? Error: {e}")

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
    
    def ensure_compatible_model(self, requested_model: Optional[str] = None) -> str:
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
    """Client for interacting with Ollama API."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.model_manager = OllamaModelManager()
        
        if not self.model_manager.check_ollama_running():
            raise RuntimeError("Ollama is not running. Please start it with 'ollama serve'")
        
        self.model_name = self.model_manager.ensure_compatible_model(model_name)
        self.client = ollama.AsyncClient()
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000, think: bool = False) -> str:
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
    """Synthesizes Taichi-compatible expert functions using Ollama."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.client = OllamaClient(model_name)
        
    def extract_code(self, response: str) -> str:
        # Try to find code between triple backticks
        code_match = re.search(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try to find @ti.func definition directly
        func_match = re.search(r'(@ti\.func.*?)(?=\n@|\n\n|\Z)', response, re.DOTALL)
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
        
        system_prompt = """You are an expert Taichi programmer. Generate a single, complete Taichi function that implements a specific particle behavior.\n\nFollow these rules precisely:\n1.  The function must have the decorator `@ti.func`.\n2.  The function signature must be `def expert_NAME(tv: ti.template(), i: ti.i32) -> ti.math.vec2:`, where `NAME` is a descriptive name based on the behavior (e.g., `gravity`, `attract_to_center`).\n3.  The function must return a 2D force vector: `ti.Vector([fx, fy])`. This vector will be added to the particle's velocity.\n4.  Access particle data using `tv.p.field[i]`. For example, `tv.p.field[i].pos` for position.\n5.  Use Taichi math functions like `ti.sqrt`, `ti.sin`, `ti.math.normalize`, etc. Do NOT use `math.sqrt`.\n6.  For type casting, use `ti.cast(value, ti.f32)`. Do NOT use `ti.f32(value)`.\n7.  **CRITICAL**: Do NOT use `return` inside of loops (`for`, `while`) or conditional blocks (`if`). Calculate forces and store them in variables, then return the final combined force at the end of the function.\n8.  Keep the function focused on a single, specific behavior as described. Do not add unrelated logic.\n9.  **CRITICAL**: All variables must be defined with a value before they are used.\n\nAvailable Tolvera properties:\n- `tv.p.field[i].pos`: vec2, particle position\n- `tv.p.field[i].vel`: vec2, particle velocity\n- `tv.p.field[i].mass`: f32, particle mass\n- `tv.p.field[i].size`: f32, particle size\n- `tv.x`, `tv.y`: f32, screen dimensions\n- `tv.pn`: i32, total particle count\n\nGenerate ONLY the complete function code, starting with `@ti.func`. Do not include any explanations or markdown."""

        example_user_1 = "Gravity"
        example_assistant_1 = """@ti.func\ndef expert_gravity(tv: ti.template(), i: ti.i32) -> ti.math.vec2:\n    # A simple, constant downward force.\n    gravity_strength = 0.05\n    return ti.Vector([0.0, gravity_strength])"""

        example_user_2 = "Attraction to center"
        example_assistant_2 = """@ti.func\ndef expert_attract_to_center(tv: ti.template(), i: ti.i32) -> ti.math.vec2:\n    # Force that pulls particle towards the screen center.\n    center = ti.Vector([tv.x / 2, tv.y / 2])\n    direction = center - tv.p.field[i].pos\n    # Normalize to get direction, handle zero-vector case.\n    direction_norm = ti.math.normalize(direction)\n    strength = 0.01\n    return direction_norm * strength"""
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': example_user_1},
            {'role': 'assistant', 'content': example_assistant_1},
            {'role': 'user', 'content': example_user_2},
            {'role': 'assistant', 'content': example_assistant_2},
            {'role': 'user', 'content': f"Now, generate the Taichi function for the description: \"{description}\"."}
        ]

        logger.debug(f"LLM Messages: {messages}")

        try:
            response = await self.client.chat(messages, temperature=0.7, think=False)
            
            logger.info(f"Raw LLM Response:\n{response}")
            
            code = self.extract_code(response)
            
            logger.info(f"Extracted Code:\n{code}")
            
            is_valid, errors = self.validate_expert_code(code)
            if not is_valid:
                logger.warning(f"Generated invalid code: {errors}")
                return {"success": False, "code": code, "errors": errors, "raw_response": response}
            
            name_match = re.search(r'def\s+(\w+)', code)
            if not name_match:
                return {"success": False, "code": code, "errors": ["Could not extract function name"], "raw_response": response}
            
            name = name_match.group(1)
            
            return {"success": True, "name": name, "code": code, "description": description, "errors": [], "raw_response": response}
            
        except Exception as e:
            logger.error(f"Expert synthesis failed: {e}")
            return {"success": False, "code": "", "errors": [str(e)], "raw_response": ""}
    
    async def synthesize_integration_kernel(self, expert_info: list) -> dict:
        """Synthesize a Taichi kernel that integrates all active experts."""
        
        logger.info(f"Synthesizing integration kernel for {len(expert_info)} experts")
        
        expert_calls = [f"            total_force += {expert['name']}(tv, i) * {expert['weight']:.2f}" for expert in expert_info]
        expert_calls_str = "\n".join(expert_calls)

        system_prompt = "You are a Taichi programmer. Your task is to generate a complete, precise Taichi kernel to integrate particle forces. Do not add any logic or functions that are not in the template. Generate ONLY the complete kernel function code as defined in the template. Do not add any other text, explanations, or markdown formatting."
        
        user_prompt = f"""Use this exact template:\n\n@ti.kernel\ndef apply_all_experts(tv: ti.template(), dt: ti.f32):\n    for i in range(tv.pn):\n        if tv.p.field[i].active > 0:\n            # Initialise a total force for the particle\n            total_force = ti.Vector([0.0, 0.0])\n\n            # Accumulate forces from all experts\n{expert_calls_str}\n\n            # Apply the final combined force to the particle\n            tv.p.field[i].vel += total_force * dt\n            # Apply simple damping to prevent runaway speeds\n            tv.p.field[i].vel *= 0.98\n            # Update particle position based on new velocity\n            tv.p.field[i].pos += tv.p.field[i].vel * dt\n"""
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        try:
            response = await self.client.chat(messages, temperature=0.7, think=False)
            
            logger.info(f"Raw kernel response:\n{response}")
            
            code = self.extract_code(response)
            
            logger.info(f"Extracted kernel code:\n{code}")
            
            is_valid, errors = self._validate_kernel_code(code)
            if not is_valid:
                logger.warning(f"Generated invalid kernel: {errors}")
                return {"success": False, "code": code, "errors": errors, "raw_response": response}
            
            name_match = re.search(r'def\s+(\w+)', code)
            if not name_match:
                return {"success": False, "code": code, "errors": ["Could not extract kernel function name"], "raw_response": response}
            
            name = name_match.group(1)
            
            return {"success": True, "name": name, "code": code, "errors": [], "raw_response": response}
            
        except Exception as e:
            logger.error(f"Kernel synthesis failed: {e}")
            return {"success": False, "code": "", "errors": [str(e)], "raw_response": ""}
    
    def _validate_kernel_code(self, code: str) -> tuple:
        """Validate kernel code for Taichi compatibility."""
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
