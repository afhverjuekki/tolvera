"""
Ollama integration for PoE expert synthesis.

This module provides LLM integration for generating Taichi-compatible
expert functions from natural language descriptions.
"""

import logging
import re
from typing import Optional, List, Dict, Any, Tuple
import subprocess

logger = logging.getLogger(__name__)


class OllamaModelManager:
    """Manages Ollama models and ensures compatibility."""
    
    def __init__(self):
        self.compatible_models = [
            "qwen2.5:3b", "qwen2.5:7b", "qwen2.5-coder:7b",
            "gemma2:2b", "gemma2:9b",
            "llama3.2:3b", "llama3.2:1b",
            "mistral:7b", "mistral-nemo:12b"
        ]
        self.default_model = "qwen2.5:3b"
        
    def check_ollama_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def list_available_models(self) -> List[str]:
        """List models available locally."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
        except:
            return []
    
    def ensure_compatible_model(self, requested_model: Optional[str] = None) -> str:
        """Ensure a compatible model is available."""
        available = self.list_available_models()
        
        # If requested model is available, use it
        if requested_model and requested_model in available:
            logger.info(f"Using requested model: {requested_model}")
            return requested_model
        
        # Check for compatible models
        for model in self.compatible_models:
            if model in available:
                logger.info(f"Using compatible model: {model}")
                return model
        
        # No compatible model found
        logger.warning(f"No compatible model found. Available: {available}")
        logger.info(f"Pulling default model: {self.default_model}")
        
        # Try to pull default model
        try:
            subprocess.run(
                ["ollama", "pull", self.default_model],
                timeout=300  # 5 minutes
            )
            return self.default_model
        except:
            raise RuntimeError("Failed to pull default model")


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, model_name: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.model_manager = OllamaModelManager()
        
        # Ensure Ollama is running
        if not self.model_manager.check_ollama_running():
            raise RuntimeError("Ollama is not running. Please start it with 'ollama serve'")
        
        # Ensure compatible model
        self.model_name = self.model_manager.ensure_compatible_model(model_name)
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text using Ollama API."""
        # Use synchronous version for simplicity
        return self.generate_sync(prompt, temperature, max_tokens)
    
    def generate_sync(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Synchronous version of generate."""
        import requests
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                raise RuntimeError(f"Ollama API error: {response.text}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama connection error: {e}")


class PoEExpertSynthesizer:
    """Synthesizes Taichi-compatible expert functions using Ollama."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.client = OllamaClient(model_name)
        
    def extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
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
        """Validate expert code for Taichi compatibility and safety."""
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
        if re.search(r'math\.(sqrt|sin|cos|tan)', code):
            errors.append("Use ti.sqrt, ti.sin, ti.cos instead of Python math")
        
        return len(errors) == 0, errors
    
    async def synthesize_expert(self, description: str) -> Dict[str, Any]:
        """Synthesize a single expert from description."""
        
        # Log the synthesis request
        logger.info(f"Synthesizing expert for: '{description}'")
        
        prompt = f"""Generate a Taichi function that implements this particle behavior:
"{description}"

Rules:
1. Use @ti.func decorator
2. Function signature: def expert_NAME(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
3. Access particle state via tv.p.field[i].pos and tv.p.field[i].vel
4. Return a 2D force vector using ti.Vector([fx, fy])
5. Keep the implementation focused on ONE specific behavior
6. Use Taichi math functions (ti.sqrt, ti.sin, etc.) not Python math
7. Avoid division by zero with small epsilon values

Available particle properties:
- tv.p.field[i].pos - particle position (vec2)
- tv.p.field[i].vel - particle velocity (vec2)
- tv.p.field[i].mass - particle mass (float)
- tv.p.field[i].size - particle size (float)
- tv.p.field[i].species_id - particle type (int)
- tv.x, tv.y - screen dimensions
- tv.pn - total particle count

Example structure:
@ti.func
def expert_gravity(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Downward gravity force
    gravity_strength = 9.8
    return ti.Vector([0.0, gravity_strength])

Generate ONLY the function code, no explanations:"""

        # Log the prompt
        logger.debug(f"LLM Prompt:\n{prompt}")

        try:
            response = await self.client.generate(prompt, temperature=0.7)
            
            # Log raw LLM response
            logger.info(f"Raw LLM Response:\n{response}")
            
            code = self.extract_code(response)
            
            # Log extracted code
            logger.info(f"Extracted Code:\n{code}")
            
            # Validate code
            is_valid, errors = self.validate_expert_code(code)
            if not is_valid:
                logger.warning(f"Generated invalid code: {errors}")
                return {
                    "success": False,
                    "code": code,
                    "errors": errors,
                    "raw_response": response,
                    "prompt": prompt
                }
            
            # Extract function name
            name_match = re.search(r'def\s+(\w+)', code)
            if not name_match:
                return {
                    "success": False,
                    "code": code,
                    "errors": ["Could not extract function name"],
                    "raw_response": response,
                    "prompt": prompt
                }
            
            name = name_match.group(1)
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "description": description,
                "errors": [],
                "raw_response": response,
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Expert synthesis failed: {e}")
            return {
                "success": False,
                "code": "",
                "errors": [str(e)],
                "raw_response": "",
                "prompt": prompt if 'prompt' in locals() else ""
            }
    
    async def synthesize_integration_kernel(self, expert_info: list) -> dict:
        """Synthesize a Taichi kernel that integrates all active experts."""
        
        logger.info(f"Synthesizing integration kernel for {len(expert_info)} experts")
        
        # Build expert descriptions for the prompt
        expert_descriptions = []
        for expert in expert_info:
            expert_descriptions.append(f"  - {expert['name']}(tv, i) with weight {expert['weight']}")
        
        expert_list = "\n".join(expert_descriptions)
        
        prompt = f"""Generate a Taichi kernel that calls all these expert functions and applies their forces:

{expert_list}

Rules:
1. Use @ti.kernel decorator
2. Function signature: def apply_all_experts(tv: ti.template(), combined_forces: ti.template(), dt: ti.f32):
3. For each particle i in range(tv.pn):
   - Check if tv.p.field[i].active > 0
   - Call each expert function and accumulate forces with their weights
   - Apply accumulated force to tv.p.field[i].vel
   - Apply damping: tv.p.field[i].vel *= 0.98
   - Update position: tv.p.field[i].pos += tv.p.field[i].vel * dt
4. Clear combined_forces[i] at start of each particle
5. Use Taichi math functions (ti.Vector, etc.)

Example structure for reference:
@ti.kernel
def apply_all_experts(tv: ti.template(), combined_forces: ti.template(), dt: ti.f32):
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            combined_forces[i] = ti.Vector([0.0, 0.0])
            # Add forces from each expert...
            # Apply to particle...

Generate ONLY the kernel function code, no explanations:"""

        try:
            response = await self.client.generate(prompt, temperature=0.7)
            
            logger.info(f"Raw kernel response:\n{response}")
            
            code = self.extract_code(response)
            
            logger.info(f"Extracted kernel code:\n{code}")
            
            # Basic validation for kernel
            is_valid, errors = self._validate_kernel_code(code)
            if not is_valid:
                logger.warning(f"Generated invalid kernel: {errors}")
                return {
                    "success": False,
                    "code": code,
                    "errors": errors,
                    "raw_response": response,
                    "prompt": prompt
                }
            
            # Extract function name
            name_match = re.search(r'def\s+(\w+)', code)
            if not name_match:
                return {
                    "success": False,
                    "code": code,
                    "errors": ["Could not extract kernel function name"],
                    "raw_response": response,
                    "prompt": prompt
                }
            
            name = name_match.group(1)
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "errors": [],
                "raw_response": response,
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Kernel synthesis failed: {e}")
            return {
                "success": False,
                "code": "",
                "errors": [str(e)],
                "raw_response": "",
                "prompt": prompt if 'prompt' in locals() else ""
            }
    
    def _validate_kernel_code(self, code: str) -> tuple:
        """Validate kernel code for Taichi compatibility."""
        errors = []
        
        # Check for required structure
        if "@ti.kernel" not in code:
            errors.append("Missing @ti.kernel decorator")
        
        if "def " not in code:
            errors.append("Missing function definition")
        
        # Check for unsafe operations (same as expert validation)
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