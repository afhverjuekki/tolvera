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
            "llama3.2:3b", "qwen2.5:7b", "qwen2.5-coder:7b",
            "gemma2:2b", "gemma2:9b",
            "llama3.2:3b", "llama3.2:1b",
            "mistral:7b", "mistral-nemo:12b"
        ]
        self.default_model = "llama3.2:3b"
        
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
    
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
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
        if re.search(r'(?<!ti\.)math\.(sqrt|sin|cos|tan)', code):
            errors.append("Use ti.sqrt, ti.sin, ti.cos instead of Python math")
        
        return len(errors) == 0, errors
    
    async def synthesize_expert(self, description: str) -> Dict[str, Any]:
        """Synthesize a single expert from description."""
        
        # Log the synthesis request
        logger.info(f"Synthesizing expert for: '{description}'")
        
        prompt = f"""You are an expert Taichi programmer. Generate a single, complete Taichi function that implements a specific particle behavior.

Behavior description: "{description}"

Follow these rules precisely:
1.  The function must have the decorator `@ti.func`.
2.  The function signature must be `def expert_NAME(tv: ti.template(), i: ti.i32) -> ti.math.vec2:`, where `NAME` is a descriptive name based on the behavior (e.g., `gravity`, `attract_to_center`).
3.  The function must return a 2D force vector: `ti.Vector([fx, fy])`. This vector will be added to the particle's velocity.
4.  Access particle data using `tv.p.field[i]`. For example, `tv.p.field[i].pos` for position.
5.  Use Taichi math functions like `ti.sqrt`, `ti.sin`, `ti.math.normalize`, etc. Do NOT use `math.sqrt`.
6.  For type casting, use `ti.cast(value, ti.f32)`. Do NOT use `ti.f32(value)`.
7.  **CRITICAL**: Do NOT use `return` inside of loops (`for`, `while`) or conditional blocks (`if`). Calculate forces and store them in variables, then return the final combined force at the end of the function.
8.  Keep the function focused on a single, specific behavior as described. Do not add unrelated logic.
9.  **CRITICAL**: All variables must be defined with a value before they are used.

Available Tolvera properties:
- `tv.p.field[i].pos`: vec2, particle position
- `tv.p.field[i].vel`: vec2, particle velocity
- `tv.p.field[i].mass`: f32, particle mass
- `tv.p.field[i].size`: f32, particle size
- `tv.x`, `tv.y`: f32, screen dimensions
- `tv.pn`: i32, total particle count

---
Good Example 1: Gravity
@ti.func
def expert_gravity(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # A simple, constant downward force.
    gravity_strength = 0.05
    return ti.Vector([0.0, gravity_strength])
---
Good Example 2: Attraction to center
@ti.func
def expert_attract_to_center(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Force that pulls particle towards the screen center.
    center = ti.Vector([tv.x / 2, tv.y / 2])
    direction = center - tv.p.field[i].pos
    # Normalize to get direction, handle zero-vector case.
    direction_norm = ti.math.normalize(direction)
    strength = 0.01
    return direction_norm * strength
---
Good Example 3: Repulsion from other particles
@ti.func
def expert_repel_others(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Pushes particle away from nearby particles.
    repulsion_force = ti.Vector([0.0, 0.0])
    my_pos = tv.p.field[i].pos
    repulsion_radius = 25.0
    strength = 0.02
    # This loop is OK because the return is outside of it.
    for j in range(tv.pn):
        if i != j:
            other_pos = tv.p.field[j].pos
            dist_vec = my_pos - other_pos
            dist_mag = ti.math.length(dist_vec)
            if dist_mag > 0 and dist_mag < repulsion_radius:
                # Force is inversely proportional to distance
                repulsion_force += ti.math.normalize(dist_vec) / dist_mag * strength
    return repulsion_force
---
Good Example 4: Clockwise spiral motion
@ti.func
def expert_spiral_clockwise(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Creates a spiral force by combining an inward pull with a tangential push.
    center = ti.Vector([tv.x / 2, tv.y / 2])
    to_center = center - tv.p.field[i].pos
    
    # Tangential force is perpendicular to the vector to the center.
    # For clockwise, the perpendicular of (x, y) is (y, -x).
    tangential_force = ti.Vector([to_center.y, -to_center.x])
    
    # Normalize forces to get pure direction
    tangential_norm = ti.math.normalize(tangential_force)
    inward_norm = ti.math.normalize(to_center)
    
    # Combine the forces. Adjust strengths to change spiral shape.
    final_force = (tangential_norm * 0.02) + (inward_norm * 0.01)
    return final_force
---

Now, generate the Taichi function for the description: "{description}".
Generate ONLY the complete function code, starting with `@ti.func`. Do not include any explanations or markdown.
"""''

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
        expert_calls = []
        for expert in expert_info:
            expert_calls.append(f"            total_force += {expert['name']}(tv, i) * {expert['weight']:.2f}")
        expert_calls_str = "\n".join(expert_calls)

        prompt = f"""You are a Taichi programmer. Your task is to generate a complete, precise Taichi kernel to integrate particle forces. Do not add any logic or functions that are not in the template.

Use this exact template:

@ti.kernel
def apply_all_experts(tv: ti.template(), combined_forces: ti.template(), dt: ti.f32):
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            # Initialise a total force for the particle
            total_force = ti.Vector([0.0, 0.0])

            # Accumulate forces from all experts
{expert_calls_str}

            # Apply the final combined force to the particle
            tv.p.field[i].vel += total_force * dt
            # Apply simple damping to prevent runaway speeds
            tv.p.field[i].vel *= 0.98
            # Update particle position based on new velocity
            tv.p.field[i].pos += tv.p.field[i].vel * dt

Generate ONLY the complete kernel function code as defined in the template. Do not add any other text, explanations, or markdown formatting.
"""

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