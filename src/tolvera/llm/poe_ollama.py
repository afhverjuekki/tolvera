"""
Ollama integration for PoE expert synthesis.

This module provides the LLM integration for generating Taichi-compatible
expert functions from natural language descriptions.
"""

import logging
import re
import asyncio
from typing import Optional, List, Dict, Any, Tuple
import subprocess
import json

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
    
    async def synthesize_expert(self, description: str, context: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Synthesize a single expert from description."""
        
        prompt = f"""Generate a Taichi kernel function that implements this particle behavior:
"{description}"

Context:
- Window dimensions: {context.get('width', 800)}x{context.get('height', 600)}
- Number of particles: {context.get('particle_count', 1000)}

Rules:
1. Use @ti.func decorator
2. Function signature: def expert_NAME(particles: ti.template(), i: ti.i32, context: ti.template()) -> ti.math.vec2:
3. Access particle state via particles[i].pos and particles[i].vel
4. Return a 2D force vector using ti.Vector([fx, fy])
5. Keep the implementation focused on ONE specific behavior
6. Use Taichi math functions (ti.sqrt, ti.sin, etc.) not Python math
7. Avoid division by zero with small epsilon values

Example structure:
@ti.func
def expert_attraction_to_center(particles: ti.template(), i: ti.i32, context: ti.template()) -> ti.math.vec2:
    pos = particles[i].pos
    center = ti.Vector([context[0], context[1]])  # Window center
    
    to_center = center - pos
    dist = to_center.norm()
    
    if dist > 1.0:  # Avoid singularity
        force = to_center.normalized() * (100.0 / dist)
        return force
    
    return ti.Vector([0.0, 0.0])

Generate ONLY the function code, no explanations:"""

        try:
            response = await self.client.generate(prompt, temperature=0.7)
            code = self.extract_code(response)
            
            # Validate code
            is_valid, errors = self.validate_expert_code(code)
            if not is_valid:
                logger.warning(f"Generated invalid code: {errors}")
                return {
                    "success": False,
                    "code": code,
                    "errors": errors
                }
            
            # Extract function name
            name_match = re.search(r'def\s+(\w+)', code)
            if not name_match:
                return {
                    "success": False,
                    "code": code,
                    "errors": ["Could not extract function name"]
                }
            
            name = name_match.group(1)
            
            return {
                "success": True,
                "name": name,
                "code": code,
                "description": description,
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Expert synthesis failed: {e}")
            return {
                "success": False,
                "code": "",
                "errors": [str(e)]
            }
    
    async def synthesize_multiple_experts(self, description: str, context: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """Synthesize multiple experts from a complex description."""
        
        # First, analyze the description to identify sub-behaviors
        analysis_prompt = f"""Analyze this particle behavior description and identify the individual force components:
"{description}"

Break it down into simple, independent behaviors. For each behavior, provide:
1. A short name (e.g., "mouse_attraction", "edge_avoidance")
2. A one-line description

Output as a JSON list:
[
    {{"name": "behavior_name", "description": "what it does"}},
    ...
]

Focus on atomic behaviors that can be combined. Output ONLY the JSON:"""

        try:
            response = await self.client.generate(analysis_prompt, temperature=0.3)
            
            # Extract JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                # Fallback to single expert
                logger.warning("Could not parse behavior breakdown, creating single expert")
                expert = await self.synthesize_expert(description, context)
                return [expert] if expert["success"] else []
            
            behaviors = json.loads(json_match.group(0))
            
            # Synthesize each behavior
            experts = []
            for behavior in behaviors[:5]:  # Limit to 5 experts
                expert = await self.synthesize_expert(
                    behavior["description"], 
                    context
                )
                if expert["success"]:
                    # Override name with analyzed name
                    expert["name"] = f"expert_{behavior['name']}"
                    experts.append(expert)
            
            return experts
            
        except Exception as e:
            logger.error(f"Multiple expert synthesis failed: {e}")
            # Fallback to single expert
            expert = await self.synthesize_expert(description, context)
            return [expert] if expert["success"] else []
    
    def synthesize_expert_sync(self, description: str, context: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Synchronous version of synthesize_expert."""
        return asyncio.run(self.synthesize_expert(description, context))
    
    def synthesize_multiple_experts_sync(self, description: str, context: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """Synchronous version of synthesize_multiple_experts."""
        return asyncio.run(self.synthesize_multiple_experts(description, context))


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_synthesis():
        synthesizer = PoEExpertSynthesizer()
        
        # Test single expert
        print("Testing single expert synthesis...")
        result = await synthesizer.synthesize_expert(
            "particles should be attracted to the mouse position"
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Name: {result['name']}")
            print(f"Code:\n{result['code']}")
        else:
            print(f"Errors: {result['errors']}")
        
        print("\n" + "="*50 + "\n")
        
        # Test multiple experts
        print("Testing multiple expert synthesis...")
        results = await synthesizer.synthesize_multiple_experts(
            "particles should swarm towards the mouse but avoid getting too close, "
            "while also avoiding the edges of the screen"
        )
        print(f"Generated {len(results)} experts:")
        for i, expert in enumerate(results):
            if expert['success']:
                print(f"\n{i+1}. {expert['name']}:")
                print(f"   Description: {expert['description']}")
                print(f"   Code preview: {expert['code'][:100]}...")
            else:
                print(f"\n{i+1}. Failed: {expert['errors']}")
    
    # Run test
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_synthesis())