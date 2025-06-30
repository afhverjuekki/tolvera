"""LLM-based expert synthesis for the PoE behavior system.

This module provides functionality to generate behavior experts from
natural language descriptions using large language models.
"""

import re
import logging
from typing import Optional, List, Dict, Any
import asyncio

from .poe_core import SimpleProgrammaticExpert
from .poe_ollama import PoEExpertSynthesizer

logger = logging.getLogger(__name__)


class ExpertSynthesizer:
    """Synthesizes behavior experts from natural language descriptions.
    
    This class uses LLMs to generate Taichi-compatible expert code
    from human-readable behavior descriptions.
    """
    
    def __init__(self, llm_model=None):
        """Initialize the synthesizer.
        
        Args:
            llm_model: LLM model instance (e.g., from Google Gemini)
        """
        self.model = llm_model
        self.synthesis_cache = {}
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> str:
        """Create the prompt template for expert generation."""
        return """Generate a Taichi function that implements this particle behavior:
"{description}"

Rules:
1. Function should calculate a force (fx, fy) for particle i
2. Use Tölvera's particle system: tv.p.field[i].pos, tv.p.field[i].vel
3. Return the force as a ti.Vector([fx, fy])
4. Keep it simple and focused on ONE aspect
5. Use descriptive function name based on the behavior
6. Include helpful comments

Available variables and functions:
- ti.Vector([x, y]): Create 2D vector
- ti.sqrt(x): Square root
- ti.sin(x), ti.cos(x): Trigonometric functions
- ti.atan2(y, x): Arctangent
- ti.random(): Random number 0-1
- ti.max(a, b), ti.min(a, b): Min/max functions

Example format:
@ti.func
def expert_behavior_name(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Access particle state
    pos = tv.p.field[i].pos
    vel = tv.p.field[i].vel
    
    # Calculate force for this behavior
    force = ti.Vector([0.0, 0.0])
    
    # Behavior implementation
    # ...
    
    return force

Generate ONLY the function, no explanations:"""
    
    async def synthesize_expert(self, description: str) -> Optional[SimpleProgrammaticExpert]:
        """Synthesize an expert from a natural language description.
        
        Args:
            description: Human-readable behavior description
            
        Returns:
            SimpleProgrammaticExpert if successful, None otherwise
        """
        # Check cache first
        if description in self.synthesis_cache:
            logger.debug(f"Using cached expert for: {description}")
            return self.synthesis_cache[description]
        
        if not self.model:
            logger.error("No LLM model configured")
            return None
            
        try:
            # Generate expert code
            prompt = self.prompt_template.format(description=description)
            response = await self._generate_with_llm(prompt)
            
            if not response:
                return None
                
            # Extract and clean code
            code = self._extract_code(response)
            if not code:
                logger.error("Failed to extract code from LLM response")
                return None
                
            # Extract function name
            name = self._extract_function_name(code)
            if not name:
                name = self._generate_expert_name(description)
                
            # Create expert
            expert = SimpleProgrammaticExpert(
                name=name,
                code=code,
                weight=1.0
            )
            
            # Add metadata
            expert.metadata["description"] = description
            expert.metadata["category"] = self._categorize_behavior(description)
            
            # Cache the result
            self.synthesis_cache[description] = expert
            
            logger.info(f"Synthesized expert: {name}")
            return expert
            
        except Exception as e:
            logger.error(f"Failed to synthesize expert: {e}")
            return None
    
    async def _generate_with_llm(self, prompt: str) -> Optional[str]:
        """Generate code using the LLM model.
        
        Args:
            prompt: Generation prompt
            
        Returns:
            Generated text or None
        """
        try:
            if hasattr(self.model, 'generate_content'):
                # Google Gemini style
                response = self.model.generate_content(prompt)
                return response.text
            elif hasattr(self.model, 'complete'):
                # OpenAI style
                response = await self.model.complete(prompt)
                return response
            else:
                logger.error("Unknown LLM model type")
                return None
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted code or None
        """
        # Try to find code blocks
        if "```python" in response:
            match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in response:
            match = re.search(r'```\n(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()
                
        # If no code blocks, try to find function definition
        if "@ti.func" in response or "def " in response:
            lines = response.split('\n')
            code_lines = []
            in_function = False
            
            for line in lines:
                if "@ti.func" in line or (line.strip().startswith("def ") and not in_function):
                    in_function = True
                    code_lines.append(line)
                elif in_function:
                    code_lines.append(line)
                    # Simple heuristic: stop at empty line after return
                    if line.strip().startswith("return") and len(code_lines) > 3:
                        # Include one more line if it's indented (part of return)
                        idx = lines.index(line)
                        if idx + 1 < len(lines) and lines[idx + 1].startswith("    "):
                            code_lines.append(lines[idx + 1])
                        break
                        
            if code_lines:
                return '\n'.join(code_lines)
                
        return None
    
    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract function name from code.
        
        Args:
            code: Python code
            
        Returns:
            Function name or None
        """
        match = re.search(r'def\s+(\w+)', code)
        if match:
            return match.group(1)
        return None
    
    def _generate_expert_name(self, description: str) -> str:
        """Generate expert name from description.
        
        Args:
            description: Behavior description
            
        Returns:
            Generated name
        """
        # Extract key words
        words = re.findall(r'\w+', description.lower())
        
        # Filter common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
                      'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'between', 'under', 'particle', 'particles'}
        
        keywords = [w for w in words if w not in stop_words][:3]
        
        if keywords:
            return f"expert_{'_'.join(keywords)}"
        else:
            return f"expert_{hash(description) % 10000}"
    
    def _categorize_behavior(self, description: str) -> str:
        """Categorize behavior based on description.
        
        Args:
            description: Behavior description
            
        Returns:
            Category name
        """
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['mouse', 'cursor', 'pointer']):
            return 'mouse_interaction'
        elif any(word in desc_lower for word in ['boundary', 'edge', 'wall', 'bound']):
            return 'boundary'
        elif any(word in desc_lower for word in ['flock', 'swarm', 'group', 'cohesion']):
            return 'flocking'
        elif any(word in desc_lower for word in ['avoid', 'repel', 'separate']):
            return 'separation'
        elif any(word in desc_lower for word in ['attract', 'pull', 'gravitate']):
            return 'attraction'
        elif any(word in desc_lower for word in ['random', 'noise', 'wander']):
            return 'noise'
        else:
            return 'custom'
    
    async def synthesize_multiple(
        self, 
        descriptions: List[str]
    ) -> List[Optional[SimpleProgrammaticExpert]]:
        """Synthesize multiple experts concurrently.
        
        Args:
            descriptions: List of behavior descriptions
            
        Returns:
            List of experts (None for failed syntheses)
        """
        tasks = [self.synthesize_expert(desc) for desc in descriptions]
        return await asyncio.gather(*tasks)
    
    def create_example_expert(self, behavior_type: str) -> Optional[SimpleProgrammaticExpert]:
        """Create an example expert of a given type.
        
        Args:
            behavior_type: Type of behavior
            
        Returns:
            Example expert or None
        """
        examples = {
            "mouse_attraction": """@ti.func
def expert_mouse_attraction(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Get particle position
    pos = tv.p.field[i].pos
    
    # Get mouse position (assumed to be in tv.s.behavior)
    mouse_pos = ti.Vector([
        tv.s.behavior.field[0].mouse_x,
        tv.s.behavior.field[0].mouse_y
    ])
    
    # Calculate attraction force
    to_mouse = mouse_pos - pos
    dist = to_mouse.norm()
    
    force = ti.Vector([0.0, 0.0])
    if 0 < dist < 200.0:  # Within influence radius
        # Inverse square law attraction
        force = to_mouse.normalized() * (10.0 / (dist * dist))
    
    return force""",
            
            "boundary_avoidance": """@ti.func
def expert_boundary_avoidance(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Get particle position
    pos = tv.p.field[i].pos
    
    # World boundaries (should be passed as context)
    world_width = 800.0  # Default
    world_height = 600.0  # Default
    margin = 50.0
    strength = 0.5
    
    force = ti.Vector([0.0, 0.0])
    
    # Left/right boundaries
    if pos[0] < margin:
        force[0] += strength * (margin - pos[0]) / margin
    elif pos[0] > world_width - margin:
        force[0] -= strength * (pos[0] - (world_width - margin)) / margin
    
    # Top/bottom boundaries
    if pos[1] < margin:
        force[1] += strength * (margin - pos[1]) / margin
    elif pos[1] > world_height - margin:
        force[1] -= strength * (pos[1] - (world_height - margin)) / margin
    
    return force""",
            
            "random_walk": """@ti.func
def expert_random_walk(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
    # Add random noise to movement
    noise_strength = 0.1
    
    # Generate random force
    angle = ti.random() * 2.0 * 3.14159
    force = ti.Vector([
        ti.cos(angle) * noise_strength,
        ti.sin(angle) * noise_strength
    ])
    
    return force"""
        }
        
        if behavior_type in examples:
            code = examples[behavior_type]
            name = self._extract_function_name(code)
            
            expert = SimpleProgrammaticExpert(
                name=name,
                code=code,
                weight=1.0
            )
            expert.metadata["description"] = f"Example {behavior_type} behavior"
            expert.metadata["category"] = behavior_type
            
            return expert
            
        return None


class SimpleSynthesizer(ExpertSynthesizer):
    """Synthesizer that can use both LLM and templates.
    
    This synthesizer primarily uses Ollama for generation but falls
    back to templates when LLM is unavailable.
    """
    
    def __init__(self, use_llm: bool = True, model_name: Optional[str] = None):
        """Initialize synthesizer.
        
        Args:
            use_llm: Whether to use LLM (Ollama) for synthesis
            model_name: Ollama model name to use
        """
        super().__init__(llm_model=None)
        self.use_llm = use_llm
        self.ollama_synthesizer = None
        
        if use_llm:
            try:
                self.ollama_synthesizer = PoEExpertSynthesizer(model_name)
                logger.info("Initialized with Ollama LLM synthesis")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama, falling back to templates: {e}")
                self.use_llm = False
        
        if not self.use_llm:
            logger.info("Using template-based synthesis")
            
        self.keyword_mapping = self._create_keyword_mapping()
    
    def _create_keyword_mapping(self) -> Dict[str, str]:
        """Create mapping from keywords to behavior types."""
        return {
            "mouse": "mouse_attraction",
            "cursor": "mouse_attraction",
            "attract": "mouse_attraction",
            "boundary": "boundary_avoidance",
            "edge": "boundary_avoidance",
            "wall": "boundary_avoidance",
            "random": "random_walk",
            "noise": "random_walk",
            "wander": "random_walk",
        }
    
    async def synthesize_expert(self, description: str) -> Optional[SimpleProgrammaticExpert]:
        """Synthesize expert using LLM or templates.
        
        Args:
            description: Behavior description
            
        Returns:
            Expert if successful, None otherwise
        """
        if self.use_llm and self.ollama_synthesizer:
            # Try LLM synthesis first
            try:
                result = await self.ollama_synthesizer.synthesize_expert(
                    description,
                    context={"width": 800, "height": 600, "particle_count": 1000}
                )
                
                if result["success"]:
                    expert = SimpleProgrammaticExpert(
                        name=result["name"],
                        code=result["code"],
                        weight=1.0
                    )
                    expert.metadata["description"] = description
                    expert.metadata["category"] = self._categorize_behavior(description)
                    logger.info(f"LLM synthesized expert: {expert.name}")
                    return expert
                else:
                    logger.warning(f"LLM synthesis failed: {result['errors']}")
                    
            except Exception as e:
                logger.error(f"Error in LLM synthesis: {e}")
        
        # Fall back to template matching
        return await self._synthesize_from_template(description)
    
    async def _synthesize_from_template(self, description: str) -> Optional[SimpleProgrammaticExpert]:
        """Synthesize expert by matching keywords to templates.
        
        Args:
            description: Behavior description
            
        Returns:
            Expert if matched, None otherwise
        """
        desc_lower = description.lower()
        
        # Find matching behavior type
        for keyword, behavior_type in self.keyword_mapping.items():
            if keyword in desc_lower:
                expert = self.create_example_expert(behavior_type)
                if expert:
                    # Customize the expert name based on description
                    expert.name = self._generate_expert_name(description)
                    expert.metadata["description"] = description
                    logger.info(f"Template matched '{description}' to {behavior_type}")
                    return expert
                    
        logger.warning(f"No template match for: {description}")
        return None
    
    async def synthesize_multiple(self, descriptions: List[str]) -> List[Optional[SimpleProgrammaticExpert]]:
        """Synthesize multiple experts from descriptions.
        
        Args:
            descriptions: List of behavior descriptions
            
        Returns:
            List of experts (None for failed syntheses)
        """
        if self.use_llm and self.ollama_synthesizer:
            # For complex descriptions, use the multi-expert synthesis
            if len(descriptions) == 1 and any(word in descriptions[0].lower() 
                                              for word in ['and', 'while', 'but', 'also']):
                try:
                    results = await self.ollama_synthesizer.synthesize_multiple_experts(
                        descriptions[0],
                        context={"width": 800, "height": 600, "particle_count": 1000}
                    )
                    
                    experts = []
                    for result in results:
                        if result["success"]:
                            expert = SimpleProgrammaticExpert(
                                name=result["name"],
                                code=result["code"],
                                weight=1.0
                            )
                            expert.metadata["description"] = result.get("description", descriptions[0])
                            expert.metadata["category"] = self._categorize_behavior(expert.metadata["description"])
                            experts.append(expert)
                    
                    if experts:
                        logger.info(f"LLM synthesized {len(experts)} experts from complex description")
                        return experts
                        
                except Exception as e:
                    logger.error(f"Error in multiple LLM synthesis: {e}")
        
        # Default: synthesize each description individually
        return await super().synthesize_multiple(descriptions)