"""
Behavior Decomposer for breaking down complex particle behavior descriptions into simpler, atomic behaviors that the synthesis engine can handle effectively.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .poe_ollama import OllamaClient

logger = logging.getLogger(__name__)

@dataclass
class SubBehavior:
    description: str
    weight: float
    relationship: str  # 'independent', 'simultaneous', 'conditional', 'sequential'
    priority: int = 0  # For ordering sequential behaviors


class BehaviorDecomposer:
    """Decomposes complex behavior descriptions into simpler atomic behaviors."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the decomposer with an LLM client."""
        self.client = OllamaClient(model_name or "qwen2.5:3b")
        
        # Patterns that indicate behaviors needing decomposition
        self.complexity_indicators = [
            r'\bwhile\b',           # "A while B"
            r'\band\b',             # "A and B"
            r'\bbut\b',             # "A but B"
            r'\bthen\b',            # "A then B"
            r',\s*(and|while)',     # Lists with conjunctions
            r'at the same time',    # Explicit simultaneity
            r'simultaneously',      # Explicit simultaneity
            r'multiple species',    # Multi-species interactions
            r'species \d+.*species \d+.*species \d+',  # 3+ species mentioned
        ]
        
    def check_complexity(self, description: str) -> Dict[str, bool]:
        desc_lower = description.lower()
        
        indicators = {
            'has_conjunction': any(re.search(pattern, desc_lower) for pattern in self.complexity_indicators),
            'multiple_behaviors': len(re.findall(r'(chase|flee|repel|attract|migrate|orbit|flock|hunt|protect)', desc_lower)) > 1,
            'conditional_behavior': any(word in desc_lower for word in ['while', 'but', 'except', 'unless']),
            'sequential_behavior': any(word in desc_lower for word in ['then', 'after', 'before', 'first']),
            'multi_species': len(re.findall(r'species \d+', desc_lower)) > 2,
            'compound_sentence': ',' in description or ';' in description,
            'balanced_forces': any(phrase in desc_lower for phrase in ['stronger than', 'weaker than', 'more than', 'less than'])
        }
        
        indicators['complexity_score'] = sum(1 for v in indicators.values() if v and isinstance(v, bool))
        
        return indicators
    
    async def should_decompose(self, description: str) -> Tuple[bool, str]:
        indicators = self.check_complexity(description)
        
        if indicators['complexity_score'] >= 2:
            return True, f"Multiple complexity indicators found (score: {indicators['complexity_score']})"
        
        if indicators['complexity_score'] == 1:
            prompt = f"""Analyze if this particle behavior description should be decomposed into simpler behaviors:

Description: "{description}"

Answer with YES or NO followed by a brief reason.

Consider:
- Does it describe multiple distinct behaviors?
- Are there conditional relationships (if/then, while)?
- Does it involve complex multi-species interactions?
- Would breaking it down make it clearer?

Format: YES/NO: reason"""

            try:
                messages = [
                    {'role': 'user', 'content': prompt}
                ]
                response = await self.client.chat(messages, temperature=0.1)
                response_lower = response.lower().strip()
                
                if response_lower.startswith('yes'):
                    return True, response.split(':', 1)[1].strip() if ':' in response else "LLM analysis suggests decomposition"
                else:
                    return False, response.split(':', 1)[1].strip() if ':' in response else "Simple enough for direct synthesis"
                    
            except Exception as e:
                logger.warning(f"LLM analysis failed, using heuristic: {e}")
                return False, "Unable to analyze, assuming simple behavior"
        
        return False, "No complexity indicators found"
    
    async def decompose_behavior(self, description: str) -> List[SubBehavior]:
        should_decompose, reason = await self.should_decompose(description)
        if not should_decompose:
            # Return as single behavior
            return [SubBehavior(description=description, weight=1.0, relationship='independent')]
        
        logger.info(f"Decomposing behavior: {description}")
        logger.info(f"Reason: {reason}")
        
        # Use LLM to decompose
        prompt = self._build_decomposition_prompt(description)
        
        try:
            messages = [
                {'role': 'user', 'content': prompt}
            ]
            response = await self.client.chat(messages, temperature=0.3)
            sub_behaviors = self._parse_decomposition_response(response, description)
            
            if not sub_behaviors:
                # Fallback if parsing fails
                logger.warning("Failed to parse decomposition, returning original")
                return [SubBehavior(description=description, weight=1.0, relationship='independent')]
            
            return sub_behaviors
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return [SubBehavior(description=description, weight=1.0, relationship='independent')]
    
    def _build_decomposition_prompt(self, description: str) -> str:
        return f"""Decompose this complex particle behavior into simpler, atomic behaviors:

Original: "{description}"

Break it down into simple behaviors that can be implemented independently. Each behavior should:
- Focus on ONE action or force
- Be clear and specific
- Include which species are involved

Output format (one per line):
BEHAVIOR: description | WEIGHT: 0.1-2.0 | RELATIONSHIP: independent/simultaneous/conditional

Examples:
Input: "species 0 chases species 1 while species 1 flees from species 0"
BEHAVIOR: species 0 chases species 1 | WEIGHT: 1.0 | RELATIONSHIP: simultaneous
BEHAVIOR: species 1 flees from species 0 | WEIGHT: 1.0 | RELATIONSHIP: simultaneous

Input: "particles migrate to the center but repel each other when too close"
BEHAVIOR: particles migrate to the center | WEIGHT: 1.5 | RELATIONSHIP: independent
BEHAVIOR: particles repel each other when close | WEIGHT: 1.0 | RELATIONSHIP: conditional

Now decompose the original behavior:"""
    
    def _parse_decomposition_response(self, response: str, original: str) -> List[SubBehavior]:
        sub_behaviors = []
        
        pattern = r'BEHAVIOR:\s*(.+?)\s*\|\s*WEIGHT:\s*([\d.]+)\s*\|\s*RELATIONSHIP:\s*(\w+)'
        matches = re.finditer(pattern, response, re.IGNORECASE)
        
        priority = 0
        for match in matches:
            desc = match.group(1).strip()
            try:
                weight = float(match.group(2))
                weight = max(0.1, min(2.0, weight))  # Clamp to valid range
            except ValueError:
                weight = 1.0
                
            relationship = match.group(3).lower()
            if relationship not in ['independent', 'simultaneous', 'conditional', 'sequential']:
                relationship = 'independent'
            
            sub_behaviors.append(SubBehavior(
                description=desc,
                weight=weight,
                relationship=relationship,
                priority=priority
            ))
            priority += 1
        
        # If no valid parsing, try simple splitting as fallback
        if not sub_behaviors:
            logger.warning("Failed to parse structured response, attempting simple split")
            sub_behaviors = self._simple_decompose(original)
        
        return sub_behaviors
    
    # Only if the LLM can't parse this.
    def _simple_decompose(self, description: str) -> List[SubBehavior]:
        desc_lower = description.lower()
        behaviors = []
        
        # Try to split on common conjunctions
        if ' while ' in desc_lower:
            parts = description.split(' while ', 1)
            behaviors.append(SubBehavior(parts[0].strip(), 1.0, 'simultaneous'))
            behaviors.append(SubBehavior(parts[1].strip(), 1.0, 'simultaneous'))
        elif ' and ' in desc_lower:
            parts = re.split(r',?\s+and\s+', description)
            for part in parts:
                behaviors.append(SubBehavior(part.strip(), 1.0, 'independent'))
        elif ' but ' in desc_lower:
            parts = description.split(' but ', 1)
            behaviors.append(SubBehavior(parts[0].strip(), 1.5, 'independent'))
            behaviors.append(SubBehavior(parts[1].strip(), 0.8, 'conditional'))
        elif ',' in description:
            # Handle comma-separated behaviors
            parts = [p.strip() for p in description.split(',')]
            for part in parts:
                if part:
                    behaviors.append(SubBehavior(part, 1.0, 'independent'))
        
        # If no decomposition happened, return original
        if not behaviors:
            behaviors = [SubBehavior(description, 1.0, 'independent')]
        
        return behaviors
    
    def adjust_weights_for_balance(self, sub_behaviors: List[SubBehavior], original_description: str) -> List[SubBehavior]:
        desc_lower = original_description.lower()
        
        if 'stronger than' in desc_lower or 'more than' in desc_lower:
            # Find what should be stronger
            for i, behavior in enumerate(sub_behaviors):
                if any(word in behavior.description.lower() for word in ['migrate', 'move', 'go']):
                    sub_behaviors[i].weight *= 1.5
                elif any(word in behavior.description.lower() for word in ['repel', 'avoid', 'flee']):
                    sub_behaviors[i].weight *= 0.7
        
        elif 'weaker than' in desc_lower or 'less than' in desc_lower:
            # Opposite adjustment
            for i, behavior in enumerate(sub_behaviors):
                if any(word in behavior.description.lower() for word in ['migrate', 'move', 'go']):
                    sub_behaviors[i].weight *= 0.7
                elif any(word in behavior.description.lower() for word in ['repel', 'avoid', 'flee']):
                    sub_behaviors[i].weight *= 1.5
        
        return sub_behaviors