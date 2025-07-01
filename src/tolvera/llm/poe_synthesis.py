"""
Pure LLM-based expert synthesis for the PoE behavior system.

This module provides functionality to generate behavior experts from
natural language descriptions using large language models without any
templates or fallback mechanisms.
"""

import re
import logging
from typing import Optional
import asyncio
import time

from .poe_core import SimpleProgrammaticExpert
from .poe_ollama import PoEExpertSynthesizer
from .poe_logger import get_logger

logger = logging.getLogger(__name__)
csv_logger = get_logger()


class PureLLMSynthesizer:
    """
    Synthesizes behavior experts from natural language descriptions using only LLM.
    
    No templates, no fallbacks - just raw LLM generation with extensive logging.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize synthesizer with Ollama model."""
        self.synthesizer = PoEExpertSynthesizer(model_name)
        logger.info(f"Initialized PureLLMSynthesizer with model: {model_name or 'default'}")
    
    async def synthesize_expert(self, description: str, weight: float = 1.0) -> SimpleProgrammaticExpert:
        """
        Synthesize an expert from natural language description.
        
        This method shows the entire process with full logging:
        1. The input description
        2. The LLM prompt
        3. The raw LLM response
        4. The extracted code
        5. Any errors or issues
        
        Args:
            description: Human-readable behavior description
            weight: Expert weight for PoE combination
            
        Returns:
            SimpleProgrammaticExpert if successful
            
        Raises:
            ValueError if synthesis fails
        """
        
        logger.info(f"Synthesizing expert for: '{description}'")
        
        # Start timing
        start_time = time.time()
        
        # Call the Ollama synthesizer
        result = await self.synthesizer.synthesize_expert(description)
        
        # Calculate synthesis time
        synthesis_time_ms = (time.time() - start_time) * 1000
        
        # Log the raw response
        logger.debug(f"Raw LLM response: {result.get('raw_response', 'No response')}")
        
        # Log the extracted code
        logger.debug(f"Extracted code: {result.get('code', 'No code extracted')}")
        
        # Log to CSV
        csv_logger.log_synthesis_attempt(
            user_description=description,
            llm_prompt=result.get("prompt", ""),
            raw_response=result.get("raw_response", ""),
            extracted_code=result.get("code", ""),
            success=result["success"],
            errors=result.get("errors", []),
            model_name=self.synthesizer.client.model_name,
            expert_name=result.get("name", None),
            synthesis_time_ms=synthesis_time_ms
        )
        
        if result["success"]:
            # Create expert
            expert = SimpleProgrammaticExpert(
                name=result["name"],
                code=result["code"],
                weight=weight
            )
            expert.metadata["description"] = description
            expert.metadata["synthesis_method"] = "pure_llm"
            expert.metadata["raw_llm_response"] = result.get("raw_response", "")
            
            logger.info(f"Successfully created expert '{result['name']}'")
            
            return expert
        else:
            # Log errors
            logger.error(f"Synthesis failed: {result.get('errors', ['Unknown error'])}")
            
            raise ValueError(f"Failed to synthesize expert: {result.get('errors', ['Unknown error'])}")
    
    


async def create_expert_from_description(description: str, model_name: Optional[str] = None, weight: float = 1.0) -> SimpleProgrammaticExpert:
    """
    Convenience function to create an expert from a description.
    
    Args:
        description: Natural language behavior description
        model_name: Optional Ollama model name
        weight: Expert weight
        
    Returns:
        SimpleProgrammaticExpert
        
    Raises:
        ValueError if synthesis fails
    """
    synthesizer = PureLLMSynthesizer(model_name)
    return await synthesizer.synthesize_expert(description, weight)


def create_expert_from_description_sync(description: str, model_name: Optional[str] = None, weight: float = 1.0) -> SimpleProgrammaticExpert:
    """Synchronous version of create_expert_from_description."""
    return asyncio.run(create_expert_from_description(description, model_name, weight))