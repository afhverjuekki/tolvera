
"""
Ollama integration for PoE expert synthesis.

This module provides the core Ollama client functionality for LLM interactions.
"""

import logging
import re
from typing import Optional, List, Dict, Any, Type
import ollama
from ollama import ListResponse, list as ollama_list
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OllamaModelManager:

    def __init__(self):
        self.compatible_models = [
            "qwen3:4b",
            "qwen2.5:3b", "qwen2.5:7b", "qwen2.5-coder:7b",
            "gemma2:2b", "gemma2:9b",
            "qwen2.5:3b", "llama3.2:1b",
            "mistral:7b", "mistral-nemo:12b"
        ]
        self.default_model = "qwen3:4b"
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
            response: ListResponse = ollama_list()
            return [model.model for model in response.models]
        except Exception as e:
            logger.error(f"Failed to list models from Ollama: {e}")
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
        logger.warning(f"Compatible models: {self.compatible_models}")
        logger.info(f"Pulling default model: {self.default_model}")

        if requested_model:
            logger.info(f"Using requested model anyway: {requested_model}")
            return requested_model
        
        logger.info(f"Using default model anyway: {self.default_model}")
        return self.default_model


class OllamaClient:

    def __init__(self, model_name: str = "qwen3:4b"):
        self.model_name = model_name
        self.model_manager = OllamaModelManager()

        if not self.model_manager.check_ollama_running():
            raise RuntimeError(
                "Ollama is not running. Please start it with 'ollama serve'")

        self.model_name = self.model_manager.ensure_compatible_model(
            model_name)
        self.client = ollama.AsyncClient(timeout=90)
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")

    async def chat(self,
                   messages: List[Dict[str, str]],
                   temperature: float = 0.7,
                   max_tokens: int = 50000, 
                   think: bool = False) -> str:
        try:
            # Append /no_think to avoid thinking mode issues
            messages_with_no_think = messages.copy()
            if messages_with_no_think:
                # Append /no_think to the last message's content
                messages_with_no_think[-1]['content'] += "\n\n/no_think"
            
            response = await self.client.chat(
                model=self.model_name,
                messages=messages_with_no_think,
                stream=False,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                },
                # think=think
            )
            
            # Remove <think> tags and their content
            content = response['message']['content']
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.strip()
            
            return content
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    async def chat_structured(self,
                            messages: List[Dict[str, str]],
                            response_format: Type[BaseModel],
                            temperature: float = 0.0,
                            max_tokens: int = 50000) -> BaseModel:
        """
        Chat with structured output using Pydantic models.
        
        Args:
            messages: List of chat messages
            response_format: Pydantic model class for structured output
            temperature: Temperature for generation (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Instance of the response_format model
        """
        try:
            # Add instruction to return JSON for better compliance
            messages_with_json_instruction = messages.copy()
            if messages_with_json_instruction:
                messages_with_json_instruction[-1]['content'] += "\n\nReturn as valid JSON matching the required format."
            
            response = await self.client.chat(
                model=self.model_name,
                messages=messages_with_json_instruction,
                stream=False,
                format=response_format.model_json_schema(),
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            
            content = response['message']['content']
            logger.debug(f"Structured response content: {content[:200]}...")
            
            # Parse the structured response
            try:
                return response_format.model_validate_json(content)
            except Exception as parse_error:
                logger.error(f"Failed to parse structured response: {parse_error}")
                logger.error(f"Raw content: {content}")
                raise ValueError(f"Failed to parse structured response: {parse_error}")
                
        except Exception as e:
            raise RuntimeError(f"Ollama structured API error: {e}")


