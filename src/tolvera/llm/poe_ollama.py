
"""
Ollama integration for PoE expert synthesis.

This module provides the core Ollama client functionality for LLM interactions.
"""

import logging
import re
from typing import Optional, List, Dict
import ollama

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
                   messages: List[Dict[str,
                                       str]],
                   temperature: float = 0.7,
                   max_tokens: int = 30000, 
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


