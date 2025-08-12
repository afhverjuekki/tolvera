import os
import logging
from typing import Optional
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMClient:
    
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = self._init_client(model_name, api_key)
        
    def _init_client(self, model_name: str, api_key: Optional[str] = None) -> AsyncOpenAI:
        
        if "gemini" in model_name.lower():
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            logger.info(f"Initializing Gemini client for model: {model_name}")
            return AsyncOpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            
        elif "claude" in model_name.lower():
            raise NotImplementedError("Claude models not yet implemented")
            
        elif "gpt" in model_name.lower():
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            logger.info(f"Initializing OpenAI client for model: {model_name}")
            return AsyncOpenAI(api_key=api_key)
            
        else:
            logger.warning(f"Unknown model type: {model_name}, defaulting to Gemini configuration")
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY", "dummy-key-for-testing")
            
            return AsyncOpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
    
    async def generate_structured(
        self,
        prompt: str,
        response_model: BaseModel,
        temperature: float = 0.1,
        max_tokens: int = 30000,
        reasoning_effort: Optional[str] = None
    ) -> BaseModel:
        messages = [
            {"role": "system", "content": "You are an expert at generating Taichi code for particle simulations."},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": response_model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if reasoning_effort and "2.5" in self.model_name:
            kwargs["reasoning_effort"] = reasoning_effort
        
        try:
            completion = await self.client.beta.chat.completions.parse(**kwargs)
            return completion.choices[0].message.parsed
            
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 30000,
        reasoning_effort: Optional[str] = None
    ) -> str:
        messages = [
            {"role": "system", "content": "You are an expert at generating Taichi and Tölvera code for particle simulations."},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if reasoning_effort and "2.5" in self.model_name:
            kwargs["reasoning_effort"] = reasoning_effort
        
        try:
            completion = await self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    async def chat(self, messages: list, temperature: float = 0.1, **kwargs) -> str:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise