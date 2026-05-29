"""
Model Factory for Multi-Provider LLM Support

This module provides a factory for creating pydantic-ai model instances for different LLM providers including OpenAI, Anthropic, Mistral, HuggingFace, Ollama, and Google Gemini.
"""

import os
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv


class ModelFactory:
    """Factory class for creating pydantic-ai model instances for different providers."""
    
    # Provider configurations: (env_var_name, module_path, class_name, needs_provider)
    PROVIDERS: Dict[str, Tuple[Optional[str], str, str, bool]] = {
        'gemini': ('GEMINI_API_KEY', 'pydantic_ai.models.gemini', 'GeminiModel', False),
        # 'google': ('GEMINI_API_KEY', 'pydantic_ai.models.google', 'GoogleModel', True),
        'openai': ('OPENAI_API_KEY', 'pydantic_ai.models.openai', 'OpenAIModel', False),
        'anthropic': ('ANTHROPIC_API_KEY', 'pydantic_ai.models.anthropic', 'AnthropicModel', False),
        'mistral': ('MISTRAL_API_KEY', 'pydantic_ai.models.mistral', 'MistralModel', False),
        'huggingface': ('HF_TOKEN', 'pydantic_ai.models.huggingface', 'HuggingFaceModel', False),
        'ollama': (None, 'pydantic_ai.models.openai', 'OpenAIModel', True),  # Uses OpenAI compatibility
        'bedrock': (None, 'pydantic_ai.models.bedrock', 'BedrockConverseModel', True),
    }
    
    # Default models for each provider
    DEFAULT_MODELS = {
        'gemini': 'gemini-2.0-flash',
        # 'google': 'gemini-2.0-flash',
        'openai': 'gpt-5-mini',
        'anthropic': 'claude-sonnet-4-0',
        'mistral': 'mistral-medium-2508',
        'huggingface': 'Qwen/QwQ-32B-Preview',
        'ollama': 'llama3.2',
        'bedrock': 'us.anthropic.claude-opus-4-8',
    }
    
    # Model name aliases for convenience
    MODEL_ALIASES = {
        # Gemini aliases
        'gemini-flash': 'gemini-2.0-flash',
        'gemini-pro': 'gemini-2.0-pro',
        # OpenAI aliases
        'gpt4': 'gpt-4o',
        'gpt-4': 'gpt-4o',
        'gpt-3.5': 'gpt-3.5-turbo',
        # Anthropic aliases
        'claude': 'claude-3-5-sonnet-latest',
        'claude-3': 'claude-3-5-sonnet-latest',
        'claude-opus': 'claude-3-opus-20240229',
        # Mistral aliases
        'mistral': 'mistral-large-latest',
        'mistral-small': 'mistral-small-latest',
        # Ollama aliases
        'llama': 'llama3.2',
        'llama3': 'llama3.2',
        'codellama': 'codellama',
        # Bedrock aliases (cross-region inference profiles)
        'bedrock-opus-4-8': 'us.anthropic.claude-opus-4-8',
        'bedrock-opus-4-7': 'us.anthropic.claude-opus-4-7',
        'bedrock-opus-4-6': 'us.anthropic.claude-opus-4-6-v1',
        'bedrock-opus-4-5': 'us.anthropic.claude-opus-4-5-20251101-v1:0',
        'bedrock-sonnet-4-6': 'us.anthropic.claude-sonnet-4-6',
        'bedrock-sonnet-4-5': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'bedrock-haiku-4-5': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
    }
    
    @classmethod
    def _load_env(cls):
        """Load environment variables from .env file if it exists."""
        env_paths = [
            Path.cwd() / ".env",
            Path(__file__).parent.parent.parent.parent.parent / ".env",
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                break
    
    @classmethod
    def parse_model_string(cls, model_string: str) -> Tuple[str, str]:
        """
        Parse a model string to extract provider and model name.
        
        Examples:
            "openai:gpt-4" -> ("openai", "gpt-4")
            "gpt-4" -> ("openai", "gpt-4")  # Uses default provider mapping
            "gemini-2.0-flash" -> ("gemini", "gemini-2.0-flash")
            "claude" -> ("anthropic", "claude-3-5-sonnet-latest")
        
        Args:
            model_string: Model specification string
            
        Returns:
            Tuple of (provider, model_name)
        """
        # Bedrock model IDs (us./global./eu.) embed ':' in version suffixes
        # like '-v1:0', so they must be detected before any provider:model split.
        if model_string.startswith(('us.', 'global.', 'eu.')):
            return 'bedrock', model_string

        # Check if it's in provider:model format
        if ':' in model_string:
            parts = model_string.split(':', 1)
            provider = parts[0].lower()
            model_name = parts[1]

            if provider in cls.PROVIDERS:
                # Resolve model alias if needed
                model_name = cls.MODEL_ALIASES.get(model_name, model_name)
                return provider, model_name
            # Not a real provider prefix — fall through (e.g. embedded ':' in a model id)
        
        # Check if it's a known alias
        model_lower = model_string.lower()
        if model_lower in cls.MODEL_ALIASES:
            resolved = cls.MODEL_ALIASES[model_lower]
            # Determine provider from resolved model name
            if model_lower.startswith('bedrock-') or resolved.startswith(('us.', 'global.', 'eu.')):
                return 'bedrock', resolved
            elif 'gemini' in resolved:
                return 'gemini', resolved
            elif 'gpt' in resolved:
                return 'openai', resolved
            elif 'claude' in resolved:
                return 'anthropic', resolved
            elif 'mistral' in resolved:
                return 'mistral', resolved
            elif 'llama' in model_lower or 'codellama' in model_lower:
                return 'ollama', resolved
            else:
                return 'gemini', resolved  # Default to gemini

        # Check for known model patterns
        if model_string.startswith(('us.', 'global.', 'eu.')):
            return 'bedrock', model_string
        elif 'gemini' in model_string:
            return 'gemini', model_string
        elif 'gpt' in model_string:
            return 'openai', model_string
        elif 'claude' in model_string:
            return 'anthropic', model_string
        elif 'mistral' in model_string:
            return 'mistral', model_string
        elif 'llama' in model_string.lower():
            return 'ollama', model_string
        elif '/' in model_string:  # HuggingFace format (org/model)
            return 'huggingface', model_string
        
        # Default to gemini for backward compatibility
        return 'gemini', model_string
    
    @classmethod
    def get_api_key(cls, provider: str, api_key: Optional[str] = None) -> Optional[str]:
        """
        Get API key for a provider from parameter or environment.
        
        Args:
            provider: Provider name
            api_key: Optional API key parameter
            
        Returns:
            API key string or None
        """
        if api_key:
            return api_key
        
        # Load environment if not already loaded
        cls._load_env()
        
        # Get the environment variable name for this provider
        config = cls.PROVIDERS.get(provider)
        if not config or not config[0]:
            return None
        
        env_var = config[0]
        key = os.getenv(env_var)
        
        if not key:
            # Try alternative environment variable names
            alternatives = {
                'GEMINI_API_KEY': ['GOOGLE_API_KEY', 'GOOGLE_GEMINI_API_KEY'],
                'ANTHROPIC_API_KEY': ['CLAUDE_API_KEY'],
                'HF_TOKEN': ['HUGGINGFACE_TOKEN', 'HUGGINGFACE_API_KEY'],
            }
            
            if env_var in alternatives:
                for alt in alternatives[env_var]:
                    key = os.getenv(alt)
                    if key:
                        break
        
        return key
    
    @classmethod
    def create_model(
        cls,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Create a pydantic-ai model instance for the specified provider.
        
        Args:
            model_name: Model specification (e.g., "openai:gpt-4", "claude", "gemini-2.0-flash")
            api_key: Optional API key (will use environment variable if not provided)
            **kwargs: Additional provider-specific configuration
            
        Returns:
            Configured pydantic-ai model instance
            
        Raises:
            ValueError: If provider is not supported or configuration fails
            ImportError: If required provider package is not installed
        """
        # Parse the model string to get provider and actual model name
        provider, actual_model = cls.parse_model_string(model_name)
        
        # Get provider configuration
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Provider '{provider}' not supported. "
                f"Available providers: {', '.join(cls.PROVIDERS.keys())}"
            )
        
        env_var, module_path, class_name, needs_provider = cls.PROVIDERS[provider]
        
        # Handle API key (ollama and bedrock use other auth)
        if provider not in ('ollama', 'bedrock'):
            api_key = cls.get_api_key(provider, api_key)
            if not api_key and env_var:
                raise ValueError(
                    f"No API key found for {provider}. "
                    f"Please set {env_var} environment variable or pass api_key parameter."
                )
        
        # Import the model class
        try:
            import importlib
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except ImportError as e:
            # Provide helpful error message for missing dependencies
            provider_packages = {
                'openai': 'pydantic-ai-slim[openai]',
                'anthropic': 'pydantic-ai-slim[anthropic]',
                'mistral': 'pydantic-ai-slim[mistral]',
                'huggingface': 'pydantic-ai-slim[huggingface]',
                'google': 'pydantic-ai-slim[google]',
            }
            package = provider_packages.get(provider, 'pydantic-ai')
            raise ImportError(
                f"Failed to import {provider} model. "
                f"Please install it with: pip install '{package}'"
            ) from e
        
        # Create model instance based on provider
        try:
            if provider == 'ollama':
                # Ollama uses OpenAI compatibility mode
                ollama_host = kwargs.pop('base_url', os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
                # Ensure proper URL format for OpenAI compatibility
                if not ollama_host.endswith('/v1'):
                    ollama_host = f"{ollama_host.rstrip('/')}/v1"
                
                # Import OpenAI classes for Ollama compatibility
                from pydantic_ai.models.openai import OpenAIModel
                from pydantic_ai.providers.openai import OpenAIProvider
                
                # Create OpenAI provider instance for Ollama
                provider_instance = OpenAIProvider(
                    base_url=ollama_host,
                    api_key='ollama',  # Ollama doesn't validate API keys
                    **kwargs
                )
                
                # Create OpenAI model with Ollama provider
                model = OpenAIModel(actual_model, provider=provider_instance)
                
            elif provider == 'bedrock':
                from pydantic_ai.providers.bedrock import BedrockProvider

                profile_name = kwargs.pop('profile_name', os.getenv('AWS_PROFILE'))
                region_name = kwargs.pop('region_name', os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1')

                provider_instance = BedrockProvider(
                    profile_name=profile_name,
                    region_name=region_name,
                    **kwargs,
                )
                model = model_class(actual_model, provider=provider_instance)

            elif provider == 'google':
                # Google provider needs special handling
                from pydantic_ai.providers.google import GoogleProvider
                
                # Check if using Vertex AI
                use_vertex = kwargs.pop('vertexai', os.getenv('USE_VERTEX_AI', 'false').lower() == 'true')
                location = kwargs.pop('location', os.getenv('VERTEX_AI_LOCATION', 'us-central1'))
                
                if use_vertex:
                    provider_instance = GoogleProvider(vertexai=True, location=location, **kwargs)
                else:
                    provider_instance = GoogleProvider(api_key=api_key, **kwargs)
                
                model = model_class(actual_model, provider=provider_instance)
                
            elif needs_provider:
                # For providers that require explicit provider instance
                provider_module = importlib.import_module(module_path.replace('.models.', '.providers.'))
                provider_class_name = f"{provider.capitalize()}Provider"
                provider_class = getattr(provider_module, provider_class_name)
                provider_instance = provider_class(api_key=api_key, **kwargs)
                model = model_class(actual_model, provider=provider_instance)
                
            else:
                # Standard model creation
                if api_key:
                    # Set the API key in environment for the model to use
                    if env_var:
                        os.environ[env_var] = api_key
                model = model_class(actual_model, **kwargs)
            
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to create {provider} model: {e}") from e
    
    @classmethod
    def list_available_providers(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available providers and their configuration status.
        
        Returns:
            Dictionary with provider information including API key status
        """
        cls._load_env()
        providers_info = {}
        
        for provider, config in cls.PROVIDERS.items():
            env_var, module_path, class_name, needs_provider = config

            # Check if API key is configured
            has_key = False
            if provider == 'ollama':
                has_key = True  # Ollama doesn't need an API key
            elif provider == 'bedrock':
                # Bedrock uses AWS credentials (env vars, ~/.aws/credentials,
                # SSO, instance profile, etc.). Ask boto3's default credential
                # chain — that's the same path BedrockProvider takes at runtime.
                try:
                    import boto3  # type: ignore
                    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE'))
                    has_key = session.get_credentials() is not None
                except Exception:
                    has_key = False
            elif env_var:
                has_key = bool(os.getenv(env_var))
            
            # Check if module is installed
            try:
                import importlib
                importlib.import_module(module_path)
                is_installed = True
            except ImportError:
                is_installed = False
            
            providers_info[provider] = {
                'api_key_configured': has_key,
                'api_key_env_var': env_var,
                'is_installed': is_installed,
                'default_model': cls.DEFAULT_MODELS.get(provider),
                'status': 'ready' if (has_key and is_installed) else 'not configured'
            }
        
        return providers_info
    
    @classmethod
    def get_provider_for_model(cls, model_name: str) -> str:
        """
        Get the provider name for a given model string.
        
        Args:
            model_name: Model specification string
            
        Returns:
            Provider name
        """
        provider, _ = cls.parse_model_string(model_name)
        return provider