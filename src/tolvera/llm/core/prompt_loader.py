"""
Utility for loading LLM prompts from external text files.
Provides centralized prompt management with variable substitution.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loads and manages LLM prompts from external text files."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the prompt loader.
        
        Args:
            base_path: Base directory for prompt files. If None, uses default.
        """
        if base_path is None:
            # Default to prompts directory relative to this file
            self.base_path = Path(__file__).parent.parent / "prompts"
        else:
            self.base_path = Path(base_path)
        
        logger.debug(f"PromptLoader initialized with base path: {self.base_path}")
    
    def load_prompt(self, file_path: str, **kwargs) -> str:
        """
        Load a single prompt from a file with variable substitution.
        
        Args:
            file_path: Relative path to the prompt file from base_path
            **kwargs: Variables for string formatting
            
        Returns:
            The loaded and formatted prompt content
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            ValueError: If string formatting fails
        """
        full_path = self.base_path / file_path
        
        logger.info(f"[PROMPT_LOADER] Loading prompt file: {full_path}")
        logger.debug(f"[PROMPT_LOADER] Substitution variables provided: {list(kwargs.keys())}")
        
        try:
            # Read the file content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            raw_length = len(content)
            logger.info(f"[PROMPT_LOADER] Raw content loaded: {raw_length} chars from {file_path}")
            
            # Check for placeholders in the content
            import re
            placeholders = re.findall(r'\{(\w+)\}', content)
            if placeholders:
                logger.debug(f"[PROMPT_LOADER] Found placeholders: {placeholders}")
            
            # Apply variable substitution if kwargs provided
            if kwargs:
                logger.debug(f"[PROMPT_LOADER] Applying variable substitution...")
                content = content.format(**kwargs)
                formatted_length = len(content)
                logger.info(f"[PROMPT_LOADER] Formatted content: {formatted_length} chars (delta: {formatted_length - raw_length})")
            else:
                logger.debug(f"[PROMPT_LOADER] No variable substitution needed")
            
            # Check if any placeholders remain
            remaining_placeholders = re.findall(r'\{(\w+)\}', content)
            if remaining_placeholders:
                logger.warning(f"[PROMPT_LOADER] Unsubstituted placeholders remain: {remaining_placeholders}")
            
            logger.info(f"[PROMPT_LOADER] Successfully loaded {file_path}: {len(content)} chars")
            return content
            
        except FileNotFoundError:
            logger.error(f"[PROMPT_LOADER] ERROR: Prompt file not found: {full_path}")
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        except KeyError as e:
            logger.error(f"[PROMPT_LOADER] ERROR: Missing variable in prompt {file_path}: {e}")
            logger.error(f"[PROMPT_LOADER] Required variable: {e}, Provided: {list(kwargs.keys())}")
            raise ValueError(f"Missing variable in prompt {file_path}: {e}")
        except Exception as e:
            logger.error(f"[PROMPT_LOADER] ERROR: Failed loading prompt {file_path}: {e}")
            logger.error(f"[PROMPT_LOADER] Exception type: {type(e).__name__}")
            raise
    
    def load_multi_part_prompt(self, parts: List[str], separator: str = "\n\n", **kwargs) -> str:
        """
        Load and combine multiple prompt parts into a single prompt.
        
        Args:
            parts: List of relative file paths to prompt parts
            separator: String to join the parts with
            **kwargs: Variables for string formatting
            
        Returns:
            The combined and formatted prompt content
        """
        logger.info(f"[PROMPT_LOADER] Loading multi-part prompt with {len(parts)} parts")
        logger.debug(f"[PROMPT_LOADER] Parts to load: {parts}")
        
        prompt_parts = []
        part_sizes = []
        
        for i, part_path in enumerate(parts):
            try:
                logger.debug(f"[PROMPT_LOADER] Loading part {i+1}/{len(parts)}: {part_path}")
                part_content = self.load_prompt(part_path, **kwargs)
                prompt_parts.append(part_content)
                part_sizes.append(len(part_content))
                logger.info(f"[PROMPT_LOADER] Part {i+1} loaded: {len(part_content)} chars")
            except Exception as e:
                logger.warning(f"[PROMPT_LOADER] WARNING: Failed to load prompt part {part_path}: {e}")
                continue
        
        combined = separator.join(prompt_parts)
        logger.info(f"[PROMPT_LOADER] Combined {len(prompt_parts)}/{len(parts)} parts successfully")
        logger.info(f"[PROMPT_LOADER] Total combined size: {len(combined)} chars")
        logger.debug(f"[PROMPT_LOADER] Part sizes: {part_sizes}")
        
        return combined
    
    def prompt_exists(self, file_path: str) -> bool:
        """
        Check if a prompt file exists.
        
        Args:
            file_path: Relative path to the prompt file
            
        Returns:
            True if the file exists, False otherwise
        """
        full_path = self.base_path / file_path
        return full_path.exists()
    
    def list_prompts(self, subdirectory: Optional[str] = None) -> List[str]:
        """
        List all available prompt files.
        
        Args:
            subdirectory: Optional subdirectory to search in
            
        Returns:
            List of relative paths to prompt files
        """
        search_path = self.base_path
        if subdirectory:
            search_path = search_path / subdirectory
        
        if not search_path.exists():
            return []
        
        prompt_files = []
        for file_path in search_path.rglob("*.txt"):
            relative_path = file_path.relative_to(self.base_path)
            prompt_files.append(str(relative_path))
        
        return sorted(prompt_files)


# Global instance for convenience
_default_loader = None

def get_prompt_loader() -> PromptLoader:
    """Get the default global prompt loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader