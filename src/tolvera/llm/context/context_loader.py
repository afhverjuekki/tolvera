"""
Central loader for all context files.
Provides consistent interface for loading text-based context documentation.
"""

from pathlib import Path
from typing import Dict, Optional

class ContextLoader:
    """Load context from text files with support for multi-section documents."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self._cache = {}
    
    def load_file(self, filename: str) -> str:
        """Load a complete text file."""
        if filename in self._cache:
            return self._cache[filename]
        
        path = self.base_path / filename
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self._cache[filename] = content
        return content
    
    def load_section(self, filename: str, section_name: str) -> str:
        """Load a specific section from a multi-section file."""
        content = self.load_file(filename)
        
        # Split on section markers
        sections = content.split('\n---\n')
        
        # If no sections, return whole content if section_name is 'main'
        if len(sections) == 1:
            return content if section_name == 'main' else ''
        
        # Parse sections
        for i in range(0, len(sections)):
            if i % 2 == 1:  # Section name
                if sections[i].strip() == section_name:
                    # Return the next section (content)
                    return sections[i + 1] if i + 1 < len(sections) else ''
            elif i == 0 and section_name == 'main':
                # First section before any markers
                return sections[0]
        
        return ''
    
    def get_all_sections(self, filename: str) -> Dict[str, str]:
        """Get all sections from a file as a dictionary."""
        content = self.load_file(filename)
        sections = content.split('\n---\n')
        
        if len(sections) == 1:
            return {'main': content}
        
        result = {}
        current_key = 'main'
        
        for i, section in enumerate(sections):
            if i % 2 == 0:  # Content
                if i == 0:
                    result['main'] = section
                else:
                    result[current_key] = section
            else:  # Key
                current_key = section.strip()
        
        return result

# Global instance
_loader = ContextLoader()

def load_context(filename: str) -> str:
    """Load a complete context file."""
    return _loader.load_file(filename)

def load_section(filename: str, section: str) -> str:
    """Load a specific section from a context file."""
    return _loader.load_section(filename, section)