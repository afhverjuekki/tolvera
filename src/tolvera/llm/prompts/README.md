# Tölvera LLM Prompts

This directory contains all LLM prompts used throughout the Tölvera system, extracted from Python source files for better maintainability and separation of concerns.

## Directory Structure

- **synthesis/** - Prompts for behavior synthesis and expert generation
- **decomposition/** - Prompts for complex behavior decomposition  
- **refinement/** - Prompts for sketch analysis and refinement (3 types)
- **drawing/** - Prompts for visual effects and drawing behaviors
- **utilities/** - Small utility prompts (color resolution, etc.)

## Loading

Prompts are loaded via the `PromptLoader` utility class in `src/tolvera/llm/core/prompt_loader.py`, which handles:
- File reading from this directory
- Variable substitution using Python string formatting
- Multi-part prompt assembly
- Error handling and fallbacks

## Variable Injection

Most prompts support variable injection using Python's `str.format()` syntax. Variables are passed as keyword arguments to the loader methods.

Example:
```python
loader = PromptLoader()
prompt = loader.load_prompt("synthesis/expert_synthesis_system.txt", 
                           available_states=states_str,
                           examples=examples_str)
```