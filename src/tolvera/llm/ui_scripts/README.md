# UI & Demo Scripts

This directory contains interactive demonstration scripts and user interfaces for the Tolvera LLM behavior synthesis system. These tools provide both command-line and terminal-based interfaces for generating particle-based behaviors from natural language descriptions.

## Available Scripts

### tolvera_llm_demo.py

Command-line demonstration script showcasing the full range of behavior synthesis capabilities including basic behaviors, complex multi-component behaviors, visual effects, species interactions, and artificial life patterns.

**Description:** Comprehensive demo that runs through multiple synthesis examples, generates complete sketches, and produces detailed debug traces with interactive HTML reports.

**Usage:**
```bash
poetry run python src/tolvera/llm/ui_scripts/tolvera_llm_demo.py
```

Optional debug flags:
- `--debug`: Enable detailed logging output
- `--prompt-debug`: Show full LLM prompts and responses

### tolvera_textual_ui.py

Interactive terminal user interface providing a guided experience for behavior synthesis with real-time code generation, model selection, and sketch management.

**Description:** Full-featured TUI application with chat panel for behavior input, code editor with syntax highlighting, integrated help system, tutorial mode, and support for saving/loading sketches.

**Usage:**
```bash
poetry run python src/tolvera/llm/ui_scripts/tolvera_textual_ui.py
```

## Directory Structure

### textual_components/

Supporting components for the Textual UI application:
- **dialogs.py**: Modal dialogs for save/load operations and help display
- **enhanced_panels.py**: Enhanced code editor and chat panel widgets

## Requirements

Both scripts require:
- Configured API keys in `.env` file (see main README for setup)
- Poetry environment with all dependencies installed
- Tolvera and Taichi properly configured

## Output

Generated sketches and traces are saved to:
- `examples/generated_sketches/`: Complete Python sketch files
- `examples/generated_sketches/traces/`: Debug traces and reports
  - `.json`: Raw trace data
  - `.html`: Interactive HTML reports
  - `.md`: Mermaid flow diagrams