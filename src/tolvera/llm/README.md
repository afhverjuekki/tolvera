# src/llm Module Documentation

## Introduction

The src/llm module contains the core LLM-powered code generation engine for Tölvera. This module enables the synthesis of particle-based behaviors and visual effects from natural language descriptions, changing user intent into executable Taichi GPU kernels and functions. The system leverages foundation language models to generate physics simulations, particle interactions, and emergent behaviors for artificial life simulations.

## Architectural Overview

The system follows a multi-stage orchestrated pipeline that processes natural language through specialized components:

1. **Analysis Phase**: A user's natural language description is analyzed by the BehaviorAnalyzer, which identifies species, required states, and decomposes complex behaviors into implementable components.

2. **Orchestration Phase**: The BehaviorOrchestrator coordinates the synthesis process, managing state creation, component generation, and agent registration.

3. **Code Generation Phase**: The CodeGenerator synthesizes Taichi kernel functions based on the analyzed components, using dynamic context selection for code generation.

4. **Template Rendering Phase**: The TemplateRenderer assembles the generated code into complete, executable sketches using Jinja2 templates.

5. **Refinement Phase**: The SketchRefiner applies architectural patterns and corrections to ensure the generated code follows best practices.

## Key Components

### Core Orchestration and Synthesis

| Component                | File                            | Responsibility                                                                                                                                     |
| ------------------------ | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BehaviorOrchestrator** | `core/behavior_orchestrator.py` | Central coordinator that manages the entire synthesis pipeline, delegating tasks to specialized components and maintaining the workflow state      |
| **CodeGenerator**        | `core/code_generator.py`        | Generates Taichi expert functions from natural language descriptions using structured LLM outputs, handles state analysis and synthesis delegation |
| **BehaviorAnalyzer**     | `core/behavior_analyzer.py`     | Decomposes complex behavior descriptions into implementable components, identifies species configurations and required states                      |
| **SketchRefiner**        | `core/sketch_refiner.py`        | Applies architectural patterns and refinements to generated sketches, ensuring code quality and pattern consistency                                |

### Data Models and Registry

| Component            | File                        | Responsibility                                                                                                                         |
| -------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **DataModels**       | `core/data_models.py`       | Pydantic models defining the structure for behavior synthesis requests/responses, expert functions, states, and species configurations |
| **BehaviorRegistry** | `core/behavior_registry.py` | Maintains a registry of synthesized expert functions, tracks their types, weights, and species associations for kernel generation      |

### State and Species Management

| Component          | File                      | Responsibility                                                                                                                 |
| ------------------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **StateManager**   | `core/state_manager.py`   | Manages custom particle and global states, tracks available fields, generates initialization code, and detects required states |
| **SpeciesManager** | `core/species_manager.py` | Analyzes descriptions for species mentions, manages configurations and mappings, generates species-aware initialization code   |

### Context and Prompt Management

| Component           | File                          | Responsibility                                                                                                                       |
| ------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **ContextSelector** | `context/context_selector.py` | LLM-powered intelligent context selector that dynamically chooses relevant documentation and patterns based on behavior descriptions |
| **PromptLoader**    | `prompts/prompt_loader.py`    | Centralized prompt management with variable substitution, integrates with context selector for dynamic prompt building               |

### Template and Rendering

| Component            | File                             | Responsibility                                                                                                                  |
| -------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **TemplateRenderer** | `templates/template_renderer.py` | Centralized Jinja2 template rendering for code generation, produces integration kernels, drawing kernels, and complete sketches |

### Additional Components

| Component               | File                           | Responsibility                                                                                   |
| ----------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------ |
| **ColorResolver**       | `core/color_resolver.py`       | Maps semantic color descriptions to RGBA values, handles species-specific color assignment       |
| **ConversationManager** | `core/conversation_manager.py` | Manages conversation state and history for interactive sessions                                  |
| **LLMFactory**          | `core/llm_factory.py`          | Factory pattern for creating model instances across different providers (Gemini, Claude, OpenAI) |

## Directory Structure

### prompts/

Contains structured prompt templates organized by functionality:

- **decomposition/**: Templates for behavior analysis and decomposition
- **synthesis/**: Templates for code generation and expert synthesis
- **refinement/**: Templates for code refinement and error correction
- **context_selector/**: Templates for intelligent context selection
- **utilities/**: Helper templates for specific tasks

Each directory contains `.txt` files with templated prompts that support variable substitution.

### context/

Provides domain knowledge and patterns for code generation:

- **library_docs.py**: Core API documentation and usage patterns
- **patterns.py**: Common particle behavior patterns (flocking, cellular automata, etc.)
- **taichi_patterns.py**: Taichi-specific programming patterns and best practices
- **alife_patterns.py**: Artificial life simulation patterns
- **drawing_patterns.py**: Visual effect and rendering patterns
- **vera_patterns.py**: Tölvera-specific implementation patterns
- **temporal_dynamics.py**: Time-based behavior patterns
- **exemplars/**: Complete working examples (boids, slime mold, particle life)

### templates/

Jinja2 templates for code generation, organized by component type:

- **kernel/**: Integration, drawing, and utility kernel templates
- **sketch/**: Complete sketch file templates
- **state/**: State initialization and temporal update templates
- **init/**: Particle and system initialization templates
- **drawing/**: Drawing instruction templates for visual effects
- **render/**: Render loop templates

## Usage

### API Configuration

1. Copy the environment template:

   ```bash
   cp .env.example .env
   ```

2. Add your API key(s) to the `.env` file:
   ```
   GEMINI_API_KEY=your-key-here      # Google Gemini (recommended)
   OPENAI_API_KEY=your-key-here      # OpenAI GPT models
   ANTHROPIC_API_KEY=your-key-here   # Claude models
   MISTRAL_API_KEY=your-key-here     # Mistral models
   ```

### Textual User Interface (Recommended)

Launch the interactive terminal UI for a guided experience:

```bash
poetry run python src/tolvera/llm/ui_scripts/tolvera_textual_ui.py
```

The UI provides:

- Interactive behavior input with autocomplete suggestions
- Real-time code generation and preview
- Model selection across all configured providers
- Sketch management (save/load/export)
- Integrated help system (F1) and tutorial (F2)
- Chat panel for iterative refinement
- Diff viewer to see code changes

### Command-Line Demo

For direct API usage and testing, run the comprehensive demo:

```bash
poetry run python src/tolvera/llm/ui_scripts/tolvera_llm_demo.py
```

The demo showcases:

- Basic single-particle behaviors (gravity, random movement)
- Complex multi-component behaviors with decomposition
- Visual effects and drawing behaviors
- Multi-species ecosystem interactions
- Automatic state generation for complex behaviors
- Artificial life pattern synthesis (cellular automata, swarms)

## Example Behaviors

### Simple Behaviors

```
"Particles are attracted to the center"
"Add gravity and boundary collisions"
"Particles randomly walk around the screen"
```

### Species Interactions

```
"Red predators chase blue prey"
"Three species form a rock-paper-scissors dynamic"
"Green algae grows while being eaten by fish"
```

### Complex Ecosystems

```
"Fish school together and avoid sharks"
"Implement Conway's Game of Life with particles"
"Slime mold simulation with nutrient seeking"
```

### Visual Effects

```
"Particles leave glowing trails"
"Draw connections between nearby particles"
"Create pulsing halos around each species"
```

## Key Features

### Interactive Textual UI

The system includes a terminal-based user interface built with Textual. The UI features a chat panel for behavior input, real-time code preview, model selection, and integrated sketch management.

<!-- [TODO: SCREENSHOT_OF_UI_HERE] -->

### HTML Trace Reporting

Comprehensive debugging and analysis capabilities through interactive HTML reports that visualize the entire synthesis pipeline. Each report includes:

- Hierarchical trace visualization
- Timing metrics for each component
- LLM call details and token usage
- State and species analysis results
- Generated code snippets at each stage

<!-- [TODO: SCREENSHOT_OF_HTML_REPORT_HERE] -->

## Architecture Highlights

### Pipeline Orchestration

The system uses a orchestration pattern where the BehaviorOrchestrator coordinates multiple specialized agents, each responsible for a specific aspect of the synthesis process. This separation of concerns helps with modularity and maintainability.

### Dynamic Context Selection

Rather than static prompt templates, the system uses an LLM-powered context selector that chooses relevant documentation and patterns based on the specific behavior being synthesized (think very simplistic RAG), improving generation quality.

### Structured Output Generation

All LLM interactions use Pydantic models for structured output.

### Template-Based Code Assembly

The final code generation uses Jinja2 templates, separating code structure from generation logic and allowing for consistent, maintainable output formatting.

## Development

### Adding New Behaviors

To extend the system with new behavior types:

1. Add pattern examples to the appropriate context file in `context/`
2. Create or modify prompt templates in `prompts/synthesis/`
3. Update the BehaviorAnalyzer if new decomposition patterns are needed

### Extending Model Support

New LLM providers can be added by:

1. Implementing the provider in `core/llm_factory.py`
2. Adding appropriate model configuration
3. Ensuring compatibility with structured output requirements

### Custom Templates

New code generation templates can be added to `templates/` following the existing Jinja2 patterns, with corresponding rendering methods in `TemplateRenderer`.

## Performance Considerations

- Context selection is optimized to minimize token usage (this could be updated)
- Use models with a faster inference time but maybe lower accuracy for a better user experience while interacting with the system
