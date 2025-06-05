# Tölvera Natural Language Interface

# Overview

This branch introduces the LLM integration module for Tölvera that allows users to generate sketch configurations and code through natural language descriptions. The module leverages local LLM capabilities (tested on llama3.5 and qwen2.5) via Ollama and implements a pipeline for generating validated, runnable code.

## Components

The following describes a brief overview for some of the components in the implementation (that were general assumptions I made and should be very much be looked over) while mocking this up:

- LLM Interface: The class that manages communication with the language model and handles prompt engineering for optimal results.
- JSON Schema Adherence: Pydantic models that define and validate sketch configurations, asserting all generated parameters are within acceptable ranges (schema adherence),
- Code Generation: Jinja2 templates that transform validated configurations into executable Tölvera Python code.

## How to run

A CLI example is provided in `llm_cli_example.py` that allows natural language interaction for generating and modifying Tölvera sketches. When you run this script (`python llm_cli_example.py`), you'll be met with a choice of implementing either a flock or slime algorithm (I only worked with the `tv.vera` module for this demo) and then asked to enter a prompt and some examples are provided. After this, the query is then sent to the LLM. The LLM process the commands and sends back JSON which is validated via all the pydantic definitions in the `definitions.py` file and the the script is generated via the `codegen.py` file. In the background, we open a terminal and run the script that was just generated and once it runs, it'll open up the window. You can then visually see what you've created and modify accordingly via natural language. If any modifications are requested, the whole process is run again (with a different prompt with the current params sent to the model) and then current script is rewritten with the updated changes. The old window running the script is then terminated and a new window appears. A video demonstration of this process is documented later in this readme.

## Technical Implementation

The core of this LLM integration relies on a two-stage approach to transform natural language into executable code. The LLM class handles communication with Ollama. It implements a prompt engineering strategy that improves the model's ability to generate valid configurations. I found that including both positive and negative examples in the prompts was crucial (in many LLM papers you’ll see this too) - showing the model not just what we want, but explicitly what we DON’T want. This significantly reduced hallucination issues, particularly with complex parameters when I was dealing with slime.

For the configuration pipeline, I used Pydantic's validation capabilities. Rather than letting the LLM generate code directly (which proved wildly inconsistent), forcing it to produce a structured JSON configuration provides a validation checkpoint where we can catch and fix issues before code generation. The SketchConfig model includes nested validators that handle species indices, parameter bounds, and cross-dependencies between configuration elements.

The Jinja2 templating system was more reliable than asking the LLM to generate complete code blocks (which also wasn’t great). This is my first time using Jinja is something that isn’t a personal project on my computer so if anyone has experience with this, please look closer because I’m sure I missed something.

One particularly challenging aspect throughout this whole thing was color handling which you’ll no doubt question why that was implemented. Models frequently hallucinate color values or formats, so I implemented a multi-tiered approach: first checking a lookup table of common colors, then asking the LLM specifically about unknown colors, and finally falling back to default values (just a grey color) when needed. This strategy handled failures gracefully without breaking the entire generation pipeline. This really depended on the model for how much the code had to fall back to default even with examples provided.

This implementation is designed with the assumption that failures will occur. Each component can operate independently and provides clear feedback when issues arise, hopefully allowing for graceful degradation rather than unknown and confusing failure points. Throughout this implementation, making it clear to the user what is going on was a key design feature. Hopefully that is clear in the code though 😅

#### Generating from scratch

https://github.com/user-attachments/assets/46b89b5f-673c-45c2-b4c5-85f16db89dd0

#### Modifying the code in real time

https://github.com/user-attachments/assets/9f03912a-c450-4a51-8705-24fe540ea5ec

#### Showing the generated code

https://github.com/user-attachments/assets/6121e2b3-7be2-42d7-8e60-77a251da6e3b

For brevity (and because the video files are too large), the full video demo for slime and flock implementations can be viewed by clicking on the respective link below.

[Full Slime demo](https://drive.google.com/file/d/1ywwNx_fc9A3YJSvIGZV1-7fxlgrgaWCB/view?usp=sharing)
[Full Flock demo](https://drive.google.com/file/d/1_GY-A8FvAZjqQqOhxPJCXogDhYLO812s/view?usp=sharing)

The new tree structure I'm proposing looks like this:

- src/tolvera/
  - llm/
    - **init**.py - Module exports and documentation
    - llm.py - Core LLM implementation
    - definitions.py - Pydantic models and configuration classes
    - codegen.py - Code generation utilities
    - llm_cli_example.py - CLI tool for testing
    - prompts/
      - prompt_flock.txt - Flock-specific prompting examples
      - prompt_slime.txt - Slime-specific prompting examples
    - templates/
      - template_flock.py.j2 - Flock template
      - template_slime.py.j2 - Slime template

## Future Work/Things That Should be Addressed

Although this proof-of-concept demonstrates the feasibility of this approach, for each new example that is added significant testing needs to be conducted with multiple models to ensure that the prompts created are working as intended and do not confuse the user if they should fail. Using multiple LLMs in this way can easily break the system and I would highly recommend to develop this in a manner where failing is the norm, instead of an unintended incident. I implemented a retry approach where some common issues that I was seeing are fixed automatically (colors were notoriously difficult to handle), but your mileage may very based on what example you are implementing.
