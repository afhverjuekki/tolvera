#!/usr/bin/env python

"""
Tölvera LLM Sketch Generator

A CLI for generating and modifying Tölvera sketches (only Flock and Slime simulations for now)
using Large Language Models via Ollama (verified with qwen2:latest).

This script provides an interactive interface to:
- Select a sketch type (Flock or Slime)
- Define sketch parameters through natural language descriptions
- Generate Python code based on the configuration
- Run the generated sketch in the background (window will pop up)
- Iteratively refine/modify the sketch through natural language

The generator handles configuration validation, code generation,
and manages background processes for running sketches.

Usage:
    python llm_cli_example.py [--model MODEL_NAME] (defaults to qwen2:latest)

Arguments:
    --model: Ollama model name to use (default: "qwen2.5:latest")
             Examples: "llama3", "qwen2.5:latest"

Requirements:
    - Tölvera package with llm module (llm.py, codegen.py, definitions.py)
    - Ollama installed and running (https://ollama.com)
"""

import sys
from pathlib import Path
import argparse
import json
import subprocess
import time
import os
import atexit

# ANSI Color Support
_IS_TTY = sys.stdout.isatty()

def colorize(text, color_code):
    reset_code = "\033[0m"
    return f"{color_code}{text}{reset_code}" if _IS_TTY else text

CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_RED = "\033[91m"
CLR_BLUE = "\033[94m"
CLR_CYAN = "\033[96m"
CLR_HEADER = "\033[1;96m"

def print_header(text): print(colorize(text, CLR_HEADER))
def print_success(text): print(colorize(text, CLR_GREEN))
def print_warning(text): print(colorize(text, CLR_YELLOW))
def print_error(text): print(colorize(text, CLR_RED))
def print_info(text): print(colorize(text, CLR_BLUE))

def prompt(text):
    try: 
        return input(text)
    except EOFError: 
        return None

try:
    from tolvera.llm.llm import LLM
    from tolvera.llm.codegen import generate_code_from_sketch_config
except ImportError as e:
    print_error(f"Failed to import LLM components: {e}")
    sys.exit(1)

current_sketch_process = None

def save_sketch(sketch_config, generated_code):
    """Saves the generated code to a Python file (name is based on prompt to LLM)."""
    try:
        safe_filename_base = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in sketch_config.sketch_name)
        if not safe_filename_base:
            safe_filename_base = f"tolvera_sketch_{int(time.time())}"
        output_filename = f"{safe_filename_base}.py"
        output_path = Path(output_filename)

        output_path.write_text(generated_code, encoding='utf-8')
        print_success(f"Sketch saved to: {colorize(str(output_path.resolve()), CLR_BLUE)}")
        return output_path
    except Exception as e:
        print_error(f"Error saving file: {e}")
        return None

def run_sketch_background(sketch_path):
    """Runs the sketch in a background process."""
    global current_sketch_process

    if current_sketch_process and current_sketch_process.poll() is None:
        print("Stopping previous sketch...", end='', flush=True)
        try:
            current_sketch_process.terminate()
            current_sketch_process.wait(timeout=1.0)
            print(colorize(" Done.", CLR_YELLOW))
        except subprocess.TimeoutExpired:
            current_sketch_process.kill()
            current_sketch_process.wait()
            print(colorize(" Done (forced).", CLR_YELLOW))
        except Exception:
            print()  # Just finish the line
        current_sketch_process = None

    # Start new process after finishing the last running one
    if sketch_path and sketch_path.exists():
        print("Running sketch in background...", end='', flush=True)
        try:
            current_sketch_process = subprocess.Popen(
                [sys.executable, str(sketch_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0),
                preexec_fn=(os.setsid if os.name != 'nt' else None)
            )
            time.sleep(0.7)
            if current_sketch_process.poll() is None:
                print(colorize(" Running.", CLR_GREEN))
            else:
                print_error("\nProcess exited immediately. Check the generated code.")
                current_sketch_process = None
        except Exception as e:
            print_error(f"\nFailed to start process: {e}")
            current_sketch_process = None
    elif not sketch_path:
        print_error("Invalid path provided.")
    else:
        print_error(f"File not found: {sketch_path}")

def cleanup_processes():
    """Terminates any running sketch process."""
    global current_sketch_process
    if current_sketch_process and current_sketch_process.poll() is None:
        print(colorize("\nCleaning up...", CLR_YELLOW), end='', flush=True)
        try:
            current_sketch_process.terminate()
            current_sketch_process.wait(timeout=0.5)
        except Exception:
            try:
                current_sketch_process.kill()
                current_sketch_process.wait()
            except Exception:
                pass
        print(colorize(" Done.", CLR_YELLOW))
    current_sketch_process = None

def main():
    """Main function with interactive loop for generating Tölvera sketches."""
    global current_sketch_process
    atexit.register(cleanup_processes)

    parser = argparse.ArgumentParser(
        description="Generate and refine Tölvera sketches using an LLM."
    )
    parser.add_argument("--model", type=str, default="qwen2.5:latest", 
                        help="Ollama model name (e.g., 'llama3', 'qwen2.5:latest').")
    args = parser.parse_args()

    print_header("\n====================================================")
    print_header("    Tölvera LLM Sketch Generator (Flock & Slime)")
    print_header("====================================================")
    print(f"Using model: {colorize(args.model, CLR_CYAN)}\n")

    print("Initializing LLM...")
    try:
        llm_module = LLM(ollama_model=args.model)
        print_success("LLM module ready.\n")
    except Exception as e:
        print_error(f"LLM initialization failed: {e}")
        return

    print("Select the type of Tölvera sketch you want to create:")
    print(colorize("  1. Flock", CLR_YELLOW))
    print(colorize("  2. Slime", CLR_YELLOW))
    while True:
        choice = prompt("Enter choice (1 or 2) > ")
        if choice == "1":
            sketch_type = "flock"
            break
        elif choice == "2":
            sketch_type = "slime"
            break
        else:
            print_warning("Invalid choice. Please enter 1 or 2.")
    print_info(f"Selected sketch type: {colorize(sketch_type.capitalize(), CLR_CYAN)}\n")

    print(f"Enter a description for your {sketch_type.capitalize()} sketch:")
    print(colorize("Examples:", CLR_BLUE))
    if sketch_type == "flock":
        print(colorize("  'Basic flocking with 500 particles'", CLR_BLUE))
        print(colorize("  'Flock with 2 species, species 0 red and species 1 blue'", CLR_BLUE))
        print(colorize("  'Flocking, 200 particles, species 0 avoid species 1'", CLR_BLUE))
    else:
        print(colorize("  'Slime mold simulation with 2000 particles'", CLR_BLUE))
        print(colorize("  'Physarum sim, 3 species, low evaporation rate'", CLR_BLUE))
        print(colorize("  'Slime with magenta and blue species'", CLR_BLUE))

    user_prompt = prompt(f"\nDescription > ")
    if not user_prompt or not user_prompt.strip():
        print("\nNo description provided. Exiting.")
        return

    initial_llm_prompt = f"Generate a Tölvera sketch configuration. The sketch_type MUST be '{sketch_type}'. User description: {user_prompt}"
    print("\nGenerating initial configuration...")
    current_sketch_config = llm_module.generate_sketch_config_from_prompt(initial_llm_prompt)

    if current_sketch_config is None:
        print_error("\nFailed to generate initial configuration. Please try again.")
        return

    try:
        while True:
            print("\nGenerating Python code...")
            try:
                generated_code = generate_code_from_sketch_config(
                    config=current_sketch_config,
                    llm_module=llm_module
                )
                if not generated_code:
                    print_error("Code generation failed.")
                    break
            except Exception as e:
                print_error(f"Code generation error: {e}")
                break

            saved_path = save_sketch(current_sketch_config, generated_code)
            if saved_path:
                run_sketch_background(saved_path)
            else:
                print_warning("Failed to save sketch.")
                break

            # Modification, if any
            print("\n----------------------------------------------------")
            print(f"Sketch '{colorize(current_sketch_config.sketch_name, CLR_BOLD)}' ({current_sketch_config.sketch_type.capitalize()}) updated.")
            print("Enter modifications (or type 'quit'/'exit'):")
            print(colorize("Examples:", CLR_BLUE))
            if current_sketch_config.sketch_type == "flock":
                print(colorize("  'Change species 1 color to green'", CLR_BLUE))
                print(colorize("  'Use 2000 particles instead'", CLR_BLUE))
                print(colorize("  'Make species 0 separate from species 1'", CLR_BLUE))
            else:
                print(colorize("  'Decrease the evaporation rate to 0.85'", CLR_BLUE))
                print(colorize("  'Increase particle count to 3000'", CLR_BLUE))
                print(colorize("  'Make species 0 yellow with wider sensing angle'", CLR_BLUE))

            modification_prompt = prompt("\nRefinement > ")
            if modification_prompt is None or modification_prompt.lower().strip() in ['quit', 'exit', '']:
                break

            print("\nApplying modification...")
            try:
                current_json = current_sketch_config.model_dump_json(indent=2)
            except AttributeError:
                current_json = json.dumps(current_sketch_config.dict(), indent=2)

            new_sketch_config = llm_module.generate_sketch_config_from_prompt(
                user_input=modification_prompt,
                current_config_json=current_json
            )

            if new_sketch_config:
                if new_sketch_config.sketch_type != current_sketch_config.sketch_type:
                    print_warning(f"Keeping original sketch type '{current_sketch_config.sketch_type}'")
                    new_sketch_config.sketch_type = current_sketch_config.sketch_type

                print_success("Modification applied.")
                if new_sketch_config.sketch_name != current_sketch_config.sketch_name:
                    print_info(f"Sketch renamed to '{new_sketch_config.sketch_name}'")
                current_sketch_config = new_sketch_config
            else:
                print_warning("Failed to apply modification. Keeping previous version.")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print_header("\n====================================================")
        print_header("    Generator finished")
        print_header("====================================================")

if __name__ == '__main__':
    main()