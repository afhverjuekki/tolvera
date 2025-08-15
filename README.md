# Tölvera Natural Language Interface

This branch specifically focusing on extending Tölvera with a natural language interface for generating complex particle behaviors, simulations, and alife patterns from text descriptions.

## What You Can Create

- **Particle Behavior**: "Particle repel the center of the screen"
- **Physical Simulations**: "Particles fall with gravity and bounce off boundaries"
- **Species Interactions**: "Species one repels species two."
- **Complex Behaviors**: "Red hunters chase blue prey that try to escape""

All generated as complete, runnable Python code with GPU acceleration via Taichi.

## Quick Start

### Prerequisites

- **Python 3.10-3.12** (Python 3.13+ not supported due to Taichi)
- **Poetry** (Python package manager)
- **API Key** from at least one supported provider (see configuration below)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Intelligent-Instruments-Lab/tolvera.git
   cd tolvera
   ```

2. **Install dependencies:**

   ```bash
   poetry install
   ```

3. **Configure API keys** (see next section)

4. **Launch the interface:**
   ```bash
   poetry run python examples/tolvera_textual_ui.py
   ```

## API Key Configuration

**Important**: API keys are required to use this natural language interface. You can try using with Ollama, but these smaller models hallucinate too much to be dependable for generating these types of Tölvera sketches. You need at least one provider configured.

### Step 1: Copy Environment Template

```bash
cp .env.example .env
```

### Step 2: Choose a Provider & Get API Key

#### **Gemini (Recommended)**

- **Best for**: Tölvera synthesis, fast and reliable
- **Get API Key**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Add to .env**: `GEMINI_API_KEY=your-api-key-here`

#### **OpenAI (GPT-5)**

- **Best for**: High-quality code generation
- **Get API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Add to .env**: `OPENAI_API_KEY=your-api-key-here`

#### **Anthropic (Claude)**

- **Best for**: Complex reasoning and analysis
- **Get API Key**: [Anthropic Console](https://console.anthropic.com/)
- **Add to .env**: `ANTHROPIC_API_KEY=your-api-key-here`

#### **Mistral AI**

- **Best for**: European users, good balance
- **Get API Key**: [Mistral Console](https://console.mistral.ai/api-keys/)
- **Add to .env**: `MISTRAL_API_KEY=your-api-key-here`

#### **Local Models (Ollama)**

- **Best for**: Privacy, no API costs
- **Setup**:
  1. Install [Ollama](https://ollama.ai/)
  2. `ollama pull llama3.2`
  3. `ollama serve`
- **No API key required**

### Step 3: Verify Configuration

When you launch Tölvera, it will show which providers are configured:

- **Green**: Provider ready
- **Red**: Provider not configured

## Running the Application

### Launch Command

```bash
poetry run python examples/tolvera_textual_ui.py
```

### First Run Experience

1. **Welcome Screen**: Introduction to Tölvera with animated examples
2. **Model Selection**: Choose your AI provider and model
3. **Agent Initialization**: ~10-30 seconds setup (one-time)
4. **Ready to Create**: Start generating!

### Interface Overview

- **Description Input**: Write what you want to create
- **Generate Button**: Transform text into code
- **Code Editor**: View and edit generated Python
- **Controls**: Run, save, load, and refine your creations
- **Chat Panel**: Iteratively improve your simulations
- **Status Logs**: Monitor generation progress

## Getting Started Tutorial

### Interactive Tutorial

Press **F2** at any time to launch the interactive tutorial that walks you through:

1. **Your First Generation**: Create a simple two-species simulation
2. **Running Simulations**: Launch and stop your Tölvera sketch
3. **Refinement Chat**: Modify behaviors with natural language
4. **Diff Viewing**: See exactly what the LLM changed
5. **Advanced Features**: Save, load, and export capabilities

### Getting Help

- **F1**: General help and keyboard shortcuts
- **F2**: Interactive tutorial (anytime)
- **Ctrl+T**: Toggle chat panel
- **Status Logs**: Real-time progress and error information
- **Trace Report**: See the process for all the calls to the LLM

## Example Workflows

### Simple Behavior

```
Description: "Particles are attracted to the center and repel each other"
→ Click Generate → Click Run → Watch the simulation!
```

### Complex Ecosystem

```
Description: "Red predators hunt blue fish while green algae grows slowly"
→ LLM detects 3 species → Generates predator-prey behaviors → Creates ecosystem
```

### Iterative Refinement

```
1. Generate initial behavior
2. Run and observe
3. Chat: "Make the predators faster and add boundaries"
4. LLM refines the code
5. Run updated simulation
```

## Keyboard Shortcuts

| Key        | Action            |
| ---------- | ----------------- |
| **Ctrl+N** | New sketch        |
| **Ctrl+R** | Run simulation    |
| **Ctrl+S** | Save sketch       |
| **Ctrl+T** | Toggle chat panel |
| **F1**     | Help dialog       |
| **F2**     | Tutorial          |
| **Ctrl+Q** | Quit application  |

## Troubleshooting

### Common Issues

**"No providers configured"**

- Check your `.env` file exists and has valid API keys
- There's an `.env.example` you can use to format your `.env` after
- Verify API key format (no quotes, no extra spaces)
- Test API key on the provider's website

**"Generation failed"**

- Check status logs for detailed error messages
- Try simplifying your description

**Ollama Issues**

- Ensure Ollama is running: `ollama serve`
- Check model is installed: `ollama list`
- Verify connection: `curl http://localhost:11434/api/tags`

### Getting Support

- **Status Logs**: Always check for detailed error messages
- **Copy Logs Button**: Share logs when asking for help
- **GitHub Issues**: [Report bugs](https://github.com/mclemcrew/tolvera/tree/week11)
- **Documentation**: [Full docs](https://afhverjuekki.github.io/tolvera/)
- **Contact Me**: There's probably a lot wrong with the system right now. Pinging me (@MClem) on the Tölvera Discord is a great way to get my attention! Otherwise, feel free to post an issue on

## Some Other Features

### Multiple AI Providers

Switch between providers anytime with **"Change Model"** button

### Code Saving

Save complete Tölvera sketches to run later

### Debug Tracing

Generate detailed HTML reports of the synthesis process

---

**Ready to begin? Run the interface and see what sketches you create!**

```bash
poetry run python examples/tolvera_textual_ui.py
```
