# Tölvera LLM Templates

This directory contains all Jinja2 templates used throughout the Tölvera LLM code generation system. These templates separate the code structure and formatting from the Python generation logic, ensuring consistent and maintainable output.

## Directory Structure

- **kernel/** - Templates for generating Taichi kernel functions
- **sketch/** - Templates for assembling complete Python sketch files
- **state/** - Templates for state initialization code
- **init/** - Templates for particle system initialization

## Template Renderer

Templates are processed via the `TemplateRenderer` class in `template_renderer.py`, which handles:
- Jinja2 environment configuration with proper whitespace control
- Template loading and rendering with variable substitution
- Context preparation and data passing
- Error handling and template validation

## Main Templates

### kernel/integration_kernel.j2

Generates the main integration kernel that applies all behavior experts to particles. This template handles:
- Single-particle force calculations
- Particle-particle interactions
- Species-specific behavior branching
- Force aggregation and velocity updates

### kernel/drawing_kernel.j2

Generates kernels for visual effects and drawing behaviors. Features include:
- Pre and post-draw ordering
- Visual expert function calls
- Pixel buffer manipulation
- Effect compositing

### kernel/utility_kernel.j2

Generates kernels for utility functions such as:
- State updates and temporal dynamics
- Grid-based operations
- Statistical calculations
- System-wide parameter adjustments

### sketch/final_sketch.j2

Assembles complete, runnable Python sketches by combining:
- Import statements and dependencies
- Expert function definitions
- State initialization code
- Integration and drawing kernels
- Main render loop
- Tölvera instance configuration

### state/state_initialization.j2

Generates initialization code for custom particle and global states:
- Particle state field initialization
- Global state setup
- Grid-based state patterns
- Random state distributions

### init/particle_initialization.j2

Creates particle system initialization code including:
- Species-based position patterns (random, grid, clustered)
- Velocity and mass initialization
- Species ID assignment
- Activation states

## Variable Injection

Templates support extensive variable injection using Jinja2 syntax. Common variables include:

- `single_expert_names`: List of single-particle expert function names
- `interaction_expert_names`: List of interaction expert function names
- `expert_weights`: Dictionary mapping expert names to their weights
- `species_config`: Species configuration with names, IDs, and interaction pairs
- `state_definitions`: Custom state field definitions
- `description`: Original behavior description from user

## Template Features

### Conditional Rendering

Templates use Jinja2 conditionals to adapt output based on context:
```jinja
{% if has_species_specific %}
    # Species-specific behavior logic
{% endif %}
```

### Loop Constructs

Iterating over experts and states:
```jinja
{% for expert_name in single_expert_names %}
    total_force += {{ expert_name }}(pos, vel, mass, species, i) * {{ weight }}
{% endfor %}
```

### Whitespace Control

Templates are configured with `trim_blocks=True` and `lstrip_blocks=True` to ensure clean output formatting without excessive whitespace.

## Usage Example

```python
from tolvera.llm.templates.template_renderer import TemplateRenderer

renderer = TemplateRenderer()

# Render an integration kernel
kernel_code = renderer.render_integration_kernel(
    single_expert_names=['expert_gravity', 'expert_wander'],
    interaction_expert_names=['expert_chase'],
    expert_weights={'expert_gravity': 1.0, 'expert_wander': 0.5},
    species_config=species_config
)

# Render a complete sketch
sketch_code = renderer.render_sketch(
    description="Predator-prey simulation",
    experts=[expert_code],
    kernel=kernel_code,
    init_code=init_code
)
```

## Development Guidelines

When creating or modifying templates:

1. **Maintain Consistency**: Follow existing indentation and formatting patterns
2. **Use Comments**: Add Jinja2 comments to explain complex logic
3. **Test Rendering**: Verify output code is syntactically valid Python/Taichi
4. **Handle Edge Cases**: Ensure templates work with empty or minimal inputs
5. **Document Variables**: List required and optional variables in template comments