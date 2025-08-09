"""
Temporal dynamics patterns for time-based state evolution.
Provides rich context for the LLM to understand and generate temporal updates.
"""

TEMPORAL_DYNAMICS_PATTERNS = """
# Temporal State Dynamics Patterns

## Energy Systems
Energy represents the capacity for action and naturally depletes with activity:

### Depletion Patterns
```python
# Slow energy loss (resting metabolism)
energy *= 0.995  # ~0.5% loss per frame

# Moderate energy loss (normal activity)
energy *= 0.99   # ~1% loss per frame

# Fast energy loss (intense activity)
energy *= 0.95   # ~5% loss per frame

# Activity-based depletion
speed = vel.norm()
energy_loss = base_metabolism + activity_cost * speed
energy = max(0.0, energy - energy_loss)
```

### Regeneration Patterns
```python
# Constant regeneration
energy = min(max_energy, energy + regen_rate)

# Conditional regeneration (resting)
if vel.norm() < rest_threshold:
    energy = min(max_energy, energy + rest_regen_rate)

# Resource-based regeneration
if near_food and energy < max_energy:
    energy += food_energy_value
    food_consumed = True
```

### Behavioral Coupling
```python
# Energy affects movement speed
if energy < tired_threshold:
    vel *= (0.3 + 0.7 * (energy / tired_threshold))

# Energy affects behavior selection
if energy < hunt_threshold:
    behavior_mode = "rest"
else:
    behavior_mode = "hunt"

# Critical energy triggers
if energy <= 0.0:
    particle.active = 0.0  # Death from exhaustion
```

## Life Cycles and Aging

### Age Progression
```python
# Simple aging
age += dt

# Stage-based aging with transitions
age += dt
if age < juvenile_age:
    life_stage = 0  # Juvenile
elif age < adult_age:
    life_stage = 1  # Adult
else:
    life_stage = 2  # Elder

# Age affects capabilities
max_speed = base_speed * (1.0 - age_penalty * age)
```

### Growth Patterns
```python
# Exponential growth (early stage)
size *= 1.01  # 1% growth per frame

# Logistic growth (S-curve)
growth_rate = max_growth_rate * (1.0 - size / max_size)
size += growth_rate * dt

# Staged growth
if life_stage == 0:  # Juvenile
    size = min(juvenile_max_size, size + growth_rate)
elif life_stage == 1 and size < adult_size:  # Adult growth spurt
    size = min(adult_size, size + growth_rate * 2)
```

## Oscillations and Cycles

### Phase Updates
```python
# Simple oscillation
phase += frequency * dt
if phase > 2 * pi:
    phase -= 2 * pi

# Phase with decay
phase += frequency * dt
amplitude *= decay_factor
value = amplitude * sin(phase)

# Coupled oscillators (synchronization)
# Kuramoto model
for neighbor in neighbors:
    phase_diff = neighbor.phase - phase
    phase += coupling_strength * sin(phase_diff) * dt
```

### Day/Night Cycles
```python
# Global day phase
day_phase += day_speed * dt
if day_phase > 1.0:
    day_phase -= 1.0
    
is_day = day_phase < 0.5

# Behavior changes with time
if is_day:
    activity_level = 1.0
    vision_range = day_vision
else:
    activity_level = 0.3
    vision_range = night_vision
```

## Resource Dynamics

### Consumption Patterns
```python
# Constant consumption
resources -= consumption_rate

# Activity-based consumption
resources -= base_consumption + activity_consumption * speed

# Threshold-based consumption
if performing_action:
    resources -= action_cost
    if resources < 0:
        action_failed = True
        resources = 0
```

### Production Patterns
```python
# Constant production
resources += production_rate

# Conditional production
if in_resource_zone:
    resources += harvest_rate

# Efficiency-based production
efficiency = skill_level * tool_quality
resources += base_production * efficiency
```

## Temperature and Heat

### Heat Dissipation
```python
# Newton's law of cooling
temp_diff = temperature - ambient_temp
temperature -= cooling_rate * temp_diff

# Rapid cooling
temperature *= 0.95  # 5% loss per frame

# Heat with sources
if near_heat_source:
    temperature += heating_rate
temperature *= (1.0 - dissipation_rate)
```

### Temperature Effects
```python
# Temperature affects metabolism
metabolic_rate = base_rate * (1.0 + 0.1 * (temperature - optimal_temp))

# Critical temperatures
if temperature > boiling_point:
    state = "gas"
elif temperature < freezing_point:
    state = "solid"
```

## Memory and Decay

### Memory Fade
```python
# Exponential memory decay
memory_strength *= 0.999

# Discrete memory with timeout
memory_timer -= dt
if memory_timer <= 0:
    memory_value = default_value
```

### Trail/Pheromone Decay
```python
# Simple evaporation
pheromone_strength *= evaporation_rate

# Diffusion and decay
pheromone_strength = pheromone_strength * 0.99 + neighbor_avg * 0.01
```

## State-Dependent Dynamics

### Hunger/Satiation
```python
# Hunger increases over time
hunger = min(max_hunger, hunger + hunger_rate * dt)

# Eating reduces hunger
if eating:
    hunger = max(0, hunger - food_value)
    
# Hunger affects behavior
if hunger > desperate_threshold:
    risk_tolerance = high_risk
    search_radius = extended_radius
```

### Fatigue/Stamina
```python
# Stamina depletion during activity
if running:
    stamina -= sprint_cost * dt
else:
    stamina = min(max_stamina, stamina + recovery_rate * dt)

# Fatigue accumulation
fatigue += activity_level * fatigue_rate
if resting:
    fatigue *= rest_recovery_factor
```

## Temporal Update Guidelines for LLM

When generating temporal updates:

1. **Identify temporal keywords**:
   - "over time" → continuous update each frame
   - "gradually" → small increments (0.99x or ±0.01)
   - "quickly" → larger changes (0.9x or ±0.1)
   - "slowly" → tiny changes (0.999x or ±0.001)
   
2. **Choose appropriate rates**:
   - Decay: multiply by (1.0 - decay_rate)
   - Growth: multiply by (1.0 + growth_rate)
   - Linear change: add/subtract constant
   
3. **Apply constraints**:
   - Always clamp: value = max(min_val, min(max_val, value))
   - Check thresholds for state changes
   
4. **Create behavioral coupling**:
   - Map state ranges to behavior modifications
   - Use smooth transitions: lerp or curve functions
   - Define critical points for discrete changes

5. **Consider update frequency**:
   - Every frame: continuous processes
   - Every N frames: discrete updates
   - Conditional: based on events or thresholds
"""

TEMPORAL_UPDATE_EXAMPLES = """
# Example Temporal Update Implementations

## Example 1: Energy System with Activity Coupling
```python
@ti.kernel
def update_energy_dynamics():
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            # Get current state
            energy = tv.s.llm_particle.field[i].energy
            vel = tv.p.field[i].vel
            species = tv.p.field[i].species
            
            # Calculate energy loss based on activity
            speed = vel.norm()
            base_metabolism = 0.5
            movement_cost = 0.01
            energy_loss = base_metabolism + movement_cost * speed
            
            # Species-specific metabolism
            if species == 0:  # Predator
                energy_loss *= 1.2  # Higher metabolism
            elif species == 1:  # Prey
                energy_loss *= 0.8  # More efficient
            
            # Apply energy loss
            energy -= energy_loss
            
            # Regeneration when resting
            if speed < 5.0:
                energy += 0.3
            
            # Clamp and store
            energy = max(0.0, min(100.0, energy))
            tv.s.llm_particle.field[i].energy = energy
            
            # Behavioral coupling
            if energy < 20.0:
                # Tired particles move slower
                tv.p.field[i].vel *= 0.8
            
            if energy <= 0.0:
                # Death from exhaustion
                tv.p.field[i].active = 0.0
```

## Example 2: Phase-Based Oscillation
```python
@ti.kernel
def update_oscillator_phases():
    dt = 1.0 / 60.0  # Assuming 60 FPS
    
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            # Update phase
            phase = tv.s.llm_particle.field[i].phase
            frequency = tv.s.llm_particle.field[i].frequency
            
            phase += frequency * dt * 2 * 3.14159
            
            # Wrap phase
            if phase > 2 * 3.14159:
                phase -= 2 * 3.14159
            
            tv.s.llm_particle.field[i].phase = phase
            
            # Apply oscillation to visual property
            brightness = (ti.sin(phase) + 1.0) * 0.5
            tv.p.field[i].size = 2.0 + brightness * 3.0
```

## Example 3: Resource Consumption with Behavior States
```python
@ti.kernel
def update_resource_dynamics():
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            resources = tv.s.llm_particle.field[i].resources
            behavior_state = tv.s.llm_particle.field[i].behavior_state
            
            # Different consumption based on behavior
            if behavior_state == 0:  # Idle
                resources -= 0.1
            elif behavior_state == 1:  # Foraging
                resources -= 0.3
                # Chance to find resources
                if ti.random() < 0.01:
                    resources += 10.0
            elif behavior_state == 2:  # Fighting
                resources -= 1.0
            
            # State transitions based on resources
            if resources < 10.0:
                # Low resources trigger foraging
                tv.s.llm_particle.field[i].behavior_state = 1
            elif resources > 80.0:
                # Well-fed particles can afford to be idle
                tv.s.llm_particle.field[i].behavior_state = 0
            
            # Clamp and store
            resources = max(0.0, min(100.0, resources))
            tv.s.llm_particle.field[i].resources = resources
            
            # Death from starvation
            if resources <= 0.0:
                tv.p.field[i].active = 0.0
```

## Example 4: Temperature with Environmental Interaction
```python
@ti.kernel
def update_temperature_dynamics():
    ambient_temp = 20.0
    
    for i in range(tv.pn):
        if tv.p.field[i].active > 0:
            temp = tv.s.llm_particle.field[i].temperature
            pos = tv.p.field[i].pos
            
            # Heat sources in environment
            heat_source_pos = ti.Vector([tv.x * 0.5, tv.y * 0.5])
            dist_to_heat = (pos - heat_source_pos).norm()
            
            if dist_to_heat < 100.0:
                # Heating near source
                heating = (100.0 - dist_to_heat) / 100.0 * 2.0
                temp += heating
            
            # Cooling (Newton's law)
            temp_diff = temp - ambient_temp
            temp -= 0.05 * temp_diff
            
            # Clamp
            temp = max(0.0, min(100.0, temp))
            tv.s.llm_particle.field[i].temperature = temp
            
            # Temperature affects movement
            if temp < 10.0:
                # Cold particles move slowly
                tv.p.field[i].vel *= 0.9
            elif temp > 90.0:
                # Hot particles move erratically
                tv.p.field[i].vel += ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 10.0
```
"""

TEMPORAL_SYNTHESIS_PROMPTS = """
# Prompts for Temporal Update Synthesis

## Prompt Template for Temporal Analysis
When analyzing "{description}" for temporal dynamics:

1. Identify temporal indicators:
   - Explicit: "over time", "gradually", "depletes", "regenerates"
   - Implicit: "tired", "hungry", "grows", "ages", "decays"
   
2. Determine update type:
   - Continuous: smooth changes each frame
   - Discrete: periodic updates
   - Event-based: triggered by conditions
   
3. Estimate rates:
   - "slowly" → 0.001 to 0.01 change per frame
   - "gradually" → 0.01 to 0.05 change per frame
   - "quickly" → 0.05 to 0.1 change per frame
   - "rapidly" → 0.1+ change per frame

4. Identify behavioral coupling:
   - How does the state affect particle behavior?
   - What are the thresholds for behavioral changes?
   - Are there critical points (death, transformation)?

## Example Synthesis Outputs

### Input: "particles lose energy over time"
```json
{
  "state_name": "energy",
  "temporal_update": {
    "update_expression": "energy *= 0.995",
    "update_condition": null,
    "update_frequency": 1,
    "affects_behavior": "if energy < 20.0: vel *= 0.8",
    "coupling_strength": 0.8
  }
}
```

### Input: "predators get tired from hunting and must rest"
```json
{
  "state_name": "stamina",
  "temporal_update": {
    "update_expression": "stamina -= 1.0 if hunting else min(100, stamina + 2.0)",
    "update_condition": "if species == 0",
    "update_frequency": 1,
    "affects_behavior": "if stamina < 10.0: hunting = False; vel *= 0.5",
    "coupling_strength": 1.0
  }
}
```

### Input: "cells age and die after 100 cycles"
```json
{
  "state_name": "age",
  "temporal_update": {
    "update_expression": "age += 1",
    "update_condition": null,
    "update_frequency": 1,
    "affects_behavior": "if age >= 100: active = 0.0",
    "coupling_strength": 1.0
  }
}
```
"""