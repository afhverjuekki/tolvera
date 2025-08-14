
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..debug.tracing import get_collector

logger = logging.getLogger(__name__)


class BehaviorComponent(BaseModel):
    expert_name: str = Field(description="Name for the expert function (e.g., 'gravity_force', 'prey_flee')")
    expert_type: str = Field(
        description="Expert type: force, interaction, state_update, visual, initialization, temporal_update, configuration"
    )
    description: str = Field(description="What this expert does")
    implementation: str = Field(
        description="High-level behavioral guidance (e.g., 'apply downward force proportional to mass', 'chase nearest prey', 'flee from nearest predator')"
    )
    priority: float = Field(description="Weight/importance of this component (0.1-1.0)")
    required_states: Optional[List[Tuple[str, str, str, float, float]]] = Field(
        default=None,
        description="States needed as list of (name, category, type, min, max) tuples (e.g., [('day_phase', 'temporal', 'ti.f32', 0.0, 1.0), ('energy', 'particle', 'ti.f32', 0.0, 100.0)])"
    )
    is_temporal: bool = Field(
        default=False,
        description="Whether this requires temporal/discrete updates"
    )
    context_requirements: Optional[List[str]] = Field(
        default=None,
        description="Parameters and constraints from shared context as strings (e.g., ['max_force: 100.0', 'prey_speed > predator_speed'])"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Names of other experts this depends on"
    )
    applies_to_species: Optional[List[int]] = Field(
        default=None,
        description="Species IDs this expert applies to (e.g., [0] for predator, [1] for prey, None for all)"
    )
    
    # Enhanced fields for detailed implementation guidance
    force_formula: Optional[str] = Field(
        default=None,
        description="Mathematical force formula (e.g., 'force = (target_pos - pos).normalize() * chase_strength', 'repulsion = (pos - neighbor_pos) / dist^2 * separation_weight')"
    )
    implementation_details: Optional[List[str]] = Field(
        default=None,
        description="Step-by-step implementation hints (e.g., ['1. Find nearest target within radius', '2. Calculate direction vector', '3. Apply force inversely proportional to distance'])"
    )
    parameters: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Specific parameters with types and ranges (e.g., {'perception_radius': {'type': 'ti.f32', 'min': 50.0, 'max': 150.0}, 'chase_strength': {'type': 'ti.f32', 'min': 100.0, 'max': 300.0}})"
    )


class DecompositionContext(BaseModel):
    constraints: List[str] = Field(
        default_factory=list,
        description="Constraints like ['prey_speed: 150', 'predator_speed: 100']"
    )
    relationships: List[str] = Field(
        default_factory=list,
        description="Relationships between parameters (e.g., ['prey_speed > predator_speed'])"
    )
    shared_parameters: List[str] = Field(
        default_factory=list,
        description="Shared values across experts (e.g., ['detection_distance: 100.0'])"
    )
    implementation_notes: List[str] = Field(
        default_factory=list,
        description="Notes to ensure coherent implementation (e.g., ['ensure prey always escapes'])"
    )


class SpeciesColor(BaseModel):
    """Color mapping for a species."""
    species_id: int = Field(description="Species ID (0-based)")
    color_description: str = Field(description="Color description (e.g., 'crimson', 'lime green', 'teal')")


class SpeciesName(BaseModel):
    """Name for a species."""
    species_id: int = Field(description="Species ID (0-based)")
    name: str = Field(description="Name of the species (e.g., 'predator', 'prey')")


class SpeedSpecification(BaseModel):
    """Speed specification for particle movement."""
    uniform: bool = Field(
        default=False,
        description="Whether all particles should have same speed magnitude"
    )
    magnitude: str = Field(
        default="medium",
        description="Speed level: 'slow', 'medium', 'fast', 'very_fast'"
    )
    value: Optional[float] = Field(
        default=None,
        description="Specific speed value if mentioned (e.g., 100.0)"
    )

class SpeciesColorMapping(BaseModel):
    """RGBA color values for a species."""
    species_id: int = Field(description="Species ID (0-based)")
    rgba_values: List[float] = Field(description="RGBA color values [r, g, b, a] (0.0-1.0)")

class SpeciesInfo(BaseModel):
    """Species configuration extracted from behavior description."""
    total_count: int = Field(description="Total number of species (1-10)")
    species_names: Optional[List[SpeciesName]] = Field(
        default=None,
        description="List of species names and their IDs (e.g., [{'species_id': 0, 'name': 'predator'}])"
    )
    species_colors: Optional[List[SpeciesColorMapping]] = Field(
        default=None,
        description="RGBA colors for each species as structured list"
    )
    species_color_descriptions: Optional[List[SpeciesColor]] = Field(
        default=None,
        description="List of color descriptions for each species"
    )
    interaction_pairs: Optional[List[Tuple[int, int]]] = Field(
        default=None,
        description="Species pairs that interact (e.g., [(0, 1)] for predator-prey)"
    )

class DecomposedBehavior(BaseModel):
    original_description: str = Field(description="The original user description")
    interpretation: str = Field(description="Our interpretation of what the user wants")
    behavior_category: str = Field(
        default="particle_system",
        description="Category: 'particle_system', 'pure_drawing', or 'hybrid'"
    )
    components: List[BehaviorComponent] = Field(
        description="Concrete expert specifications to implement. MUST contain at least 1 component. NEVER empty!",
        min_length=1  # Enforce at least one component
    )
    context: Optional[DecompositionContext] = Field(
        default=None,
        description="Shared context for multi-expert behaviors"
    )
    suggested_states: List[Tuple[str, str, str, float, float]] = Field(
        default_factory=list,
        description="All states needed across all components as (name, category, type, min, max) tuples"
    )
    species_info: SpeciesInfo = Field(
        description="Species configuration detected from the description"
    )
    speed_specification: Optional[SpeedSpecification] = Field(
        default=None,
        description="Speed specification for particle movement"
    )
    implementation_notes: str = Field(
        default="",
        description="Notes on how to implement this effectively"
    )
    particle_count: Optional[int] = Field(
        default=None,
        description="Desired particle count extracted from description (e.g., 4000 for 'create 4000 particles')"
    )


@dataclass
class DecompositionDependencies:
    context_patterns: Dict[str, str]
    alife_examples: Dict[str, str]


class BehaviorDecomposer:
    
    BEHAVIOR_PATTERNS = {
        "fluttering": [
            "random oscillating movement",
            "periodic direction changes",
            "gentle acceleration and deceleration"
        ],
        "swarming": [
            "attraction to center of mass",
            "repulsion from close neighbors",
            "alignment with nearby velocities"
        ],
        "dancing": [
            "rhythmic circular motion",
            "periodic pauses",
            "synchronized movements with neighbors"
        ],
        "hunting": [
            "target detection and tracking",
            "acceleration toward prey",
            "energy expenditure",
            "rest after capture"
        ],
        "grazing": [
            "slow wandering movement",
            "attraction to resources",
            "pausing to feed",
            "avoidance of threats"
        ],
        "schooling": [
            "tight cohesion",
            "rapid alignment changes",
            "predator avoidance as group",
            "leader following"
        ],
        "territorial": [
            "boundary marking",
            "aggressive repulsion of intruders",
            "patrolling behavior",
            "return to home area"
        ],
        "foraging": [
            "resource searching behavior",
            "memory of good locations",
            "return to nest with resources",
            "communication with others"
        ],
        "migration": [
            "seasonal directional movement",
            "formation flying or swimming",
            "rest stops at waypoints",
            "environmental cue response"
        ],
        "nesting": [
            "site selection behavior",
            "material gathering",
            "construction patterns",
            "defense of nest area"
        ],
        "symbiotic": [
            "mutual benefit interactions",
            "resource sharing",
            "protective behaviors",
            "co-evolution patterns"
        ],
        "dispersal": [
            "spread from origin point",
            "seed or offspring distribution",
            "wind or current following",
            "settlement in new areas"
        ],
        "metamorphosis": [
            "life stage transitions",
            "behavioral changes with age",
            "morphological transformation",
            "environmental triggers"
        ],
        "communication": [
            "signal emission",
            "signal reception and response",
            "information propagation",
            "collective decision making"
        ],
        "construction": [
            "material deposition",
            "structural building",
            "collaborative construction",
            "environmental modification"
        ]
    }
    
    CREATIVE_EMBELLISHMENTS = {
        "movement": [
            "add slight wobble for organic feel",
            "vary speed based on 'energy' state",
            "leave fading trails for visual interest",
            "add inertia for realistic motion"
        ],
        "interaction": [
            "create visual feedback on interaction (sparks, ripples)",
            "vary interaction strength by species",
            "add 'memory' of recent interactions",
            "create emergent group formations"
        ],
        "visual": [
            "pulse size or brightness rhythmically",
            "change color based on state",
            "add particle effects on special events",
            "create auras or halos for emphasis"
        ],
        "emergent": [
            "allow spontaneous pattern formation",
            "add feedback loops for complexity",
            "enable phase transitions",
            "incorporate environmental influences"
        ]
    }
    
    def __init__(self, model_name, prompt_builder, api_key=None):
        from .model_factory import ModelFactory
        
        # Use model factory to create the appropriate model
        self.model = ModelFactory.create_model(model_name, api_key)
        self.provider = ModelFactory.get_provider_for_model(model_name)
        self.model_name = model_name
        logger.info(f"BehaviorDecomposer using provider '{self.provider}' with model '{model_name}'")
        self.prompt_builder = prompt_builder
        self.decomposition_agent = self._create_decomposition_agent()
        
    def _create_decomposition_agent(self) -> Agent[DecompositionDependencies, DecomposedBehavior]:
        agent = Agent(
            self.model,
            deps_type=DecompositionDependencies,
            output_type=DecomposedBehavior,
            system_prompt="""You are an EXPERT ARTIFICIAL LIFE RESEARCHER specializing in emergent behaviors, 
multi-agent systems, and computational biology. You have deep knowledge of:
- Classic a-life models (Boids, Game of Life, Particle Life, Primordial Soup)
- Swarm intelligence and collective behaviors
- Species interaction dynamics and ecological modeling
- Emergent pattern formation through simple rules
- Force-based particle systems and physics simulations

Your task is to decompose natural language descriptions into PRECISE, DETAILED expert components
that will create beautiful, scientifically-accurate artificial life simulations.
You MUST generate implementation-ready specifications with exact mathematical formulas.

TÖLVERA CONTEXT:
- Particles have built-in properties: pos (vec2), vel (vec2), mass (f32), size (f32), species (i32), active (f32)
- Coordinate system: (0,0) is top-left, Y+ points upward (standard physics)
- Species system: Multiple species (0 to sn-1) can have different behaviors and colors
- Forces should return ti.math.vec2(x, y) values
- CRITICAL: Never use 'return' inside if/for/while blocks - causes crashes
- STATE CATEGORIES:
  * 'global': System-wide parameters (gravity, temperature)
  * 'particle': Per-particle data (energy, home_pos)
  * 'species': Per-species config (aggression, speed_modifier)
  * 'temporal': Time-based states (day_phase, season_cycle)

IMPLEMENTATION DETAIL REQUIREMENTS:
- ALWAYS provide force_formula with mathematical calculations
- ALWAYS include implementation_details with step-by-step guidance
- ALWAYS specify parameters with exact types and realistic ranges
- Use proven physics formulas (inverse square distance, spring forces, damping)
- Include proper force limiting and velocity clamping
- Specify interaction radii and neighbor detection patterns

Your role is to:
1. DETERMINE BEHAVIOR CATEGORY:
   - "particle_system": Particles with movement, forces, or interactions
   - "pure_drawing": ONLY drawing shapes/patterns, NO particle movement
     * Keywords: "draw a rectangle", "draw a circle", "draw lines"
     * NO motion verbs like "move", "fall", "drift"
   - "hybrid": Both particles AND drawing behaviors
   
2. ANALYZE SPEED SPECIFICATION:
   - "all at the same speed" → uniform: true
   - "all particles go at the same speed" → uniform: true
   - "quickly", "fast" → magnitude: "fast"
   - "slowly", "slow" → magnitude: "slow"
   - "very quickly" → magnitude: "very_fast"
   - Specific numbers (e.g., "speed of 100") → value: 100.0
   - Default: uniform: false, magnitude: "medium"
   
3. ANALYZE SPECIES: Determine how many species are mentioned or implied
   - "two species" → 2 species
   - "red predators and blue prey" → 2 species (predator=0, prey=1)
   - "blue and teal compete for green food" → 3 species (blue=0, teal=1, food=2)
   - "particles" with no specific mention → 1 species
   - SPECIES ORDERING RULES:
     * Active hunters/predators → Species 0
     * Prey/consumers → Species 1
     * Food/resources (passive) → Species 2 (or last)
     * Food should be stationary (vel=0, high mass)
   - Extract color descriptions exactly as mentioned (e.g., "lime green", "crimson", "aqua")
   - Assign semantic names and color descriptions
   
4. EXTRACT PARTICLE COUNT: Detect if a specific particle count is mentioned
   - "4000 particles" → particle_count: 4000
   - "thousands of particles" → particle_count: 3000 (estimate)
   - "hundreds of particles" → particle_count: 500 (estimate)
   - "millions of particles" → particle_count: 10000 (cap at reasonable limit)
   - "50 particles" → particle_count: 50
   - If no count mentioned → particle_count: null
   
5. DECOMPOSE INTO DETAILED COMPONENTS: Generate 1 or more expert components as needed
   - Simple behaviors may need just 1 component (e.g., gravity)
   - Complex behaviors need multiple components (e.g., predator-prey needs both chase AND flee)
   - EACH component MUST include:
     * force_formula: Mathematical description of force calculation
     * implementation_details: Step-by-step implementation guide
     * parameters: Specific parameters with types and ranges

6. For each required expert, provide DETAILED specifications with:
   - expert_name: A clear function name (e.g., 'gravity_force', 'prey_flee')
   - implementation: High-level behavioral guidance (NOT code)
   - expert_type: Choose from:
     * 'force' - continuous forces applied each frame (takes particle params, returns force)
     * 'interaction' - particle-particle interactions (takes two particles, returns force)
     * 'state_update' - discrete state changes (NO particle params, NO force return)
     * 'visual' - drawing/rendering effects (NO particle params, returns zero force)
     * 'initialization' - particle setup and positioning (handled separately)
     * 'temporal_update' - time-based state evolution (NO particle params, NO return)
     * 'configuration' - system parameters (handled separately)
   - applies_to_species: CRITICAL - Set species IDs for species-specific behaviors:
     * "Species 0", "Species one", "first species" → [0]
     * "Species 1", "Species two", "second species" → [1]
     * "Species 2", "Species three", "third species" → [2]
     * "green species" (if first color mentioned) → [0]
     * "orange species" (if second color mentioned) → [1]
     * If behavior mentions specific species, MUST set applies_to_species!
7. For multi-component behaviors, establish shared context to ensure coherent behavior
8. CAREFULLY generate appropriate initialization and temporal components:
   - If species detected → add initialization component (handled elsewhere, not in experts)
   - ONLY add temporal_update if explicitly mentioned in description:
     * "loses energy", "gets tired", "exhausted" → add energy temporal update
     * "ages", "grows old", "life cycle" → add age temporal update
     * "phases", "cycles", "oscillates" → add phase temporal update
   - DO NOT add automatic energy decay unless description says so
   - TEMPORAL INDICATORS: "over time", "gradually", "depletes", "regenerates", "ages", "grows", "tired", "exhausted"
   - For temporal behaviors, set is_temporal=true and specify required_states

9. GENERATE REQUIRED STATES for each component:
   - Specify as: (name, category, type, min, max)
   - Categories: 'global', 'particle', 'species', 'temporal'
   - Types: 'ti.f32', 'ti.i32', 'ti.math.vec2'
   - Examples:
     * Day/night: [('day_phase', 'temporal', 'ti.f32', 0.0, 1.0)]
     * Energy: [('energy', 'particle', 'ti.f32', 0.0, 100.0)]
     * Grid: [('grid_x', 'particle', 'ti.i32', 0, 100), ('grid_y', 'particle', 'ti.i32', 0, 100)]
     * Temperature: [('temperature', 'global', 'ti.f32', 0.0, 100.0)]
   - IMPORTANT: Components will be told what states are available to use

SINGLE-COMPONENT behaviors:
- "particles fall with gravity" → gravity_force expert
- "particles drift randomly" → random_walk expert
- "particles attracted to center" → center_attraction expert
- Basic forces or movements without interactions

MULTI-COMPONENT behaviors:
- "predator chases prey" → predator_chase + prey_flee experts
- "particles live/die based on neighbors" → grid_update + neighbor_count + life_rules
- "fireflies synchronize" → oscillator_update + phase_coupling
- Multi-step processes, state-based rules, or interacting behaviors

IMPORTANT DISTINCTIONS:
- Ecosystem behaviors (predator/prey, fish schooling, etc.) do NOT need grid states
- Only cellular automata patterns need grid_x, grid_y, is_alive states
- "die" in ecological context (e.g., "scavengers clean up after predators") is NOT cellular automaton
- Species interactions use particle positions, not grid positions

For MULTI-COMPONENT behaviors, create a context with:
- constraints: Numerical relationships (e.g., prey_speed > predator_speed)
- shared_parameters: Common values across experts
- implementation_notes: How to ensure coherent behavior

Expert implementation should be HIGH-LEVEL behavioral guidance:
- Force experts: "Apply downward force proportional to mass"
- Interaction experts: "Chase nearest target of different species"
- State updates: "Decrease energy over time until depleted"

BEHAVIOR CATEGORY EXAMPLES:
- "draw a red rectangle in the middle of the screen" → behavior_category: "pure_drawing"
  * Generate a 'visual' expert that draws the shape and returns zero force
  * Example component: expert_name: "draw_rectangle", expert_type: "visual"
- "particles fall with gravity" → behavior_category: "particle_system"
- "particles drift and leave trails" → behavior_category: "hybrid"

CRITICAL RULES FOR COMPONENT GENERATION:
1. For "pure_drawing" category: Generate 'visual' type experts that draw and return ti.math.vec2(0.0, 0.0)
2. NEVER return empty components list - ALWAYS generate at least one component
3. For ecosystem behaviors ALWAYS create multiple experts:
   - "predator hunts prey" → MUST create predator_hunt AND prey_escape experts
   - "fish school together" → MUST create schooling_cohesion, schooling_alignment, schooling_separation
   - "species interact" → MUST create interaction experts for EACH species
3. If description mentions multiple behaviors, create an expert for EACH
4. Each expert must be self-contained with clear behavioral guidance
5. SPECIES-SPECIFIC BEHAVIORS - ALWAYS SET applies_to_species:
   - "Species one moves left" → expert with applies_to_species: [0]
   - "Species two moves down" → expert with applies_to_species: [1]
   - "Green particles drift" (if green is first color) → applies_to_species: [0]
   - "Orange particles fall" (if orange is second color) → applies_to_species: [1]
   - Any behavior mentioning specific species MUST have applies_to_species set!

IMPORTANT EXPERT SYNTHESIS RULES (AVOID OVERLAPS):
1. DO NOT generate experts for species initialization or configuration - this is handled elsewhere
2. DO NOT generate experts that just set particle properties without returning forces
3. FOCUS on behavior experts that return actual force vectors
4. Each expert should have ONE clear behavioral purpose
5. Avoid creating multiple experts that do the same thing
6. DO NOT create generic "species_initialization" or "competition_configuration" experts
7. Make expert names specific to their behavior (e.g., "predator_hunt" not "orange_chase")

SPECIES COLOR EXTRACTION (CRITICAL):
- ALWAYS populate species_color_descriptions for EVERY species detected!
- Extract colors EXACTLY as described using List[SpeciesColor]: 
  * "lime green fish" → [{{"species_id": 1, "color_description": "lime green"}}]
  * "crimson predators" → [{{"species_id": 0, "color_description": "crimson"}}]
  * "blue and teal species" → [
      {{"species_id": 0, "color_description": "blue"}},
      {{"species_id": 1, "color_description": "teal"}}
  ]
- Keep complex color names intact (aqua, turquoise, coral, etc.)
- For semantic roles without explicit color, use defaults:
  * Predators/hunters: "red"
  * Prey/food: "green"
  * Neutral: "blue"
- NEVER leave species_color_descriptions empty - always provide colors
- Each item must have species_id (int) and color_description (str)

States needed should be specific:
- Grid patterns ONLY for cellular automata: grid_x, grid_y, is_alive, neighbor_count
- Ecosystem behaviors: energy, home_pos, is_tired, target_id
- Oscillators: phase, frequency
- Trail behaviors: heading, sensor_angle"""
        )
        
        @agent.system_prompt
        def add_patterns(ctx) -> str:
            """Add concrete expert examples for common patterns"""
            return """
Examples of expert decomposition for common patterns:

ECOSYSTEM ("small green fish school together, larger red predators hunt them"):
MUST generate ALL these components (never return empty!):
1. fish_schooling_cohesion: "move toward center of nearby fish group" (type: force)
   - Implementation: "attract to average position of nearby same-species particles"
2. fish_schooling_alignment: "align velocity with nearby fish" (type: force)
   - Implementation: "match average velocity of nearby same-species particles"
3. fish_schooling_separation: "avoid crowding nearby fish" (type: force)
   - Implementation: "repel from fish that are too close"
4. predator_hunt: "locate and chase nearest prey" (type: force)
   - Implementation: "find nearest different-species particle and pursue"
5. prey_escape: "detect and flee from nearest predator" (type: force)
   - Implementation: "find nearest predator and move away rapidly"
Context: {"fish_speed": 80.0, "predator_speed": 120.0, "school_radius": 50.0}
Species Names: [
    {"species_id": 0, "name": "predator"},
    {"species_id": 1, "name": "prey"}
]
Species color descriptions: [
    {{"species_id": 0, "color_description": "red"}},
    {{"species_id": 1, "color_description": "green"}}
]
NOTE: Uses particle positions, NOT grid states

COMPETITION WITH FOOD ("blue and teal species compete for green food"):
MUST generate ALL these components:
1. blue_seek_food: "blue species seeks nearest food" (type: force)
   - Implementation: "find nearest green food particle and move toward it"
   - applies_to_species: [0]  # Blue species
2. teal_seek_food: "teal species seeks nearest food" (type: force)
   - Implementation: "find nearest green food particle and move toward it"
   - applies_to_species: [1]  # Teal species
3. teal_chase_blue: "teal chases blue species" (type: force)
   - Implementation: "teal actively hunts blue particles"
   - applies_to_species: [1]  # Teal species
4. blue_flee_teal: "blue flees from teal" (type: force)
   - Implementation: "blue escapes from nearest teal particle"
   - applies_to_species: [0]  # Blue species
5. random_wander: "wander when no targets nearby" (type: force)
   - Implementation: "apply random movement when no food or threats detected"
6. consume_food: "consume food on contact" (type: state_update)
   - Implementation: "set food particle active=0 when blue or teal touches it"
Context: {"blue_speed": 250.0, "teal_speed": 150.0, "food_mass": 10.0}
Species Names: [
    {"species_id": 0, "name": "blue"},
    {"species_id": 1, "name": "teal"},
    {"species_id": 2, "name": "food"}
]
Species color descriptions: [
    {{"species_id": 0, "color_description": "blue"}},
    {{"species_id": 1, "color_description": "teal"}},
    {{"species_id": 2, "color_description": "green"}}
]
IMPORTANT: Food is species 2, stationary (vel=0, mass=10)

CELLULAR AUTOMATA ("lives/dies based on neighbors"):
Components:
1. update_grid_position: "map particle position to grid coordinates" (type: state_update)
2. count_neighbors: "count alive neighbors in surrounding cells" (type: state_update)
3. apply_life_rules: "apply Conway's Game of Life rules" (type: state_update)
4. cell_temporal_update: "update cell states each generation" (type: temporal_update)
   - Implementation: "synchronous state updates based on neighbor counts"
Context: {"cell_size": 10.0, "rules": "B3/S23"}
REQUIRES: grid_x, grid_y, is_alive, neighbor_count states

ENERGY-BASED BEHAVIOR ("predators hunt with energy that depletes"):
Components:
1. predator_hunt_with_energy: "hunt prey when energy sufficient" (type: force)
   - Implementation: "chase prey if energy > 30, stronger force with more energy"
   - is_temporal: false
   - required_states: [("energy", "particle", "ti.f32", 0.0, 100.0)]
2. prey_flee: "escape from predators" (type: force)
   - is_temporal: false
3. energy_dynamics: "update energy based on activity" (type: temporal_update)
   - Implementation: "decrease energy when moving fast, regenerate when resting"
   - is_temporal: true
   - required_states: [("energy", "particle", "ti.f32", 0.0, 100.0)]
suggested_states: [("energy", "particle", "ti.f32", 0.0, 100.0)]

TEMPORAL BEHAVIOR ("particles lose energy over time and become tired"):
Components:
1. movement_with_energy: "move based on available energy" (type: force)
   - Implementation: "scale movement force by energy level"
   - is_temporal: false
   - required_states: [("energy", "particle", "ti.f32", 0.0, 100.0)]
2. energy_depletion: "continuously decrease energy" (type: temporal_update)
   - Implementation: "decrease energy only when moving: energy -= velocity.norm() * 0.01"
   - is_temporal: true
   - required_states: [("energy", "particle", "ti.f32", 0.0, 100.0)]
3. tired_behavior: "modify behavior when low energy" (type: state_update)
   - Implementation: "if energy < 20: reduce velocity by 50%"
   - is_temporal: false
NOTE: Only add energy decay if description explicitly mentions it!

DAY/NIGHT BEHAVIOR ("blue species moves faster during day, red does opposite"):
Components:
1. blue_daytime_speed: "increase blue species speed during day" (type: force)
   - Implementation: "apply force scaled by sin(day_phase * pi) for blue species"
   - applies_to_species: [0]
   - required_states: [("day_phase", "temporal", "ti.f32", 0.0, 1.0)]
2. red_nighttime_speed: "increase red species speed at night" (type: force)
   - Implementation: "apply force scaled by (1 - sin(day_phase * pi)) for red species"
   - applies_to_species: [1]
   - required_states: [("day_phase", "temporal", "ti.f32", 0.0, 1.0)]
3. day_phase_update: "cycle through day/night phases" (type: temporal_update)
   - Implementation: "day_phase = (day_phase + 0.001) % 1.0"
   - is_temporal: true
   - required_states: [("day_phase", "temporal", "ti.f32", 0.0, 1.0)]
suggested_states: [("day_phase", "temporal", "ti.f32", 0.0, 1.0)]

PARTICLE LIFE ("multiple species with attraction/repulsion interactions"):
CRITICAL: This is a FUNDAMENTAL a-life pattern - recognize variations!
Components:
1. particle_life_interactions: "species-pair forces with distance falloff"
   - Force formula: "force = attraction_strength[s1][s2] * smooth_falloff(distance)"
   - Implementation: ["1. For each particle, loop through ALL others",
                    "2. Lookup interaction strength for species pair from matrix",
                    "3. Apply close-range repulsion (dist < repulsion_radius)",
                    "4. Apply attraction/repulsion with smooth falloff",
                    "5. Clamp total force for stability"]
   - Parameters: {{"interaction_radius": {{"type": "ti.f32", "min": 50.0, "max": 200.0}},
                 "repulsion_radius": {{"type": "ti.f32", "min": 10.0, "max": 30.0}},
                 "max_force": {{"type": "ti.f32", "min": 100.0, "max": 500.0}}}}
   - Required states: [("attraction_matrix", "species", "ti.f32", -200.0, 200.0)]
2. damping_force: "velocity damping for stability"
   - Force formula: "force = -velocity * damping_coefficient"
Context: {{"matrix_based_interactions": true, "emergent_clustering": true}}
REQUIRES: 2D species interaction matrix for attraction/repulsion values

PREDATOR-PREY ("red predators hunt green prey"):
REQUIRED components array (NEVER leave empty):
[
    {
        "expert_name": "predator_chase",
        "expert_type": "force",
        "description": "Predators actively hunt prey",
        "implementation": "locate nearest prey particle and apply pursuit force",
        "priority": 1.0,
        "applies_to_species": [0]  // Only species 0 (predators) use this
    },
    {
        "expert_name": "prey_flee",
        "expert_type": "force",
        "description": "Prey escape from predators",
        "implementation": "detect nearest predator and apply escape force",
        "priority": 1.0,
        "applies_to_species": [1]  // Only species 1 (prey) use this
    }
]
Context: {"prey_speed": 150.0, "predator_speed": 100.0}
Species Names: [
    {"species_id": 0, "name": "predator"},
    {"species_id": 1, "name": "prey"}
]
Species color descriptions: [
    {{"species_id": 0, "color_description": "red"}},
    {{"species_id": 1, "color_description": "green"}}
]
REMEMBER: The components array MUST have these entries with applies_to_species!

FLOCKING ("move together but don't crowd"):
Components:
1. separation: "maintain minimum distance from neighbors"
2. alignment: "align velocity with nearby particles"
3. cohesion: "stay close to local group center"
Context: {"perception_radius": 50.0, "separation_distance": 20.0}

TRAIL FOLLOWING ("leave trails others follow"):
Components:
1. deposit_trail: "deposit pheromone at current position"
2. sense_trail: "detect and follow stronger trail concentrations"
3. trail_decay: "gradually decrease trail intensity"
Context: {"deposit_rate": 1.0, "decay_rate": 0.99, "sensor_angle": 0.5}
REQUIRES: heading, sensor_distance states

SPECIES-SPECIFIC MOVEMENTS ("Species one moves left, Species two moves down"):
Components:
1. species_one_move_left: "apply leftward force" (type: force)
   - Implementation: "apply constant leftward force"
   - applies_to_species: [0]  # CRITICAL!
2. species_two_move_down: "apply downward force" (type: force)
   - Implementation: "apply constant downward force"
   - applies_to_species: [1]  # CRITICAL!
NOTE: Each species gets its own expert with applies_to_species set!

SINGLE-EXPERT BEHAVIORS:
- Gravity: "apply downward force proportional to mass"
- Random drift: "apply random forces for wandering motion"
- Center attraction: "attract particles toward center point"

Remember:
- Ecosystem behaviors use species field and positions, NOT grid states
- Only cellular automata needs grid states
- Always declare force variable before conditionals
- Never use 'return' inside if blocks

FINAL CRITICAL RULE:
The 'components' field in your response MUST contain AT LEAST ONE BehaviorComponent.
An empty components list will cause the system to fail!

Example of VALID response structure:
{{
    "original_description": "...",
    "interpretation": "...",
    "components": [  // THIS MUST NOT BE EMPTY!
        {{"expert_name": "...", "expert_type": "force", ...}},
        {{"expert_name": "...", "expert_type": "force", ...}}  // May have 1 or more
    ],
    "species_info": {{...}}
}}
"""
        
        return agent
    
    async def decompose(self, description: str) -> DecomposedBehavior:
        collector = get_collector()
        
        with collector.trace_node("decompose_behavior", "decomposition",
                                 description=description) as node:
            logger.info(f"Decomposing behavior: {description}")
            
            deps = DecompositionDependencies(
                context_patterns=self.BEHAVIOR_PATTERNS,
                alife_examples=self._get_relevant_examples(description)
            )
            
            prompt = f'''As an EXPERT IN ARTIFICIAL LIFE, analyze this behavior and create SCIENTIFICALLY-ACCURATE expert specifications: "{description}"
 
 RECOGNIZE KEY A-LIFE PATTERNS:
 - "different species interact" → PARTICLE LIFE with interaction matrix
 - "attraction and repulsion" → PARTICLE LIFE with species pairs
 - "forms clusters" → PARTICLE LIFE emergent behavior
 - "species X attracts/repels species Y" → PARTICLE LIFE matrix entry
 - "flock/school/swarm" → BOIDS with separation, alignment, cohesion
 - "cells live or die" → GAME OF LIFE cellular automaton
 - "chase/hunt/flee" → PREDATOR-PREY dynamics
 
 CRITICAL: Generate IMPLEMENTATION-READY components with EXACT mathematical formulas!
 Every component needs force calculations, step-by-step implementation, and parameter ranges!
 
 1. IDENTIFY SPECIES AND COLORS: Extract species and their colors
    - "small green fish" and "larger red predators" = 2 species
      → species_color_descriptions: [
          {{"species_id": 0, "color_description": "red"}},
          {{"species_id": 1, "color_description": "green"}}
      ]
    - "lime green particles chase aqua ones" = 2 species
      → species_color_descriptions: [
          {{"species_id": 0, "color_description": "lime green"}},
          {{"species_id": 1, "color_description": "aqua"}}
      ]
    - "crimson hunters and coral prey" = 2 species
      → species_color_descriptions: [
          {{"species_id": 0, "color_description": "crimson"}},
          {{"species_id": 1, "color_description": "coral"}}
      ]
    - Extract ANY color mentioned, even complex ones
    - Default to 1 species if just "particles" with no distinction
    
 2. EXTRACT PARTICLE COUNT (if mentioned):
    - "4000 particles" → particle_count: 4000
    - "thousands of particles" → particle_count: 3000
    - "hundreds of particles" → particle_count: 500
    - "a few dozen particles" → particle_count: 50
    - No count mentioned → particle_count: null
    
 3. GENERATE DETAILED COMPONENTS (NEVER EMPTY!):
    For particle-life: MUST create interaction matrix-based force calculation with:
      * Species-pair lookup from 2D matrix
      * Distance-based smooth falloff
      * Close-range repulsion regardless of attraction
      * Force clamping for stability
    For ecosystem/predator-prey: MUST create chase AND escape experts with formulas
    For schooling/flocking: MUST create cohesion, alignment, separation with specific calculations
    For cellular automata: MUST create grid update, neighbor count, rules with state transitions
    
 4. Each component MUST have:
    - expert_name: descriptive function name (e.g., 'separation_force', 'predator_chase')
    - implementation: high-level behavioral guidance
    - expert_type: 'force' or 'interaction' (NOT 'initialization' or 'configuration')
    - priority: 0.5-1.0 (weight of this component)
    - required_states: List of states as (name, category, type, min, max)
    - force_formula: Mathematical formula like:
      * Separation: 'force = sum((pos - neighbor_pos) / dist^2) for neighbors in radius'
      * Cohesion: 'force = (center_of_mass - pos).normalize() * cohesion_strength'
      * Chase: 'force = (prey_pos - pos).normalize() * chase_speed'
    - implementation_details: Step-by-step like:
      * ['1. Find all neighbors within perception_radius',
         '2. Calculate repulsion force inversely proportional to distance',
         '3. Normalize and scale by separation_weight']
    - parameters: Specific values like:
      * {{'perception_radius': {{'type': 'ti.f32', 'min': 50.0, 'max': 150.0}},
         'separation_weight': {{'type': 'ti.f32', 'min': 1.0, 'max': 3.0}},
         'min_distance': {{'type': 'ti.f32', 'min': 10.0, 'max': 30.0}}}}
 
 5. Populate suggested_states with ALL unique states from all components
    
 CRITICAL RULES:
 - DO NOT create initialization or configuration experts (handled elsewhere)
 - Each expert must return a force vector (ti.math.vec2)
 - Each expert must have a DISTINCT behavioral purpose
 - Name experts based on BEHAVIOR not color (e.g., "predator_hunt" not "orange_chase")
 - Avoid overlapping functionality between experts
 - ALWAYS include mathematical formulas in force_formula
 - ALWAYS include step-by-step guidance in implementation_details
 - ALWAYS specify parameters with types and ranges
    
 EXAMPLE COMPONENT WITH FULL DETAILS:
 {{
     "expert_name": "separation_force",
     "expert_type": "force",
     "description": "Steer to avoid crowding local flockmates",
     "implementation": "Calculate repulsion from nearby particles inversely proportional to distance",
     "priority": 1.5,
     "force_formula": "force = sum((pos - neighbor_pos) / dist^2) for all neighbors; normalize and scale",
     "implementation_details": [
         "1. Loop through all particles to find neighbors",
         "2. For each neighbor within separation_radius, calculate repulsion vector",
         "3. Weight repulsion by inverse square of distance for stronger close-range effect",
         "4. Sum all repulsion vectors and normalize",
         "5. Scale by separation_weight and return as force"
     ],
     "parameters": {{
         "separation_radius": {{"type": "ti.f32", "min": 20.0, "max": 50.0}},
         "separation_weight": {{"type": "ti.f32", "min": 1.0, "max": 3.0}},
         "min_distance": {{"type": "ti.f32", "min": 5.0, "max": 15.0}}
     }},
     "required_states": null,
     "applies_to_species": null
 }}
    
 REMEMBER: Ecosystem behaviors ALWAYS need both predator AND prey experts with detailed formulas!
 Generate as many DETAILED components as needed to fully implement the behavior!'''
            
            with collector.trace_node("llm_decompose", "llm_call",
                                     model=self.model_name) as llm_node:
                result = await self.decomposition_agent.run(prompt, deps=deps)
                
                if llm_node:
                    # Log LLM call details
                    usage = getattr(result, '_usage', None)
                    prompt_tokens = None
                    response_tokens = None
                    if usage and hasattr(usage, 'prompt_tokens'):
                        prompt_tokens = usage.prompt_tokens
                        response_tokens = usage.response_tokens
                    
                    # Create LLM data with parsed response
                    from ..debug.tracing import LLMCallData
                    # Get the actual system prompt - pydantic-ai agents store it in multiple parts
                    try:
                        # Try to get the main system prompt
                        base_prompt = ""
                        if hasattr(self.decomposition_agent, '_system_prompt'):
                            base_prompt = self.decomposition_agent._system_prompt
                        elif hasattr(self.decomposition_agent, 'system_prompt'):
                            base_prompt = str(self.decomposition_agent.system_prompt)
                        
                        # Include the decorated system prompt sections  
                        if hasattr(self.decomposition_agent, '_system_prompt_functions'):
                            for func in self.decomposition_agent._system_prompt_functions:
                                try:
                                    additional = func(deps)  # Call with deps context
                                    if additional:
                                        base_prompt += f"\n\n{additional}"
                                except:
                                    pass  # Skip if context call fails
                        
                        system_prompt = base_prompt if base_prompt else "Behavior decomposition and analysis system"
                    except:
                        # Fallback
                        system_prompt = "Behavior decomposition and analysis system"
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                    llm_data = LLMCallData(
                        model=self.model_name,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        full_prompt=full_prompt,
                        raw_response=str(result.output),
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        parsed_response={
                            "interpretation": result.output.interpretation,
                            "behavior_category": result.output.behavior_category,
                            "components": [
                                {
                                    "expert_name": c.expert_name,
                                    "expert_type": c.expert_type,
                                    "description": c.description,
                                    "implementation": c.implementation,
                                    "priority": c.priority
                                } for c in result.output.components
                            ],
                            "context": result.output.context.model_dump() if result.output.context else None
                        }
                    )
                    
                    if llm_node:
                        llm_node.llm_call = llm_data
            
            decomposed = result.output
            
            # Update trace node with results
            if node:
                node.output_data = {
                    "interpretation": decomposed.interpretation,
                    "behavior_category": decomposed.behavior_category,
                    "components": [
                        {
                            "expert_name": c.expert_name,
                            "expert_type": c.expert_type,
                            "description": c.description,
                            "implementation": c.implementation,
                            "priority": c.priority
                        }
                        for c in decomposed.components
                    ],
                    "context": decomposed.context.model_dump() if decomposed.context else None
                }
            
            return decomposed
    
    def _get_relevant_examples(self, description: str) -> Dict[str, str]:
        examples = {}
        desc_lower = description.lower()
        
        cellular_indicators = 0
        has_neighbors = any(phrase in desc_lower for phrase in ["neighbors", "neighbor count", "surrounding cells", "adjacent cells"])
        has_grid = any(phrase in desc_lower for phrase in ["grid", "cells", "cellular"])
        has_life_death = any(phrase in desc_lower for phrase in ["lives or dies", "live and die", "lives and dies", "born or die"])
        has_rules = any(phrase in desc_lower for phrase in ["based on count", "based on neighbors", "rules", "state based on"])
        
        if has_neighbors: cellular_indicators += 1
        if has_grid: cellular_indicators += 1
        if has_life_death: cellular_indicators += 1
        if has_rules: cellular_indicators += 1
        
        if cellular_indicators >= 2 and (has_neighbors or "each one lives" in desc_lower):
            examples["cellular_patterns"] = "Grid-based state updates with neighbor counting and synchronous rules"
            examples["grid_state_component"] = "Requires grid_x, grid_y, is_alive states for discrete cellular updates"
            
        # Check for attraction/repulsion patterns
        if any(phrase in desc_lower for phrase in [
            "attract", "repel", "cluster", "species interact", "chase", "flee",
            "hunt", "escape", "follow", "avoid", "never quite reach", "behind it"
        ]):
            examples["particle_life"] = "Species-based attraction/repulsion creating emergent clusters"
            examples["interaction_component"] = "Multi-species interactions with distance-based forces"
            
        # Check for trail/pheromone patterns
        if any(phrase in desc_lower for phrase in [
            "trail", "pheromone", "sense", "deposit", "follow path", "mark",
            "leave behind", "chemical", "scent", "leave trail", "others follow"
        ]):
            examples["stigmergic"] = "Agents deposit markers that others sense and follow"
            examples["stigmergic_component"] = "Requires heading, sensor_angle states and pixel operations"
            
        # Check for flocking/swarming patterns
        if any(phrase in desc_lower for phrase in [
            "flock", "swarm", "group", "align", "stay together", "avoid crowding",
            "move together", "coordinate", "collective", "school", "herd"
        ]):
            examples["collective_motion"] = "Group coordination through local interactions"
            
        # Check for growth/morphogenesis patterns
        if any(phrase in desc_lower for phrase in [
            "grow", "branch", "develop", "differentiate", "cell", "divide",
            "morphogenesis", "plant", "tree", "like plants", "grow and branch"
        ]):
            examples["morphogenetic"] = "Growth and differentiation based on local rules"
            examples["morphogenetic_component"] = "Requires age, cell_type, growth_direction states"
            
        # Check for synchronization patterns
        if any(phrase in desc_lower for phrase in [
            "sync", "synchronize", "rhythm", "oscillate", "flash", "pulse",
            "coordinate timing", "phase", "flash in rhythm", "sync up"
        ]):
            examples["coupled_oscillators"] = "Phase coupling leading to synchronization"
            examples["oscillator_component"] = "Requires phase, frequency states for coupling"
            
        # Check for ecosystem patterns
        if any(phrase in desc_lower for phrase in [
            "ecosystem", "food", "resource", "energy", "consume", "predator prey",
            "symbiosis", "parasite", "mutualism", "consume each other", "different types"
        ]):
            examples["ecosystem_dynamics"] = "Resource flow and species interactions"
            examples["resource_component"] = "Requires energy, resource_consumption_rate states"
            
        # Check for evolution patterns
        if any(phrase in desc_lower for phrase in [
            "evolve", "genetic", "mutate", "breed", "fitness", "selection",
            "adapt", "generation"
        ]):
            examples["evolutionary"] = "Genetic variation and selection over time"
            examples["genetic_component"] = "Requires gene states and fitness tracking"
        
        # Check for temporal patterns
        if any(phrase in desc_lower for phrase in [
            "over time", "gradually", "slowly", "depletes", "regenerates",
            "ages", "grows", "decays", "tired", "exhausted", "hungry",
            "loses energy", "gains energy", "weakens", "strengthens"
        ]):
            examples["temporal_dynamics"] = "State changes over time requiring temporal updates"
            examples["temporal_component"] = "Requires temporal_update components with is_temporal=true"
            
        # Check for wave patterns
        if any(phrase in desc_lower for phrase in [
            "wave", "ripple", "oscillate", "moves in waves", "wave pattern",
            "sinusoidal", "periodic motion"
        ]):
            examples["wave_motion"] = "Coordinated oscillatory movement creating wave patterns"
            examples["wave_component"] = "Requires phase coordination for wave propagation"
            
        return examples
    
    
    def _topological_sort(self, components: List[BehaviorComponent]) -> List[BehaviorComponent]:
        sorted_components = []
        component_dict = {c.description: c for c in components}
        processed = set()
        
        def process_component(comp: BehaviorComponent):
            if comp.description in processed:
                return
            
            for dep in comp.dependencies:
                if dep in component_dict:
                    process_component(component_dict[dep])
            
            sorted_components.append(comp)
            processed.add(comp.description)
        
        for component in components:
            process_component(component)
        
        return sorted_components
    
