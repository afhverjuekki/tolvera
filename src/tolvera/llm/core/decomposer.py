
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..debug.tracing import get_collector

logger = logging.getLogger(__name__)


class BehaviorComponent(BaseModel):
    expert_name: str = Field(description="Name for the expert function (e.g., 'gravity_force', 'prey_flee')")
    expert_type: str = Field(
        description="Expert type: force, interaction, state_update, visual"
    )
    description: str = Field(description="What this expert does")
    implementation: str = Field(
        description="Concrete implementation guidance with specific force calculations or logic"
    )
    priority: float = Field(description="Weight/importance of this component (0.1-1.0)")
    required_states: Optional[List[Tuple[str, str]]] = Field(
        default=None,
        description="States needed as list of (name, category) tuples (e.g., [('grid_x', 'particle'), ('energy', 'particle')])"
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


class DecomposedBehavior(BaseModel):
    original_description: str = Field(description="The original user description")
    interpretation: str = Field(description="Our interpretation of what the user wants")
    is_simple: bool = Field(
        description="True if behavior can be implemented with a single expert"
    )
    components: List[BehaviorComponent] = Field(
        description="Concrete expert specifications to implement"
    )
    context: Optional[DecompositionContext] = Field(
        default=None,
        description="Shared context for multi-expert behaviors"
    )
    suggested_states: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="All states needed across all components as (name, category) tuples"
    )
    implementation_notes: str = Field(
        default="",
        description="Notes on how to implement this effectively"
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
        from pydantic_ai.models.gemini import GeminiModel
        import os
        
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
        
        self.model = GeminiModel(model_name)
        self.model_name = model_name
        self.prompt_builder = prompt_builder
        self.decomposition_agent = self._create_decomposition_agent()
        
    def _create_decomposition_agent(self) -> Agent[DecompositionDependencies, DecomposedBehavior]:
        agent = Agent(
            self.model,
            deps_type=DecompositionDependencies,
            output_type=DecomposedBehavior,
            system_prompt="""You are an expert at analyzing behavior descriptions and determining 
            whether they need decomposition into multiple experts or can be handled by a single expert.
            
            TÖLVERA CONTEXT:
            - Particles have built-in properties: pos (vec2), vel (vec2), mass (f32), size (f32), species (i32), active (f32)
            - Coordinate system: (0,0) is top-left, Y+ points upward (standard physics)
            - Species system: Multiple species (0 to sn-1) can have different behaviors and colors
            - Forces should return ti.math.vec2(x, y) values
            - CRITICAL: Never use 'return' inside if/for/while blocks - causes crashes
            
            Your role is to:
            1. First assess if the behavior is SIMPLE (one expert) or COMPLEX (multiple experts)
            2. For each required expert, provide CONCRETE specifications with:
               - expert_name: A clear function name (e.g., 'gravity_force', 'prey_flee')
               - implementation: Specific force calculations or logic
               - expert_type: 'force', 'interaction', 'state_update', or 'visual'
            3. For complex behaviors, establish shared context to ensure coherent behavior
            
            SIMPLE behaviors (is_simple=true, single expert):
            - "particles fall with gravity" → gravity_force expert
            - "particles drift randomly" → random_walk expert
            - "particles attracted to center" → center_attraction expert
            - Basic forces or movements without complex interactions
            
            COMPLEX behaviors (is_simple=false, multiple experts):
            - "predator chases prey" → predator_chase + prey_flee experts
            - "particles live/die based on neighbors" → grid_update + neighbor_count + life_rules
            - "fireflies synchronize" → oscillator_update + phase_coupling
            - Multi-step processes, state-based rules, or interacting behaviors
            
            IMPORTANT DISTINCTIONS:
            - Ecosystem behaviors (predator/prey, fish schooling, etc.) do NOT need grid states
            - Only cellular automata patterns need grid_x, grid_y, is_alive states
            - "die" in ecological context (e.g., "scavengers clean up after predators") is NOT cellular automaton
            - Species interactions use particle positions, not grid positions
            
            For COMPLEX behaviors, create a context with:
            - constraints: Numerical relationships (e.g., prey_speed > predator_speed)
            - shared_parameters: Common values across experts
            - implementation_notes: How to ensure coherent behavior
            
            Expert implementation should be CONCRETE:
            - Force experts: "force = ti.math.vec2(0.0, -gravity_strength * mass); return force"
            - Interaction experts: "force = ti.math.vec2(0.0, 0.0); if distance < 100.0: force = normalize(target - pos) * strength; return force"
            - State updates: "tv.s.llm_particle.field[i].energy = max(0.0, energy - 0.01)"
            
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
            
            ECOSYSTEM ("fish school together, predators hunt"):
            Components:
            1. fish_schooling: "if same species and nearby: align velocities and stay close"
            2. predator_hunt: "if species == predator and prey nearby: return chase_force"
            3. prey_escape: "if species == prey and predator nearby: return flee_force"
            Context: {"fish_speed": 80.0, "predator_speed": 120.0, "school_radius": 50.0}
            NOTE: NO GRID STATES NEEDED - uses particle positions directly
            
            CELLULAR AUTOMATA ("lives/dies based on neighbors"):
            Components:
            1. update_grid_position: "grid_x = int(pos.x / cell_size); grid_y = int(pos.y / cell_size)"
            2. count_neighbors: "scan 8 surrounding cells, count alive neighbors"
            3. apply_life_rules: "if alive and (neighbors < 2 or neighbors > 3): next_state = 0"
            Context: {"cell_size": 10.0, "rules": "B3/S23"}
            REQUIRES: grid_x, grid_y, is_alive, neighbor_count states
            
            PREDATOR-PREY ("chase but never catch"):
            Components:
            1. predator_chase: "if prey nearby: force = normalize(prey_pos - pos) * 100.0"
            2. prey_flee: "if predator nearby: force = normalize(pos - predator_pos) * 150.0"
            Context: {"prey_speed": 150.0, "predator_speed": 100.0, "catch_distance": 20.0}
            NOTE: Uses species field, not grid states
            
            FLOCKING ("move together but don't crowd"):
            Components:
            1. separation: "avoid neighbors closer than 20 units"
            2. alignment: "match average velocity of neighbors"
            3. cohesion: "move toward local center of mass"
            Context: {"perception_radius": 50.0, "separation_distance": 20.0}
            
            TRAIL FOLLOWING ("leave trails others follow"):
            Components:
            1. deposit_trail: "mark pixel at current position with pheromone"
            2. sense_trail: "sample trail strength ahead, turn toward stronger"
            3. trail_decay: "reduce all trail strengths over time"
            Context: {"deposit_rate": 1.0, "decay_rate": 0.99, "sensor_angle": 0.5}
            REQUIRES: heading, sensor_distance states
            
            SIMPLE BEHAVIORS (single expert):
            - Gravity: "return ti.math.vec2(0.0, -300.0 * mass)"
            - Random drift: "return ti.math.vec2(ti.random()-0.5, ti.random()-0.5) * 50.0"
            - Center attraction: "force = normalize(center - pos) * 100.0"
            
            Remember:
            - Ecosystem behaviors use species field and positions, NOT grid states
            - Only cellular automata needs grid states
            - Always declare force variable before conditionals
            - Never use 'return' inside if blocks
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
            
            prompt = f"""Analyze this behavior and create expert specifications: "{description}"
            
            1. First determine if this is SIMPLE (one expert) or COMPLEX (multiple experts)
            2. For each expert needed, provide:
               - expert_name: function name (e.g., 'gravity_force')
               - implementation: concrete force/logic (e.g., 'return vec2(0, -300*mass)')
               - expert_type: 'force', 'interaction', 'state_update', or 'visual'
            3. If COMPLEX, establish context with constraints and shared parameters
            
            Examples:
            - "particles fall with gravity" → SIMPLE, one 'gravity_force' expert
            - "predator chases prey" → COMPLEX, needs 'predator_chase' + 'prey_flee' with speed constraints"""
            
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
                            "is_simple": result.output.is_simple,
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
                    "is_simple": decomposed.is_simple,
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
    
