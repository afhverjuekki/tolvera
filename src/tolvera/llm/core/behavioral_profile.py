"""Intent-enum override layer for per-species behavioural parameters.

Background — without this layer, the LLM picks numeric values like
``separation_strength = 25.0`` and ``repulsion_strength = 120.0`` that LOOK
reasonable but produce TV-static visuals (separation overpowers cohesion;
inter-species repulsion too weak to form territories). Pydantic
``Field(ge=, le=)`` only constrains LLM responses that come through Pydantic
output models — it can't constrain free-form Python code the LLM writes
inside ``TaichiCodeResponse.code``.

This module fixes that by separating "what kind of behaviour" (the LLM's
strength) from "what specific numbers" (a deterministic mapping table).
The LLM picks from a small Literal enum per species per dimension;
constrained decoding enforces those enums perfectly across every backend;
a Python dict translates to canonical numbers; then a regex pass rewrites
any matching per-species assignments in the generated sketch.

Flow:
    1. ``BehavioralProfileSelector.select(description, species_ids)``
       calls Bedrock with the ``SpeciesProfiles`` output schema.
    2. ``apply_profile_overrides(sketch_code, profiles)`` rewrites every
       ``tv.s.llm_species.field[N].cohesion_strength = X.X`` style line
       it can map to a known concept.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..debug.tracing import get_collector
from ..prompts.prompt_loader import get_prompt_loader
from .llm_factory import ModelFactory


# ----------------------------------------------------------------------------
# Pydantic output schema for the LLM call
# ----------------------------------------------------------------------------


CohesionEnum = Literal["loose", "moderate", "tight", "very_tight"]
AlignmentEnum = Literal["uncoordinated", "moderate", "synchronized", "highly_synchronized"]
SeparationEnum = Literal["touching_allowed", "personal_space", "wide_personal_space"]
InterSpeciesEnum = Literal[
    "ignore", "mild_repel", "distinct_territories", "hard_borders", "attract_to_others"
]
SpeedEnum = Literal["slow", "moderate", "fast", "very_fast"]


class SpeciesBehavioralProfile(BaseModel):
    """One species's behavioural intent across five axes.

    The LLM picks an enum value per axis; the deterministic
    ``PROFILE_VALUE_MAP`` below translates each enum to a canonical numeric
    value tuned for visible motion and audible OSC.
    """

    species_id: int = Field(ge=0, le=15, description="Species index, 0-based.")
    cohesion: CohesionEnum = Field(description="How tightly this species clusters with itself.")
    alignment: AlignmentEnum = Field(description="How synchronized this species's headings are.")
    separation: SeparationEnum = Field(description="Personal-space radius among same-species members.")
    inter_species: InterSpeciesEnum = Field(
        description=(
            "Behaviour toward OTHER species. Default to 'distinct_territories' "
            "for any multi-species prompt that does not explicitly say species "
            "mingle, predator-prey, or follow each other."
        )
    )
    speed: SpeedEnum = Field(description="Top speed cap for this species.")


class SpeciesProfiles(BaseModel):
    """LLM output: one profile per species."""

    profiles: List[SpeciesBehavioralProfile] = Field(
        description="One profile per species id."
    )


# ----------------------------------------------------------------------------
# Canonical enum → number mapping (the deterministic part)
# ----------------------------------------------------------------------------

# Per-concept enum-to-number tables. Numbers are tuned for visible motion on
# a 1920×1080 canvas with dt=0.15. If the visuals end up too aggressive or
# too tame in practice, edit these tables — the LLM never sees them.
PROFILE_VALUE_MAP: Dict[str, Dict[str, float]] = {
    "cohesion": {
        "loose": 0.5,
        "moderate": 1.5,
        "tight": 3.0,
        "very_tight": 5.0,
    },
    "alignment": {
        "uncoordinated": 0.3,
        "moderate": 1.5,
        "synchronized": 3.0,
        "highly_synchronized": 5.0,
    },
    "separation": {
        "touching_allowed": 0.5,
        "personal_space": 2.0,
        "wide_personal_space": 4.0,
    },
    "inter_species_repulsion": {
        "ignore": 0.0,
        "mild_repel": 100.0,
        "distinct_territories": 400.0,
        "hard_borders": 700.0,
        "attract_to_others": 0.0,  # attraction goes through the other table
    },
    "inter_species_attraction": {
        "ignore": 0.0,
        "mild_repel": 0.0,
        "distinct_territories": 0.0,
        "hard_borders": 0.0,
        "attract_to_others": 200.0,
    },
    "max_speed": {
        "slow": 80.0,
        "moderate": 140.0,
        "fast": 200.0,
        "very_fast": 300.0,
    },
}


# LLM-chosen field name → canonical concept. The LLM names per-species
# state fields inconsistently across runs (cohesion_strength, cohesion_weight,
# cohesion_factor, ...). We map every recognisable variant back to a
# canonical concept so the override pass catches them all.
FIELD_NAME_ALIASES: Dict[str, str] = {
    # Cohesion
    "cohesion_strength": "cohesion",
    "cohesion_weight": "cohesion",
    "cohesion_factor": "cohesion",
    "cohesion_scale": "cohesion",
    # Alignment
    "alignment_strength": "alignment",
    "alignment_weight": "alignment",
    "alignment_factor": "alignment",
    "align_strength": "alignment",
    # Separation (intra-species)
    "separation_strength": "separation",
    "separation_weight": "separation",
    "separation_factor": "separation",
    # Inter-species repulsion
    "repulsion_strength": "inter_species_repulsion",
    "inter_species_repulsion_strength": "inter_species_repulsion",
    "inter_species_repulsion": "inter_species_repulsion",
    "cross_species_repulsion_strength": "inter_species_repulsion",
    "cross_species_repulsion": "inter_species_repulsion",
    "territorial_repulsion_strength": "inter_species_repulsion",
    "interspecies_repulsion_strength": "inter_species_repulsion",
    # Inter-species attraction (canonical default 0)
    "inter_species_attraction": "inter_species_attraction",
    "inter_species_attraction_strength": "inter_species_attraction",
    "cross_attract_strength": "inter_species_attraction",
    "cross_species_attraction": "inter_species_attraction",
    "interspecies_attraction": "inter_species_attraction",
    # Speed cap
    "max_speed": "max_speed",
    "target_speed": "max_speed",
    "top_speed": "max_speed",
}


# ----------------------------------------------------------------------------
# Profile resolver: take SpeciesBehavioralProfile, return numeric values
# ----------------------------------------------------------------------------


def resolve_profile_numbers(profile: SpeciesBehavioralProfile) -> Dict[str, float]:
    """Translate a profile's enum picks into canonical numbers per concept."""
    return {
        "cohesion": PROFILE_VALUE_MAP["cohesion"][profile.cohesion],
        "alignment": PROFILE_VALUE_MAP["alignment"][profile.alignment],
        "separation": PROFILE_VALUE_MAP["separation"][profile.separation],
        "inter_species_repulsion": PROFILE_VALUE_MAP["inter_species_repulsion"][profile.inter_species],
        "inter_species_attraction": PROFILE_VALUE_MAP["inter_species_attraction"][profile.inter_species],
        "max_speed": PROFILE_VALUE_MAP["max_speed"][profile.speed],
    }


# ----------------------------------------------------------------------------
# Override engine: rewrite per-species assignments in generated sketch
# ----------------------------------------------------------------------------

# Match assignments like:
#   tv.s.llm_species.field[0].cohesion_strength = 2.5
#   tv.s.llm_species.field[2].max_speed         = 220.0
_ASSIGN_RE = re.compile(
    r"(tv\.s\.llm_species\.field\[(\d+)\]\.)([A-Za-z_][A-Za-z0-9_]*)"
    r"(\s*=\s*)"
    r"([-+]?\d+\.?\d*)"
)


def apply_profile_overrides(
    sketch_code: str,
    profiles: List[SpeciesBehavioralProfile],
) -> Tuple[str, List[str]]:
    """Rewrite per-species parameter assignments using canonical numbers.

    Returns the rewritten sketch plus a list of (species, field, old, new)
    edit descriptions for tracing.
    """
    by_species: Dict[int, Dict[str, float]] = {
        p.species_id: resolve_profile_numbers(p) for p in profiles
    }

    edits: List[str] = []

    def _sub(match: re.Match) -> str:
        prefix, sid_str, field_name, eq, old_value = match.groups()
        sid = int(sid_str)
        if sid not in by_species:
            return match.group(0)
        concept = FIELD_NAME_ALIASES.get(field_name)
        if concept is None:
            return match.group(0)
        new_value = by_species[sid].get(concept)
        if new_value is None:
            return match.group(0)
        edits.append(f"species[{sid}].{field_name}: {old_value} → {new_value}")
        # Preserve formatting: keep one decimal place
        return f"{prefix}{field_name}{eq}{new_value}"

    new_code = _ASSIGN_RE.sub(_sub, sketch_code)
    return new_code, edits


# ----------------------------------------------------------------------------
# LLM agent for selecting profiles
# ----------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are choosing BEHAVIOURAL INTENT for a multi-species particle simulation.

For each species, pick ONE value per axis from the enums in the schema. The
combination should reflect the USER PROMPT — read it carefully.

AXIS GUIDANCE:

- cohesion: how tightly each species clusters with its own members.
  - "loose" — barely cohesive, sparse
  - "moderate" — typical flock spacing
  - "tight" — dense cluster
  - "very_tight" — extremely dense ball

- alignment: how synchronized the heading is among same-species members.
  - "uncoordinated" — random heading per particle
  - "moderate" — soft tendency to align
  - "synchronized" — clear directional consensus
  - "highly_synchronized" — military-precise sync

- separation: personal-space radius among same-species members.
  - "touching_allowed" — members can pack densely against each other
  - "personal_space" — typical small gap
  - "wide_personal_space" — particles keep noticeable distance

- inter_species: behaviour toward OTHER species. This is the most important axis.
  - "ignore" — species pass through each other without interacting
  - "mild_repel" — soft push apart, species blur into one another at edges
  - "distinct_territories" — each species carves out its own visible region (DEFAULT for multi-species)
  - "hard_borders" — strong repulsion, sharp territory borders
  - "attract_to_others" — predator/prey or mingling — species are PULLED to other species

- speed: top speed cap.
  - "slow" — drift
  - "moderate" — typical walking speed
  - "fast" — active cruising
  - "very_fast" — energetic darting

CRITICAL DEFAULTS:
- For ANY multi-species prompt that does NOT explicitly say "mingle", "mix", "blend",
  "merge", "predator/prey", "hunt", "follow each other", "orbit", or similar
  pull-toward-others language: ALWAYS pick inter_species = "distinct_territories" (or
  "hard_borders" if the prompt emphasises clear separation). NEVER pick "attract_to_others"
  unless the prompt explicitly opts in.
- The phrase "flock together" in a multi-species prompt means "each species flocks
  amongst its own members" — it does NOT mean species merge. Pick "distinct_territories".
- For "predator/prey": predator species get "attract_to_others"; prey species get
  "hard_borders" (fleeing).

Pick one profile per species id provided. Output is validated against the schema.
"""


class BehavioralProfileSelector:
    """Calls Bedrock to pick a SpeciesBehavioralProfile per species id."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        if model_name is None:
            model_name = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
        self.model_name = model_name
        self.model = ModelFactory.create_model(model_name, api_key)
        self._agent: Optional[Agent] = None

    def _get_agent(self) -> Agent:
        if self._agent is None:
            self._agent = Agent(
                self.model,
                output_type=SpeciesProfiles,
                output_retries=2,
                system_prompt=_SYSTEM_PROMPT,
                # SpeciesProfiles grows with species count; reasoning-heavy
                # models can otherwise truncate the tool call and fail
                # validation. Give it room (see code_generator.SYNTHESIS_MAX_TOKENS).
                model_settings={"max_tokens": 12000},
            )
        return self._agent

    async def select(
        self,
        description: str,
        species_ids: List[int],
    ) -> List[SpeciesBehavioralProfile]:
        """Ask the LLM to pick a behavioural profile per species id."""
        agent = self._get_agent()
        user_prompt = (
            f"USER PROMPT: {description!r}\n\n"
            f"Species ids to assign profiles for: {species_ids}\n\n"
            "Return one SpeciesBehavioralProfile per id. Remember the multi-species default."
        )
        collector = get_collector()
        with collector.trace_node(
            "select_species_profiles", "behavioral_profile_selection",
            description=description, species_ids=species_ids,
        ):
            result = await agent.run(user_prompt)
            return result.output.profiles
