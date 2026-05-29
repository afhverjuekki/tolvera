# src/llm/sc — SuperCollider Companion Emitter

This module emits a companion `.scd` file alongside generated Tölvera Python sketches whenever the user's behavior description carries musical intent. The `.scd` file boots a SuperCollider server, defines per-species `SynthDef`s, and registers `OSCdef` responders that listen on the same UDP port Tölvera sends to.

## OSC Contract

The contract is fixed by `src/tolvera/llm/context/exemplars/boids_osc.py`. Per species, Tölvera broadcasts three normalized-to-`[0,1]` floats on three separate addresses:

- `/metrics/<species_id>/x` — smoothed normalized x position
- `/metrics/<species_id>/y` — smoothed normalized y position
- `/metrics/<species_id>/vel` — smoothed normalized velocity magnitude

There is no `density` channel and no separate `vx`/`vy` components — only `vel`, the smoothed magnitude.

The SC patch listens on UDP port **5000** (matching `src/tolvera/osc/osc.py` `send_port`). Each metric is mapped to a synth control:

- `x -> freq` via `linexp(0, 1, 80, 3000)` Hz
- `y -> cutoff` via `linexp(0, 1, 200, 8000)` Hz
- `vel -> amp` via `linlin(0, 1, 0, 0.4)`

## y-substitutes-for-density

The NIME 2026 paper text refers to "spatial density to filter cutoff" as a sonification choice. The OSC stream as implemented in `boids_osc.py` does not expose density per species — only x, y, and smoothed velocity. To honor the paper's filter-cutoff mapping without inventing a channel, this emitter routes y-position to the cutoff control. Treat this as a documented substitution, not a bug.

## Module Layout

- `mappings.py` — port, address template, range constants, regex for musical-intent detection.
- `synth_library.py` — `SYNTH_CATALOG`, the ordered list of six SynthDef name/body pairs (`saw, fm, pulse, fm_bell, pluck, granular`). The bodies are authored by hand; this file holds them as plain Python strings so the emitter can include them verbatim in the rendered `.scd`.
- `emitter.py` — `has_musical_intent(description)` and `emit_companion(species_config, description, out_path)`. The emitter assigns one synth per species in catalog order (round-robin with a logged warning if species count exceeds six), deduplicates `SynthDef`s actually used, and renders `templates/supercollider/companion.scd.j2`.

## Extending `synth_library.py`

To add a synth: append a `(name, body)` tuple to `SYNTH_CATALOG`. The body is the lines that go inside the `SynthDef` block, with three available controls (`freq`, `amp`, `cutoff`). Keep names short and lowercase — they appear as both `\name` symbols in SC and as Python identifiers in the catalog. Order matters: species are assigned synths by catalog index, so put the most generally useful synths first.

## Hook Site

The companion is emitted from `BehaviorOrchestrator._save_to_file` immediately after the Python sketch is written. If the description matches the musical-intent regex and a species configuration is present, the `.scd` is written next to the `.py` with the same stem.
