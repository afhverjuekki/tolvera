# NIME demos: artificial life with sound

Four hand-built Tölvera artificial-life demos, each on a native simulation engine
so the motion is genuinely fluid and never settles, and each paired with a
SuperCollider patch driven by the swarm's own motion over OSC. We verified each
one by running it, screenshotting, waiting, and screenshotting again to confirm
it keeps evolving, and by booting its `.scd` companion (server up, listening on
UDP 5000, low latency).

All four send one OSC voice per species on `/metrics/<s>/x|y|vel`: centroid x to
pitch, centroid y to filter cutoff, mean motion to amplitude. Because the engines
never stop moving, there is always sound.

## The demos

| Demo | Engine | What you see | What you hear |
|---|---|---|---|
| `nime_slime_networks` | physarum / slime | glowing cyan-magenta-amber pheromone membranes that coarsen, reroute and reform | three warm voices, crawl activity to amplitude |
| `nime_flock_murmuration` | boids / flock | dense swirling four-colour comma-trails folding like starlings | four-voice chord, speed to amplitude, height to cutoff |
| `nime_particle_life_cells` | particle life | living filament and cell chains forming, chasing and dissolving | four FM-bell voices |
| `nime_orbital_swarm` | force field | a small galaxy: spiral arms winding into a glowing core between two drifting attractors | three glassy pad voices |

## Running a demo

```bash
# terminal 1 - SuperCollider (boots the server, listens on UDP 5000)
"/Applications/SuperCollider.app/Contents/MacOS/sclang" \
  examples/generated_sketches/nime_demos/nime_orbital_swarm.scd

# terminal 2 - the Taichi simulation (streams OSC to 5000)
python examples/generated_sketches/nime_demos/nime_orbital_swarm.py
```

Swap in any of the four names. Each `.py` runs on its own (visual only) if you
skip SuperCollider.

## Helper scripts

- `_capture.py <sketch.py> <tag> [args]` runs a demo, screenshots every display,
  waits 30 s, screenshots again, then quits. Used to confirm a demo keeps moving.
- `osc_monitor.py <sketch.py> [port]` binds an OSC server in place of
  SuperCollider and prints the values a sketch emits, to debug a silent demo.
- `verify_demo.py`, `generate_demos.py`, `repair_loop.py` are from the LLM
  pipeline work (generate a sketch, run it, repair it on crash, verify it boots).
