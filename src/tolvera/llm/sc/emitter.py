import logging
import os
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..core.data_models import SpeciesConfiguration
from .mappings import (
    AMP_RANGE,
    FILTER_RANGE_HZ,
    MUSICAL_INTENT_PATTERN,
    OSC_PORT,
    PITCH_RANGE_HZ,
)
from .synth_library import SYNTH_CATALOG

logger = logging.getLogger(__name__)

_MUSICAL_INTENT_RE = re.compile(MUSICAL_INTENT_PATTERN)


def has_musical_intent(description: str) -> bool:
    if not description:
        return False
    return _MUSICAL_INTENT_RE.search(description) is not None


OSC_SENDER_SENTINEL = "# === OSC Senders (companion SuperCollider patch on port 5000) ==="
OSC_METRICS_CALL = "_compute_osc_metrics()  # OSC senders"

OSC_SENDER_BLOCK = '''# === OSC Senders (companion SuperCollider patch on port 5000) ===
# Per-species centroid + mean-velocity tracking. We deliberately do NOT lock
# onto a single particle (the original v1 pattern): that particle often ends
# up in a tight cluster with low local velocity and the audio falsely reports
# stillness even when the flock is in motion. Averaging across all active
# particles of a species gives an honest sonic representation.
#
# vel divisor (2.0): tuned so the smoothed average lands in the audible amp
# range [0.2, 1.0]. With pixel-per-step velocities of 5–30 (typical for
# flocking on a 1920×1080 canvas), this maps to amp_max bursts of 0.3+ during
# active motion and ~0.05 at near-rest. Raising the divisor flattens the
# audio; lowering saturates more frequently.
# Exponential smoothing of the per-species metrics before they are sent. This
# is a one-pole low-pass: higher alpha is smoother but adds perceptible lag
# between a motion on screen and the matching change in sound. 0.6 keeps the
# audio responsive while still removing per-frame jitter (0.85 felt sluggish).
_OSC_SMOOTH_ALPHA = 0.6
_OSC_VEL_DIVISOR = 2.0
_osc_smooth_x = ti.field(ti.f32, shape=tv.sn)
_osc_smooth_y = ti.field(ti.f32, shape=tv.sn)
_osc_smooth_vel = ti.field(ti.f32, shape=tv.sn)
_osc_sum_x = ti.field(ti.f32, shape=tv.sn)
_osc_sum_y = ti.field(ti.f32, shape=tv.sn)
_osc_sum_vel = ti.field(ti.f32, shape=tv.sn)
_osc_count = ti.field(ti.i32, shape=tv.sn)

# Visual ring tracking: bind one particle per species at startup and draw
# the ring at THAT particle's position each frame. Keeping this separate
# from the centroid-mean used for OSC: the centroid is the right *audio*
# signal (it averages out individual jitter), but a single tracked particle
# is the right *visual* — when species spread across the whole canvas the
# centroid pools at (0.5, 0.5) and all rings stack in the middle.
_osc_tracked_idx = ti.field(ti.i32, shape=tv.sn)

@ti.kernel
def _osc_assign_tracked():
    for s in range(tv.sn):
        _osc_tracked_idx[s] = tv.p.n
    for i in range(tv.p.n):
        if tv.p.field[i].active > 0:
            s = tv.p.field[i].species
            ti.atomic_min(_osc_tracked_idx[s], i)

_osc_assign_tracked()

@ti.kernel
def _osc_metrics_kernel():
    for s in range(tv.sn):
        _osc_sum_x[s] = 0.0
        _osc_sum_y[s] = 0.0
        _osc_sum_vel[s] = 0.0
        _osc_count[s] = 0
    for i in range(tv.p.n):
        if tv.p.field[i].active > 0:
            s = tv.p.field[i].species
            r = tv.p.field[i]
            _osc_sum_x[s]   += r.pos.x / tv.x
            _osc_sum_y[s]   += r.pos.y / tv.y
            _osc_sum_vel[s] += r.vel.norm() / _OSC_VEL_DIVISOR
            _osc_count[s]   += 1
    for s in range(tv.sn):
        cx = 0.5
        cy = 0.5
        cv = 0.0
        n  = _osc_count[s]
        if n > 0:
            cx = _osc_sum_x[s] / float(n)
            cy = _osc_sum_y[s] / float(n)
            cv = _osc_sum_vel[s] / float(n)
        _osc_smooth_x[s] = _OSC_SMOOTH_ALPHA * _osc_smooth_x[s] + (1.0 - _OSC_SMOOTH_ALPHA) * cx
        _osc_smooth_y[s] = _OSC_SMOOTH_ALPHA * _osc_smooth_y[s] + (1.0 - _OSC_SMOOTH_ALPHA) * cy
        _osc_smooth_vel[s] = _OSC_SMOOTH_ALPHA * _osc_smooth_vel[s] + (1.0 - _OSC_SMOOTH_ALPHA) * cv

_osc_frame = [0]
def _compute_osc_metrics():
    _osc_metrics_kernel()
    _osc_frame[0] += 1
    # Every ~5 seconds at 60 FPS — heartbeat only, not per-frame spam.
    if _osc_frame[0] % 300 == 0:
        parts = []
        for s_idx in range(tv.sn):
            parts.append(f's{s_idx}:x={float(_osc_smooth_x[s_idx]):.2f} y={float(_osc_smooth_y[s_idx]):.2f} v={float(_osc_smooth_vel[s_idx]):.2f}')
        print('[osc] ' + '  '.join(parts), flush=True)

@ti.kernel
def _draw_tracked_rings():
    # Draw a ring at the per-species TRACKED PARTICLE's position (not the
    # centroid). When species disperse across the canvas the centroid
    # collapses to (0.5, 0.5) and all rings stack uselessly in the middle;
    # tracking one specific particle keeps the rings moving with the
    # action regardless of how spread out the flock is.
    for s in range(tv.sn):
        idx = _osc_tracked_idx[s]
        if idx < tv.p.n and tv.p.field[idx].active > 0:
            r = tv.p.field[idx]
            x = ti.cast(r.pos.x, ti.i32)
            y = ti.cast(r.pos.y, ti.i32)
            base = ti.cast(r.size + 8, ti.i32)
            white = ti.Vector([1.0, 1.0, 1.0, 1.0])
            for k in ti.static(range(3)):
                tv.px.circle(x, y, base + k, white, 0)

def _make_osc_senders():
    for s_idx in range(tv.sn):
        @tv.osc.map.send_args(val=(0.5, 0, 1), send_mode='broadcast',
                              name=f'metrics/{s_idx}/x', count=2)
        def _send_x(_s=s_idx) -> list[float]:
            return [float(np.clip(_osc_smooth_x[_s], 0, 1))]

        @tv.osc.map.send_args(val=(0.5, 0, 1), send_mode='broadcast',
                              name=f'metrics/{s_idx}/y', count=2)
        def _send_y(_s=s_idx) -> list[float]:
            return [float(np.clip(_osc_smooth_y[_s], 0, 1))]

        @tv.osc.map.send_args(val=(0.5, 0, 1), send_mode='broadcast',
                              name=f'metrics/{s_idx}/vel', count=2)
        def _send_vel(_s=s_idx) -> list[float]:
            return [float(np.clip(_osc_smooth_vel[_s], 0, 1))]

_make_osc_senders()
'''


def ensure_osc_senders(sketch: str, description: str) -> str:
    """Guarantee a saved sketch contains the OSC sender block when musical
    intent is present. Idempotent: a sketch already containing the sentinel
    is returned unchanged. The block is inserted before `@tv.render` (so it
    sits inside `main()`), and the per-frame `_compute_osc_metrics()` call is
    inserted at the top of the `@tv.render` body if missing.

    Called after refinement; refinement may rewrite the whole sketch and
    drop template-injected blocks, so this is the canonical guarantee.
    """
    # A refine/repair pass may rewrite the whole sketch and drop the
    # template-injected render-loop CALLS (e.g. `_draw_tracked_rings()`) while
    # keeping the sender-block definitions. So treat the presence of the block
    # as "this is a musical sketch" and fall through to re-ensure the calls,
    # rather than early-returning on the sentinel. Block insertion itself is
    # still guarded by `has_block` below so we never double-inject it.
    has_block = OSC_SENDER_SENTINEL in sketch
    if not has_musical_intent(description) and not has_block:
        return sketch

    render_idx = sketch.find("@tv.render")
    if render_idx == -1:
        logger.warning(
            "ensure_osc_senders: no @tv.render found; skipping injection"
        )
        return sketch

    line_start = sketch.rfind("\n", 0, render_idx) + 1
    base_indent = sketch[line_start:render_idx]

    if not has_block:
        indented_block = "\n".join(
            (base_indent + line if line else line)
            for line in OSC_SENDER_BLOCK.splitlines()
        )

        sketch = (
            sketch[:line_start]
            + indented_block
            + "\n\n"
            + sketch[line_start:]
        )

    if OSC_METRICS_CALL not in sketch:
        render_idx = sketch.find("@tv.render")
        body_start = sketch.find("\n", sketch.find("def _():", render_idx))
        if body_start != -1:
            inner_indent = base_indent + "    "
            sketch = (
                sketch[: body_start + 1]
                + f"{inner_indent}{OSC_METRICS_CALL}\n"
                + sketch[body_start + 1 :]
            )

    DRAW_CALL_MARKER = "# highlight tracked particles"
    if DRAW_CALL_MARKER not in sketch:
        ret_idx = sketch.find("return tv.px")
        if ret_idx != -1:
            ret_line_start = sketch.rfind("\n", 0, ret_idx) + 1
            ret_indent = sketch[ret_line_start:ret_idx]
            sketch = (
                sketch[:ret_line_start]
                + f"{ret_indent}_draw_tracked_rings()  {DRAW_CALL_MARKER}\n"
                + sketch[ret_line_start:]
            )

    if "kwargs['osc']" not in sketch and 'kwargs["osc"]' not in sketch:
        tv_idx = sketch.find("tv = Tolvera(")
        if tv_idx != -1:
            tv_line_start = sketch.rfind("\n", 0, tv_idx) + 1
            tv_indent = sketch[tv_line_start:tv_idx]
            osc_kwarg = (
                f"{tv_indent}if 'osc' not in kwargs:\n"
                f"{tv_indent}    kwargs['osc'] = True  # Required for OSC senders\n"
            )
            sketch = sketch[:tv_line_start] + osc_kwarg + sketch[tv_line_start:]

    return sketch


def _resolve_species_name(species_config: SpeciesConfiguration, species_id: int) -> str:
    if species_config.species_names:
        for mapping in species_config.species_names:
            if mapping.species_id == species_id:
                return mapping.name
    return f"species_{species_id}"


def _assign_synth(index: int, total: int) -> str:
    catalog_size = len(SYNTH_CATALOG)
    if total > catalog_size and index == 0:
        logger.warning(
            "Species count (%d) exceeds synth catalog size (%d); rolling over to '%s'.",
            total,
            catalog_size,
            SYNTH_CATALOG[0][0],
        )
    return SYNTH_CATALOG[index % catalog_size][0]


def emit_companion(
    species_config: SpeciesConfiguration,
    description: str,
    out_path: Path,
) -> Path:
    out_path = Path(out_path)
    species_ids = list(species_config.species_ids)
    total = len(species_ids)

    species = []
    used_synths: dict[str, str] = {}
    catalog_lookup = dict(SYNTH_CATALOG)

    for idx, sid in enumerate(species_ids):
        synth_name = _assign_synth(idx, total)
        species.append(
            {
                "id": sid,
                "name": _resolve_species_name(species_config, sid),
                "synth_name": synth_name,
            }
        )
        if synth_name not in used_synths:
            used_synths[synth_name] = catalog_lookup.get(synth_name, "")

    synthdefs = [{"name": name, "body": body} for name, body in used_synths.items()]

    templates_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "templates"
    )
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("supercollider/companion.scd.j2")

    rendered = template.render(
        description=description,
        port=OSC_PORT,
        pitch_min=PITCH_RANGE_HZ[0],
        pitch_max=PITCH_RANGE_HZ[1],
        amp_max=AMP_RANGE[1],
        filter_min=FILTER_RANGE_HZ[0],
        filter_max=FILTER_RANGE_HZ[1],
        species=species,
        synthdefs=synthdefs,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(rendered)

    return out_path
