"""
Slime Networks
==============

A hand-built Tölvera piece on the native slime (physarum) engine. Thousands of
agents per species crawl along a pheromone trail they continuously deposit and
sense, so the field never settles: glowing branching networks form, thicken,
dissolve and reroute forever. This is the fluid, organic, always-moving look
that artificial life is about.

Three species lay down three coloured pheromones (cyan, magenta, amber) that
weave over and through one another.

Sonification (streamed over OSC to nime_slime_networks.scd), one voice per
species, driven by the swarm's actual on-screen motion:
  /metrics/<s>/x    centroid x  -> pitch of that species' voice
  /metrics/<s>/y    centroid y  -> filter cutoff (brightness)
  /metrics/<s>/vel  mean crawl  -> amplitude (the network is always moving, so
                                  there is always sound, swelling where a
                                  species is actively reaching and rerouting)

Run with sound (two terminals):
  "/Applications/SuperCollider.app/Contents/MacOS/sclang" \
    examples/generated_sketches/nime_demos/nime_slime_networks.scd
  python examples/generated_sketches/nime_demos/nime_slime_networks.py
"""

import numpy as np
import taichi as ti

from tolvera import Tolvera, run


def main(**kwargs):
    if "species" not in kwargs:
        kwargs["species"] = 3
    if "particles" not in kwargs:
        kwargs["particles"] = 9000     # dense networks; slime is O(n), cheap
    if "width" not in kwargs:
        kwargs["width"] = 1920
    if "height" not in kwargs:
        kwargs["height"] = 1080
    if "osc" not in kwargs:
        kwargs["osc"] = True

    tv = Tolvera(**kwargs)

    # Three glowing pheromone colours.
    tv.s.species.field[0].rgba = [0.10, 1.00, 0.90, 1.0]   # cyan
    tv.s.species.field[1].rgba = [1.00, 0.20, 0.80, 1.0]   # magenta
    tv.s.species.field[2].rgba = [1.00, 0.70, 0.10, 1.0]   # amber

    # Per-species slime parameters (multipliers on the module constants). A bit
    # of variety gives the three networks distinct textures: one fine and
    # exploratory, one broad and sweeping, one tight and clumping.
    slime_cfg = [
        dict(sense_angle=0.45, sense_dist=0.65, move_angle=0.55, move_dist=0.55, evaporate=0.965),
        dict(sense_angle=0.60, sense_dist=0.45, move_angle=0.40, move_dist=0.75, evaporate=0.955),
        dict(sense_angle=0.35, sense_dist=0.80, move_angle=0.65, move_dist=0.45, evaporate=0.975),
    ]
    for s, cfg in enumerate(slime_cfg):
        for k, v in cfg.items():
            setattr(tv.s.slime_s.field[s], k, v)

    # ---- sonification metrics (speed from position deltas) ----------------
    # Slime updates particle positions but not velocity, so we measure the
    # real per-frame displacement to drive amplitude. Wrap-around jumps at the
    # toroidal edges are clamped so they do not spike the audio.
    _prev = ti.Vector.field(2, ti.f32, shape=tv.pn)
    _sum_x = ti.field(ti.f32, shape=tv.sn)
    _sum_y = ti.field(ti.f32, shape=tv.sn)
    _sum_vel = ti.field(ti.f32, shape=tv.sn)
    _count = ti.field(ti.i32, shape=tv.sn)
    _sx = ti.field(ti.f32, shape=tv.sn)
    _sy = ti.field(ti.f32, shape=tv.sn)
    _sv = ti.field(ti.f32, shape=tv.sn)
    _osc_tracked = ti.field(ti.i32, shape=tv.sn)
    ALPHA = 0.6

    @ti.kernel
    def _init_prev():
        for i in range(tv.pn):
            _prev[i] = tv.p.field[i].pos
        for s in range(tv.sn):
            _osc_tracked[s] = tv.pn
        for i in range(tv.pn):
            if tv.p.field[i].active > 0.0:
                s = tv.p.field[i].species
                ti.atomic_min(_osc_tracked[s], i)

    @ti.kernel
    def _metrics():
        for s in range(tv.sn):
            _sum_x[s] = 0.0; _sum_y[s] = 0.0; _sum_vel[s] = 0.0; _count[s] = 0
        for i in range(tv.pn):
            if tv.p.field[i].active > 0.0:
                s = tv.p.field[i].species
                pos = tv.p.field[i].pos
                d = (pos - _prev[i]).norm()
                if d > 30.0:        # toroidal wrap, ignore the jump
                    d = 0.0
                _sum_x[s] += pos.x / tv.x
                _sum_y[s] += pos.y / tv.y
                _sum_vel[s] += d
                _count[s] += 1
                _prev[i] = pos
        for s in range(tv.sn):
            cx = 0.5; cy = 0.5; cv = 0.0
            n = _count[s]
            if n > 0:
                cx = _sum_x[s] / n
                cy = _sum_y[s] / n
                cv = ti.min((_sum_vel[s] / n) / 3.0, 1.0)   # ~3px crawl -> full
            _sx[s] = ALPHA * _sx[s] + (1.0 - ALPHA) * cx
            _sy[s] = ALPHA * _sy[s] + (1.0 - ALPHA) * cy
            _sv[s] = ALPHA * _sv[s] + (1.0 - ALPHA) * cv

    @ti.kernel
    def _draw_rings():
        for s in range(tv.sn):
            idx = _osc_tracked[s]
            if idx < tv.pn and tv.p.field[idx].active > 0.0:
                r = tv.p.field[idx]
                x = ti.cast(r.pos.x, ti.i32)
                y = ti.cast(r.pos.y, ti.i32)
                white = ti.Vector([1.0, 1.0, 1.0, 1.0])
                for k in ti.static(range(3)):
                    tv.px.circle(x, y, 7 + k, white, 0)

    _frame = [0]

    def _tick():
        _metrics()
        _frame[0] += 1
        if _frame[0] % 180 == 0:
            parts = [f"s{s}:v={_sv[s]:.2f}" for s in range(tv.sn)]
            print("[slime] " + "  ".join(parts), flush=True)

    def _make_senders():
        for s_idx in range(tv.sn):
            @tv.osc.map.send_args(val=(0.5, 0, 1), send_mode="broadcast",
                                  name=f"metrics/{s_idx}/x", count=2)
            def _sx_send(_s=s_idx) -> list:
                return [float(np.clip(_sx[_s], 0, 1))]

            @tv.osc.map.send_args(val=(0.5, 0, 1), send_mode="broadcast",
                                  name=f"metrics/{s_idx}/y", count=2)
            def _sy_send(_s=s_idx) -> list:
                return [float(np.clip(_sy[_s], 0, 1))]

            @tv.osc.map.send_args(val=(0.5, 0, 1), send_mode="broadcast",
                                  name=f"metrics/{s_idx}/vel", count=2)
            def _sv_send(_s=s_idx) -> list:
                return [float(np.clip(_sv[_s], 0, 1))]

    _init_prev()
    if tv.osc is not False:
        _make_senders()

    @tv.render
    def _():
        trail = tv.v.slime(tv.p, tv.s.species, weight=1.0)
        tv.px.set(trail)
        _tick()
        _draw_rings()
        return tv.px

    return tv


if __name__ == "__main__":
    run(main)
