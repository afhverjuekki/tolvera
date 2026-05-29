"""
Flock Murmuration
=================

A hand-built Tölvera piece on the native boids/flock engine. Four species flock
by separation, alignment and cohesion, with per-species radii and weights tuned
so the swarm keeps swirling and folding like a starling murmuration rather than
clumping and stopping. Velocity is clamped each frame so it stays fast and fluid
without blowing up. Long diffuse trails smear the motion into glowing ribbons.

Sonification (OSC -> nime_flock_murmuration.scd), one voice per species:
  /metrics/<s>/x    centroid x  -> pitch
  /metrics/<s>/y    centroid y  -> filter cutoff
  /metrics/<s>/vel  mean speed  -> amplitude (the flock is always moving)

Run with sound (two terminals):
  "/Applications/SuperCollider.app/Contents/MacOS/sclang" \
    examples/generated_sketches/nime_demos/nime_flock_murmuration.scd
  python examples/generated_sketches/nime_demos/nime_flock_murmuration.py
"""

import numpy as np
import taichi as ti

from tolvera import Tolvera, run

MAX_VEL = 7.0  # px/frame: keeps motion lively but bounded


def main(**kwargs):
    kwargs.setdefault("species", 4)
    kwargs.setdefault("particles", 2400)
    kwargs.setdefault("width", 1920)
    kwargs.setdefault("height", 1080)
    if "osc" not in kwargs:
        kwargs["osc"] = True

    tv = Tolvera(**kwargs)

    cols = [
        [0.20, 0.85, 1.00, 1.0],   # sky blue
        [1.00, 0.35, 0.30, 1.0],   # coral
        [0.70, 1.00, 0.30, 1.0],   # lime
        [0.95, 0.55, 1.00, 1.0],   # orchid
    ]
    for s in range(tv.sn):
        tv.s.species.field[s].rgba = cols[s % len(cols)]

    # Flock parameters per species pair. Same-species: cohere+align (flock
    # together); cross-species: gentle separation so the flocks interleave
    # without merging into one blob.
    for a in range(tv.sn):
        for b in range(tv.sn):
            f = tv.s.flock_s.field[a, b]
            if a == b:
                f.separate = 0.35
                f.align = 0.55
                f.cohere = 0.45
                f.radius = 0.30
            else:
                f.separate = 0.6
                f.align = 0.0
                f.cohere = 0.0
                f.radius = 0.12

    # ---- sonification metrics (pos-delta speed) ---------------------------
    _prev = ti.Vector.field(2, ti.f32, shape=tv.pn)
    _sum_x = ti.field(ti.f32, shape=tv.sn)
    _sum_y = ti.field(ti.f32, shape=tv.sn)
    _sum_vel = ti.field(ti.f32, shape=tv.sn)
    _count = ti.field(ti.i32, shape=tv.sn)
    _sx = ti.field(ti.f32, shape=tv.sn)
    _sy = ti.field(ti.f32, shape=tv.sn)
    _sv = ti.field(ti.f32, shape=tv.sn)
    _tracked = ti.field(ti.i32, shape=tv.sn)
    ALPHA = 0.6

    @ti.kernel
    def _clamp_vel():
        for i in range(tv.pn):
            v = tv.p.field[i].vel
            n = v.norm()
            if n > MAX_VEL:
                tv.p.field[i].vel = v / n * MAX_VEL

    @ti.kernel
    def _init_prev():
        for i in range(tv.pn):
            _prev[i] = tv.p.field[i].pos
        for s in range(tv.sn):
            _tracked[s] = tv.pn
        for i in range(tv.pn):
            if tv.p.field[i].active > 0.0:
                s = tv.p.field[i].species
                ti.atomic_min(_tracked[s], i)

    @ti.kernel
    def _metrics():
        for s in range(tv.sn):
            _sum_x[s] = 0.0; _sum_y[s] = 0.0; _sum_vel[s] = 0.0; _count[s] = 0
        for i in range(tv.pn):
            if tv.p.field[i].active > 0.0:
                s = tv.p.field[i].species
                pos = tv.p.field[i].pos
                d = (pos - _prev[i]).norm()
                if d > 40.0:
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
                cv = ti.min((_sum_vel[s] / n) / 5.0, 1.0)
            _sx[s] = ALPHA * _sx[s] + (1.0 - ALPHA) * cx
            _sy[s] = ALPHA * _sy[s] + (1.0 - ALPHA) * cy
            _sv[s] = ALPHA * _sv[s] + (1.0 - ALPHA) * cv

    @ti.kernel
    def _draw_rings():
        for s in range(tv.sn):
            idx = _tracked[s]
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
            print("[flock] " + "  ".join(f"s{s}:v={_sv[s]:.2f}" for s in range(tv.sn)), flush=True)

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
        tv.px.diffuse(0.90)
        tv.v.flock(tv.p, weight=0.4)
        _clamp_vel()
        _tick()
        tv.px.particles(tv.p, tv.s.species())
        _draw_rings()
        return tv.px

    return tv


if __name__ == "__main__":
    run(main)
