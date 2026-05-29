"""
Orbital Swarm
=============

A hand-built Tölvera piece on Tölvera's native force field. Two attractors drift
in slow circles while a centripetal force spins everything around the centre, so
particles fall into orbits, get handed between the two wells, and trail long
glowing spiral arms like a small galaxy being stirred. Light damping and a speed
clamp keep the orbits fluid instead of collapsing or flying apart.

Sonification (OSC -> nime_orbital_swarm.scd), one voice per species:
  /metrics/<s>/x    centroid x  -> pitch
  /metrics/<s>/y    centroid y  -> filter cutoff
  /metrics/<s>/vel  mean speed  -> amplitude (orbital energy)

Run with sound (two terminals):
  "/Applications/SuperCollider.app/Contents/MacOS/sclang" \
    examples/generated_sketches/nime_demos/nime_orbital_swarm.scd
  python examples/generated_sketches/nime_demos/nime_orbital_swarm.py
"""

import math

import numpy as np
import taichi as ti

from tolvera import Tolvera, run

MAX_VEL = 9.0
DAMP = 0.94


def main(**kwargs):
    kwargs.setdefault("species", 3)
    kwargs.setdefault("particles", 3000)
    kwargs.setdefault("width", 1920)
    kwargs.setdefault("height", 1080)
    if "osc" not in kwargs:
        kwargs["osc"] = True

    tv = Tolvera(**kwargs)

    cols = [
        [0.30, 0.70, 1.00, 1.0],   # blue
        [1.00, 0.45, 0.85, 1.0],   # pink
        [1.00, 0.85, 0.35, 1.0],   # gold
    ]
    for s in range(tv.sn):
        tv.s.species.field[s].rgba = cols[s % len(cols)]

    CX, CY = tv.x * 0.5, tv.y * 0.5
    ORBIT_R = min(tv.x, tv.y) * 0.26
    MASS = 1.6
    A_RAD = max(tv.x, tv.y)            # attractors reach the whole field
    CENTRI = 1.1

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
    def _damp_clamp():
        for i in range(tv.pn):
            v = tv.p.field[i].vel * DAMP
            m = v.norm()
            if m > MAX_VEL:
                v = v / m * MAX_VEL
            tv.p.field[i].vel = v

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
            nc = _count[s]
            if nc > 0:
                cx = _sum_x[s] / nc
                cy = _sum_y[s] / nc
                cv = ti.min((_sum_vel[s] / nc) / 5.0, 1.0)
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
            print("[orbital] " + "  ".join(f"s{s}:v={_sv[s]:.2f}" for s in range(tv.sn)), flush=True)

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
        tv.px.diffuse(0.93)
        a = _frame[0] * 0.004
        c0 = [CX + ORBIT_R * math.cos(a), CY + ORBIT_R * math.sin(a)]
        c1 = [CX - ORBIT_R * math.cos(a * 1.3), CY - ORBIT_R * math.sin(a * 1.3)]
        tv.v.attract(tv.p, c0, MASS, A_RAD)
        tv.v.attract(tv.p, c1, MASS, A_RAD)
        tv.v.centripetal(tv.p, [CX, CY], 1, CENTRI)
        _damp_clamp()
        tv.v.move(tv.p, 1.0)
        _tick()
        tv.px.particles(tv.p, tv.s.species())
        _draw_rings()
        return tv.px

    return tv


if __name__ == "__main__":
    run(main)