"""
Inspired by https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/physarum.py
"""

import taichi as ti

from ..pixels import Pixels
from ..state import State
from ..utils import CONSTS


@ti.data_oriented
class Slime:
    def __init__(self, tolvera, evaporate: ti.f32 = 0.99, **kwargs):
        self.tv = tolvera
        self.kwargs = kwargs
        brightness = kwargs.get("brightness", 1.0)
        self.CONSTS = CONSTS(
            {
                "SENSE_ANGLE": (ti.f32, ti.math.pi * 0.3),
                "SENSE_DIST": (ti.f32, 50.0),
                "MOVE_ANGLE": (ti.f32, ti.math.pi * 0.3),
                "MOVE_DIST": (ti.f32, 4.0),
                "SUBSTEP": (ti.i32, 1),
                "BRIGHTNESS": (ti.f32, brightness),
            }
        )
        self.tv.s.slime_p = {
            "state": {
                "sense_angle": (ti.f32, 0.0, 10.0),
                "sense_left": (ti.math.vec4, 0.0, 10.0),
                "sense_centre": (ti.math.vec4, 0.0, 10.0),
                "sense_right": (ti.math.vec4, 0.0, 10.0),
            },
            "shape": self.tv.pn,
            "osc": ("get"),
            "randomise": True,
        }
        self.tv.s.slime_s = {
            "state": {
                "sense_angle": (ti.f32, 0.0, 1.0),
                "sense_dist": (ti.f32, 0.0, 1.0),
                "move_angle": (ti.f32, 0.0, 1.0),
                "move_dist": (ti.f32, 0.0, 1.0),
                "evaporate": (ti.f32, 0.0, 1.0),
            },
            "shape": self.tv.sn,  # multi-species: (self.tv.sn, self.tv.sn),
            "osc": ("set"),
            "randomise": True,
        }
        self.trail = Pixels(self.tv, **kwargs)
        self.evaporate = ti.field(dtype=ti.f32, shape=())
        self.evaporate[None] = evaporate

    def randomise(self):
        self.tv.s.slime_s.randomise()
        self.tv.s.slime_p.randomise()

    @ti.kernel
    def move(self, field: ti.template(), weight: ti.f32):
        for i in range(field.shape[0]):
            if field[i].active == 0.0:
                continue

            p = field[i]
            ang = self.tv.s.slime_p[i].sense_angle
            species = self.tv.s.slime_s[p.species]

            sense_angle = species.sense_angle * self.CONSTS.SENSE_ANGLE
            sense_dist = species.sense_dist * self.CONSTS.SENSE_DIST
            move_angle = species.move_angle * self.CONSTS.MOVE_ANGLE
            move_dist = species.move_dist * self.CONSTS.MOVE_DIST

            c = self.sense(p.pos, ang, sense_dist).norm()
            l = self.sense(p.pos, ang - sense_angle, sense_dist).norm()
            r = self.sense(p.pos, ang + sense_angle, sense_dist).norm()

            if l < c < r:
                ang += move_angle
            elif l > c > r:
                ang -= move_angle
            elif r > c and c < l:
                # TODO: magic numbers, move to @ti.func inside utils?
                ang += move_angle * (2 * (ti.random() < 0.5) - 1)

            p.pos += (
                ti.Vector([ti.cos(ang), ti.sin(ang)]) * move_dist * p.active * weight
            )

            self.tv.s.slime_p[i].sense_angle = ang
            self.tv.s.slime_p[i].sense_centre = c
            self.tv.s.slime_p[i].sense_left = l
            self.tv.s.slime_p[i].sense_right = r
            field[i].pos = p.pos

    @ti.func
    def sense(self, pos: ti.math.vec2, ang: ti.f32, dist: ti.f32) -> ti.math.vec4:
        ang_cos = ti.cos(ang)
        ang_sin = ti.sin(ang)
        v = ti.Vector([ang_cos, ang_sin])
        p = pos + v * dist
        px = ti.cast(p[0], ti.i32) % self.tv.x
        py = ti.cast(p[1], ti.i32) % self.tv.y
        pixel = self.trail.px.rgba[px, py]
        return pixel

    @ti.func
    def sense_rgba(self, pos, ang, dist, rgba):
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * dist
        px = ti.cast(p[0], ti.i32) % self.tv.x
        py = ti.cast(p[1], ti.i32) % self.tv.y
        px_rgba = self.trail.px.rgba[px, py]
        px_rgba_weighted = px_rgba * (1.0 - (px_rgba - rgba).norm())
        return px_rgba_weighted

    @ti.kernel
    def deposit_particles(self, particles: ti.template(), species: ti.template()):
        for i in range(particles.shape[0]):
            if particles[i].active == 0.0:
                continue
            p, s = particles[i], species[particles[i].species]
            x = ti.cast(p.pos[0], ti.i32) % self.tv.x
            y = ti.cast(p.pos[1], ti.i32) % self.tv.y
            rgba = s.rgba * self.CONSTS.BRIGHTNESS * p.active
            self.trail.circle(x, y, p.size, rgba)

    def step(self, particles, species, weight: ti.f32 = 1.0):
        for i in range(self.CONSTS.SUBSTEP):
            self.move(particles.field, weight)
            self.deposit_particles(particles.field, species)
            self.trail.diffuse(self.evaporate[None])

    def __call__(self, particles, species, weight: ti.f32 = 1.0):
        self.step(particles, species, weight)
        return self.trail
