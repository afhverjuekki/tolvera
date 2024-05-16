"""
Inspired by https://github.com/taichi-dev/faster-python-with-taichi/blob/main/reaction_diffusion_taichi.py
"""

import numpy as np
import taichi as ti

from ..pixels import Pixels
from ..state import State
from ..utils import CONSTS


@ti.data_oriented
class ReactionDiffusion:
    def __init__(self, tolvera, **kwargs):
        self.tv = tolvera
        self.kwargs = kwargs
        self.uv = ti.Vector.field(2, ti.f32, shape=(2, self.tv.x, self.tv.y))
        self.colors = ti.Vector.field(4, ti.f32, shape=(5,))
        self.field = Pixels(self.tv, **kwargs)
        self.tv.s.rd = {
            "state": {
                "Du": (ti.f32, 0.0, 1.0),
                "Dv": (ti.f32, 0.0, 1.0),
                "feed": (ti.f32, 0.0, 1.0),
                "kill": (ti.f32, 0.0, 1.0),
                "substep": (ti.i32, 0, 100),
            },
            "shape": 1,
            "osc": ("set"),
            "randomise": False,
        }
        self.tv.s.rd.field[0] = self.tv.s.rd.struct(
            kwargs.get("Du", 0.160),
            kwargs.get("Dv", 0.080),
            kwargs.get("feed", 0.060),
            kwargs.get("kill", 0.062),
            kwargs.get("substep", 18),
        )
        self.init()

    def init(self):
        self.randomise()
        self.make_palette()
        if self.tv.osc is not False:
            self.add_to_osc_map()

    def reset(self):
        self.uv.fill(0.0)

    def randomise(self):
        self.uv_grid = np.zeros((2, self.tv.x, self.tv.y, 2), dtype=np.float32)
        self.uv_grid[0, :, :, 0] = 1.0
        rand_rows = np.random.choice(range(self.tv.x), 50)
        rand_cols = np.random.choice(range(self.tv.y), 50)
        self.uv_grid[0, rand_rows, rand_cols, 1] = 1.0
        self.uv.from_numpy(self.uv_grid)

    def add_to_osc_map(self):
        self.tv.osc.map.receive_args_inline(
            self.tv.s.rd.setter_name + "_reset", self.reset
        )
        self.tv.osc.map.receive_args_inline(
            self.tv.s.rd.setter_name + "_randomise_uv", self.randomise
        )

    def make_palette(self):
        self.colors[0] = [0.0, 0.0, 0.0, 0.3137]
        self.colors[1] = [1.0, 0.1843, 0.53333, 0.37647]
        self.colors[2] = [0.8549, 1.0, 0.53333, 0.388]
        self.colors[3] = [0.376, 1.0, 0.478, 0.392]
        self.colors[4] = [1.0, 1.0, 1.0, 1]

    @ti.kernel
    def deposit_particles(self, particles: ti.template()):
        for i in range(particles.field.shape[0]):
            p = particles.field[i]
            if p.active == 0.0:
                continue
            self.uv[0, int(p.pos.x), int(p.pos.y)] += [0.1, 0.0]

    @ti.kernel
    def compute(self, phase: int):
        p = self.tv.s.rd.field[0]
        for i, j in ti.ndrange(self.tv.x, self.tv.y):
            cen = self.uv[phase, i, j]
            lapl = (
                self.uv[phase, i + 1, j]
                + self.uv[phase, i, j + 1]
                + self.uv[phase, i - 1, j]
                + self.uv[phase, i, j - 1]
                - 4.0 * cen
            )
            du = p.Du * lapl[0] - cen[0] * cen[1] * cen[1] + p.feed * (1 - cen[0])
            dv = p.Dv * lapl[1] + cen[0] * cen[1] * cen[1] - (p.feed + p.kill) * cen[1]
            val = cen + 0.5 * ti.math.vec2(du, dv)
            self.uv[1 - phase, i, j] = val

    @ti.kernel
    def render(self):
        for i, j in ti.ndrange(self.tv.x, self.tv.y):
            value = self.uv[0, i, j].y
            color = ti.math.vec3(0)
            # if value <= self.colors[0].w:
            #     color = self.colors[0].xyz
            for k in range(4):
                c0 = self.colors[k]
                c1 = self.colors[k + 1]
                if c0.w < value < c1.w:
                    a = (value - c0.w) / (c1.w - c0.w)
                    color = ti.math.mix(c0.xyz, c1.xyz, a)
            self.field.px.rgba[i, j] = [color.x, color.y, color.z, 1.0]

    def process(self):
        # for _ in range(self.substep[None]):
        for _ in range(self.tv.s.rd.field[0].substep):
            self.compute(self.tv.ctx.i[None] % 2)

    def __call__(self):
        self.process()
        self.render()
        return self.field.px
