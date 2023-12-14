"""
Inspired by https://forum.taichi-lang.cn/t/homework0-boids/563
"""
import taichi as ti

from ..state import State
from ..utils import CONSTS


@ti.data_oriented
class Flock:
    def __init__(self, tolvera, **kwargs):
        self.tv = tolvera
        self.kwargs = kwargs
        self.CONSTS = CONSTS({"MAX_RADIUS": (ti.f32, 300.0)})
        self.tv.s.flock_p = {
            "state": {
                "separate": (ti.math.vec2, 0.0, 1.0),
                "align": (ti.math.vec2, 0.0, 1.0),
                "cohere": (ti.math.vec2, 0.0, 1.0),
                "nearby": (ti.i32, 0.0, self.tv.p.n - 1),
            },
            "shape": self.tv.pn,
            "osc": ("get"),
            "randomise": False,
        }
        self.tv.s.flock_s = {
            "state": {
                "separate": (ti.f32, 0.01, 1.0),
                "align": (ti.f32, 0.01, 1.0),
                "cohere": (ti.f32, 0.01, 1.0),
                "radius": (ti.f32, 0.01, 1.0),
            },
            "shape": (self.tv.sn, self.tv.sn),
            "osc": ("set"),
            "randomise": True,
        }
        self.tv.s.flock_dist = {
            "state": {
                "dist": (ti.f32, 0.0, self.tv.x * 2),
                "dist_wrap": (ti.f32, 0.0, self.tv.x * 2),
            },
            "shape": (self.tv.pn, self.tv.pn),
            "osc": ("get"),
            "randomise": False,
        }

    def randomise(self):
        self.tv.s.flock_s.randomise()

    @ti.kernel
    def step(self, particles: ti.template(), weight: ti.f32):
        n = particles.shape[0]
        for i in range(n):
            if particles[i].active == 0:
                continue
            p1 = particles[i]
            separate = ti.Vector([0.0, 0.0])
            align = ti.Vector([0.0, 0.0])
            cohere = ti.Vector([0.0, 0.0])
            nearby = 0
            species = self.tv.s.flock_s.struct()
            for j in range(n):
                if i == j and particles[j].active == 0:
                    continue
                p2 = particles[j]
                species = self.tv.s.flock_s[p1.species, p2.species]
                dis_wrap = p1.dist_wrap(p2, self.tv.x, self.tv.y)
                dis_wrap_norm = dis_wrap.norm()
                if dis_wrap_norm < species.radius * self.CONSTS.MAX_RADIUS:
                    separate += dis_wrap
                    align += p2.vel
                    cohere += p2.pos
                    nearby += 1
                self.tv.s.flock_dist[i, j].dist = p1.dist(p2).norm()
                self.tv.s.flock_dist[i, j].dist_wrap = dis_wrap_norm
            if nearby > 0:
                separate = (
                    separate / nearby * p1.active * ti.math.max(species.separate, 0.2)
                )
                align = align / nearby * p1.active * species.align
                cohere = (cohere / nearby - p1.pos) * p1.active * species.cohere
                vel = (separate + align + cohere).normalized()
                particles[i].vel += vel * weight
                particles[i].pos += particles[i].vel * p1.speed * p1.active
            self.tv.s.flock_p[i] = self.tv.s.flock_p.struct(
                separate, align, cohere, nearby
            )

    def __call__(self, particles, weight: ti.f32 = 1.0):
        self.step(particles.field, weight)
