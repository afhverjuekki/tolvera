"""
Based on https://www.complexity-explorables.org/explorables/swarmalators/
"""

import taichi as ti

from ..utils import CONSTS


@ti.data_oriented
class Swarmalators:
    def __init__(self, tolvera, **kwargs):
        self.tv = tolvera
        self.kwargs = kwargs

        self.CONSTS = CONSTS(
            {
                "dt": (ti.f32, 0.1),
                "dtt": (ti.f32, ti.math.sqrt(0.1)),
                "radius": (ti.f32, 4.0),
                "L": (ti.f32, 1.3),
                "two_L": (ti.f32, 2.6),
                "omega": (ti.f32, 0.44),
                "two_pi": (ti.f32, ti.math.pi * 2),
            }
        )

        self.tv.s.swarm_p = {
            "state": {
                "x": (ti.f32, 0.0, 1.0),
                "y": (ti.f32, 0.0, 1.0),
                "vx": (ti.f32, 0.0, 1.0),
                "vy": (ti.f32, 0.0, 1.0),
                "dx": (ti.f32, 0.0, 1.0),
                "dy": (ti.f32, 0.0, 1.0),
                "omega": (ti.f32, 0.0, 1.0),
                "domega": (ti.f32, 0.0, 1.0),
                "theta": (ti.f32, 0.0, 1.0),
                "dtheta": (ti.f32, 0.0, 1.0),
                "color": (ti.f32, 0.0, 1.0),
            },
            "shape": self.tv.pn,
            "randomise": False,
        }

        self.tv.s.swarm_s = {
            "state": {
                "J": (ti.f32, -1.0, 1.0),  # coupling strength
                "K": (ti.f32, -1.0, 1.0),  # synchronisation strength
                "noise": (ti.f32, 0.0, 0.05),  # wiggle
                "omega": (ti.f32, 0.01, 1.0),  # frequency variation
                "speed": (ti.f32, 0.01, 2.0),  # normal speed
            },
            "shape": 7,
            "randomise": False,
        }

        self.phasemod = ti.field(ti.f32, shape=())
        self.phasemod[None] = 0

        self.init()

    @ti.kernel
    def init(self):
        self.set_presets()
        self.reset()

    @ti.func
    def set_presets(self):
        # "Makes Me Dizzy"
        self.tv.s.swarm_s[0] = self.tv.s.swarm_s.struct(
            J=1.0, K=0.51, noise=0.0, omega=0.4439, speed=2.0
        )
        # "Rainbow Ring"
        self.tv.s.swarm_s[1] = self.tv.s.swarm_s.struct(
            J=0.5, K=0.0, noise=0.0, omega=0.0, speed=2.0
        )
        # "Dancing Circus"
        self.tv.s.swarm_s[2] = self.tv.s.swarm_s.struct(
            J=0.93, K=-0.88, noise=0.0, omega=0.0, speed=2.0
        )
        # "Uniform Blob"
        self.tv.s.swarm_s[3] = self.tv.s.swarm_s.struct(
            J=0.1, K=1.0, noise=0.0, omega=0.0, speed=2.0
        )
        # "Solar Convection"
        self.tv.s.swarm_s[4] = self.tv.s.swarm_s.struct(
            J=0.1, K=1.0, noise=0.0, omega=0.8, speed=2.0
        )
        # "Fractions"
        self.tv.s.swarm_s[5] = self.tv.s.swarm_s.struct(
            J=1.0, K=-0.12, noise=0.0, omega=0.0, speed=2.0
        )
        # ???
        self.tv.s.swarm_s[6] = self.tv.s.swarm_s.struct(
            J=0.0, K=0.0, noise=0.0, omega=0.0, speed=2.0
        )

    @ti.func
    def reset(self):
        mvx = 0.0
        mvy = 0.0
        for i in range(self.tv.pn):
            knut = self.CONSTS.two_pi * ti.random(ti.f32)
            self.tv.s.swarm_p[i] = self.tv.s.swarm_p.struct(
                self.CONSTS.two_L * (ti.random(ti.f32) - 0.5),  # x
                self.CONSTS.two_L * (ti.random(ti.f32) - 0.5),  # y
                ti.math.cos(knut),  # vx
                ti.math.sin(knut),  # vy
                0.0,  # dx
                0.0,  # dy
                self.CONSTS.omega,  # omega
                ti.randn(),  # domega
                self.CONSTS.two_pi * ti.random(ti.f32),  # theta
                0.0,  # dtheta
                0.0,  # color
            )
            mvx += self.tv.s.swarm_p[i].vx
            mvy += self.tv.s.swarm_p[i].vy
        mvx /= self.tv.pn
        mvy /= self.tv.pn
        for i in range(self.tv.pn):
            self.tv.s.swarm_p[i].vx -= mvx
            self.tv.s.swarm_p[i].vy -= mvy

    @ti.kernel
    def perturb(self):
        C = self.CONSTS
        for i in range(self.tv.pn):
            p = self.tv.s.swarm_p[i]
            w = self.CONSTS.two_pi * ti.random(ti.f32)
            p.x += C.radius * ti.math.cos(w)
            p.y += C.radius * ti.math.sin(w)
            p.theta += ti.math.pi * (ti.random(ti.f32) - 0.25 * C.two_pi)
            self.tv.s.swarm_p[i] = p

    @ti.kernel
    def step(self, particles: ti.template(), preset: ti.i32, weight: ti.f32):

        pn = self.tv.pn
        s = self.tv.s.swarm_s[preset]

        for i in range(pn):
            p1 = self.tv.s.swarm_p[i]

            p1.dx = p1.vx * s.speed
            p1.dy = p1.vy * s.speed
            p1.dtheta = p1.omega * self.phasemod[None] + s.omega * p1.domega

            for j in range(pn):
                if i == j:
                    continue
                p2 = self.tv.s.swarm_p[j]
                d = self.dist_euclid(p1, p2)
                kernel = (
                    1 + s.J * ti.math.cos(p2.theta - p1.theta) / d - 1.0 / (d * d)
                ) / pn
                p1.dx += (p2.x - p1.x) * kernel
                p1.dy += (p2.y - p1.y) * kernel
                p1.dtheta += s.K / pn * ti.math.sin(p2.theta - p1.theta) / d

            p1.dx *= self.CONSTS.dt
            p1.dy *= self.CONSTS.dt
            p1.dtheta *= self.CONSTS.dt

            p1.x += p1.dx + self.CONSTS.dtt * s.noise * (ti.random(ti.f32) - 0.5)
            p1.y += p1.dy + self.CONSTS.dtt * s.noise * (ti.random(ti.f32) - 0.5)
            p1.theta += p1.dtheta

            p1.color = p1.theta / 2 / ti.math.pi
            self.tv.s.swarm_p[i] = p1
            pos = ti.Vector([self.X(p1.x), self.Y(p1.y)])
            d = pos - particles[i].pos
            particles[i].pos += d * weight

    @ti.func
    def dist_euclid(self, p1, p2):
        return ti.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

    @ti.func
    def X(self, x):
        return (x + self.CONSTS.L) / self.CONSTS.two_L * self.tv.x / 2 + self.tv.x / 4

    @ti.func
    def Y(self, y):
        return (y + self.CONSTS.L) / self.CONSTS.two_L * self.tv.y / 2 + self.tv.y / 4

    def __call__(self, particles, preset: ti.i32 = 0, weight: ti.f32 = 1.0):
        self.step(particles.field, preset, weight)
