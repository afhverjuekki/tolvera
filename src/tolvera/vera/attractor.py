"""
TODO: why is mass so sensitive?
TODO: render()
TODO: more neighbour funcs/stats
TODO: move particles.seek|avoid here? (attract|repel)
TODO: rename to Target?
TODO: wall behaviour
TODO: should be a particle field itself? to use move functions etc
"""

import taichi as ti

from tolvera.particles import Particle, Particles

"""
@ti.kernel
def attract(tv: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
    for i in range(tv.o.n):
        p = tv.p.field[i]
        if p.active == 0: continue
        target_distance = (pos-p.pos).norm()
        if target_distance < radius:
            factor = (radius-target_distance)/radius
            tv.p.field[i].vel += (pos-p.pos).normalized() * mass * factor
attract_kernel = attract

@ti.kernel
def attract_species(tv: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32, species: ti.i32):
    for i in range(tv.o.n):
        p = tv.p.field[i]
        if p.active == 0: continue
        if p.species != species: continue
        target_distance = (pos-p.pos).norm()
        if target_distance < radius:
            factor = (radius-target_distance)/radius
            tv.p.field[i].vel += (pos-p.pos).normalized() * mass * factor
attract_species_kernel = attract_species

@ti.kernel
def repel(tv: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
   for i in range(tv.o.n):
        p = tv.p.field[i]
        if p.active == 0: continue
        target_distance = (pos-p.pos).norm()
        if target_distance < radius:
            factor = (target_distance-radius)/radius
            tv.p.field[i].vel += (pos-p.pos).normalized() * mass * factor
repel_kernel = repel
"""


@ti.dataclass
class Attractor:
    p: Particle
    radius: ti.f32


@ti.data_oriented
class Attractors:
    def __init__(self, x=1920, y=1080, n=1) -> None:
        self.x = x
        self.y = y
        self.n = n
        self.field = Attractor.field(shape=(n))
        self.particles = Particles(x, y, n, 1)
        self.randomise()

    def set(self, i, attractor: Attractor):
        self.field[i] = attractor

    def get(self, i):
        return self.field[i]

    @ti.kernel
    def randomise(self):
        for i in range(self.n):
            self.field[i].p.vel = ti.Vector([ti.random() * 2 - 1, ti.random() * 2 - 1])
            self.field[i].p.pos = ti.Vector(
                [
                    (0.2 * self.x) + ti.random() * (0.8 * self.x),
                    (0.2 * self.y) + ti.random() * (0.8 * self.y),
                ]
            )
            self.field[i].p.mass = ti.random() * 1.0
            self.field[i].p.active = 1.0
            self.field[i].p.speed = 1.0
            self.field[i].p.max_speed = 2.0
            self.field[i].radius = ti.random() * self.y

    @ti.kernel
    def nn(self, field: ti.template()):
        for i in range(self.n):
            if self.field[i].p.active > 0.0:
                self.nn_inner(field, i)

    @ti.func
    def nn_inner(self, field: ti.template(), i: ti.i32):
        a = self.field[i]
        nearby = 0
        for j in range(field.shape[0]):
            p = field[j]
            if p.active > 0.0:
                dis = self.field[i].p.dist(p).norm()
                if dis < a.radius:
                    nearby += 1
        if nearby != 0:
            self.field[i].p.nearby = nearby

    # @ti.kernel
    # def render(self, pixels):
    #     # draw circle with no fill
    #     # color based on p.mass
    #     # fill based on p.nearby
    #     # radius based on radius
    #     pass
    def update(self, i, px, py, d, w):
        # self.field[i].p.vel[0] += vx
        # self.field[i].p.vel[1] += vy
        # self.particles.field[i].vel[0] += vx
        # self.particles.field[i].vel[0] += vy
        self.field[i].p.pos[0] = px
        self.field[i].p.pos[1] = py
        self.field[i].radius = d
        self.field[i].p.mass = w
        self.particles.field[i].pos[0] = px
        self.particles.field[i].pos[0] = py

    @ti.kernel
    def process(self):
        for i in range(self.n):
            self.field[i].p.pos = self.particles.field[i].pos

    def __call__(self, particles):
        # self.particles()
        # self.process()
        self.nn(particles.field)
