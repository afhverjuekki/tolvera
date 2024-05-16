"""Particle Life model."""

import taichi as ti

from ..utils import CONSTS

@ti.data_oriented
class ParticleLife():
    """Particle Life model.

    The Particle Life model is a simple model of particle behaviour, where
    particles are either attracted or repelled by other particles, depending
    on their species. Popularised by Jeffrey Ventrella (Clusters), Tom Mohr
    and others:

    https://www.ventrella.com/Clusters/
    https://github.com/tom-mohr/particle-life-app
    
    """
    def __init__(self, tolvera, **kwargs) -> None:
        """Initialise the Particle Life model.

        'plife' stores the species rule matrix.

        Args:
            tolvera (Tolvera): A Tolvera instance.
            **kwargs: Keyword arguments (currently none).
        """
        self.tv = tolvera
        self.kwargs = kwargs
        self.CONSTS = CONSTS({
            "V": (ti.f32, 0.25),
        })
        self.tv.s.plife = {
            "state": {
                "attract": (ti.f32, -.5, .5),
                "radius": (ti.f32, 100., 300.0),
            },
            "shape": (self.tv.sn, self.tv.sn),
            "randomise": True,
        }
    @ti.kernel
    def step(self, particles: ti.template(), weight: ti.f32):
        """Step the Particle Life model.

        Args:
            particles (Particles.field): The particles to step.
            weight (ti.f32): The weight of the step.
        """
        for i in range(particles.shape[0]):
            if particles[i].active == 0.: continue
            p1 = particles[i]
            fx, fy = 0., 0.
            for j in range(particles.shape[0]):
                if particles[j].active == 0.: continue
                p2 = particles[j]
                s = self.tv.s.plife[p1.species, p2.species]
                dx = p1.pos[0] - p2.pos[0]
                dy = p1.pos[1] - p2.pos[1]
                d = ti.sqrt(dx*dx + dy*dy)
                if 0. < d and d < s.radius:
                    F = s.attract/d
                    fx += F*dx
                    fy += F*dy
            # particles[i].vel = (particles[i].vel + ti.Vector([fx, fy])) * self.CONSTS.V * weight
            # particles[i].pos += (particles[i].vel * p1.speed * p1.active * weight)
            particles[i].vel = (particles[i].vel + ti.Vector([fx, fy])) * self.CONSTS.V * weight * p1.speed * p1.active
            particles[i].pos += particles[i].vel
    def __call__(self, particles, weight: ti.f32 = 1.0):
        """Call the Particle Life model.

        Args:
            particles (Particles): The particles to step.
        """
        self.step(particles.field, weight)
