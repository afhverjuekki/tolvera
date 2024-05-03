"""Slime behaviour based on the Physarum polycephalum slime mould."""

import taichi as ti

from ..pixels import Pixels
from ..utils import CONSTS


@ti.data_oriented
class Slime:
    """Slime behaviour based on the Physarum polycephalum slime mould.
    
    The slime mould is a single-celled organism that exhibits complex behaviour
    such as foraging, migration, and decision-making. It is a popular model for
    emergent behaviour in nature-inspired computing.
    
    The slime mould is simulated by a set of particles that move around the
    simulation space. The particles sense their environment and move in response
    to the sensed information. The particles leave a "pheromone trail" behind them,
    which evaporates over time. The particles can be of different species, which 
    have different sensing and moving parameters.
    
    Taichi Physarum implementation inspired by:
    https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/physarum.py
    """
    def __init__(self, tolvera, **kwargs):
        """Initialise the Slime behaviour.

        `slime_p` stores the particle state.
        `slime_s` stores the species state.
        `trail` is a Pixels instance that stores the pheromone trail.
        
        Args:
            tolvera (Tolvera): A Tolvera instance.
            evaporate (ti.f32, optional): Evaporation rate. Defaults to 0.99.
            **kwargs: Keyword arguments.
                brightness (ti.f32, optional): Brightness of the pheromone trail. Defaults to 1.0.
        """
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
        self.evaporate[None] = kwargs.get("evaporate", 0.99)

    def randomise(self):
        """Randomise the Slime behaviour."""
        self.tv.s.slime_s.randomise()
        self.tv.s.slime_p.randomise()

    @ti.kernel
    def move(self, field: ti.template(), weight: ti.f32):
        """Move the particles based on the sensed environment.

        Each particle senses the trail to its left, centre and right. Depending on the 
        strength of the sensed trail in each direction, and the species parameters,
        a movement angle is calculated. The particle moves in this direction by a 
        distance proportional to its active state and the weight parameter.

        Args:
            field (ti.template): Particle field.
            weight (ti.f32): Weight of the movement.
        """
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
        """Sense the trail at a given position and angle.

        Args:
            pos (ti.math.vec2): Position.
            ang (ti.f32): Angle.
            dist (ti.f32): Distance.

        Returns:
            ti.math.vec4: RGBA value of the sensed trail point.
        """
        ang_cos = ti.cos(ang)
        ang_sin = ti.sin(ang)
        v = ti.Vector([ang_cos, ang_sin])
        p = pos + v * dist
        px = ti.cast(p[0], ti.i32) % self.tv.x
        py = ti.cast(p[1], ti.i32) % self.tv.y
        pixel = self.trail.px.rgba[px, py]
        return pixel

    @ti.func
    def sense_rgba(self, pos: ti.math.vec2, ang: ti.f32, dist: ti.f32, rgba: ti.math.vec4) -> ti.math.vec4:
        """Sense the trail at a given position and angle and return a weighted RGBA value.

        Args:
            pos (ti.math.vec2): Position.
            ang (ti.f32): Angle.
            dist (ti.f32): Distance.
            rgba (ti.math.vec4): RGBA value.
        
        Returns:
            ti.math.vec4: Weighted RGBA value.
        """
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * dist
        px = ti.cast(p[0], ti.i32) % self.tv.x
        py = ti.cast(p[1], ti.i32) % self.tv.y
        px_rgba = self.trail.px.rgba[px, py]
        px_rgba_weighted = px_rgba * (1.0 - (px_rgba - rgba).norm())
        return px_rgba_weighted

    @ti.kernel
    def deposit_particles(self, particles: ti.template(), species: ti.template()):
        """Deposit particles onto the trail.

        Args:
            particles (ti.template): Particle field.
            species (ti.template): Species field.
        """
        for i in range(particles.shape[0]):
            if particles[i].active == 0.0:
                continue
            p, s = particles[i], species[particles[i].species]
            x = ti.cast(p.pos[0], ti.i32) % self.tv.x
            y = ti.cast(p.pos[1], ti.i32) % self.tv.y
            rgba = s.rgba * self.CONSTS.BRIGHTNESS * p.active
            self.trail.circle(x, y, p.size, rgba)

    def step(self, particles, species, weight: ti.f32 = 1.0):
        """Step the Slime behaviour.

        Args:
            particles (Particles): A Particles instance.
            species (Species): A Species instance.
            weight (ti.f32, optional): Weight parameter. Defaults to 1.0.
        """
        for i in range(self.CONSTS.SUBSTEP):
            self.move(particles.field, weight)
            self.deposit_particles(particles.field, species)
            self.trail.diffuse(self.evaporate[None])

    def __call__(self, particles, species, weight: ti.f32 = 1.0):
        """Call the Slime behaviour.

        Args:
            particles (Particles): A Particles instance.
            species (Species): A Species instance.
            weight (ti.f32, optional): Weight parameter. Defaults to 1.0.

        Returns:
            Pixels: A Pixels instance containing the pheromone trail.
        """
        self.step(particles, species, weight)
        return self.trail
