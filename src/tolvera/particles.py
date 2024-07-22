"""Particle system.

The Tolvera particle system consists of a Particle class and a Particles class.
The Particle class is a Taichi dataclass for a single particle, and the Particles
class is a Taichi data_oriented class containing a Particle field. 

The Particles class also contains methods for processing the particle system, 
such as updating the particles, and getting and setting particle properties.
"""

import taichi as ti

from .species import Species
from .state import State
from .utils import CONSTS

@ti.dataclass
class Particle:
    """Particle data structure and methods."""
    species: ti.i32
    active: ti.f32
    pos: ti.math.vec2
    vel: ti.math.vec2
    ppos: ti.math.vec2
    pvel: ti.math.vec2
    mass: ti.f32
    size: ti.f32
    speed: ti.f32

    @ti.func
    def dist(self, other):
        """Distance between two particles.

        Args:
            other (Particle): Other particle.

        Returns:
            ti.math.vec2: Distance between the two particles.
        """
        return self.pos - other.pos

    @ti.func
    def dist_norm(self, other):
        """ti.math.norm() distance between two particles.

        Args:
            other (Particle): Other particle.

        Returns:
            ti.math.vec2: ti.math.norm() distance between the two particles.
        """
        return self.dist(self.pos - other.pos).norm()

    @ti.func
    def dist_normalized(self, other):
        """ti.math.normalized() distance between two particles.

        Args:
            other (Particle): Other particle.

        Returns:
            ti.math.vec2: ti.math.normalized() distance between the two particles.
        """
        return self.dist(self.pos - other.pos).normalized()

    @ti.func
    def dist_wrap(self, other, x, y):
        """Wrap around distance between two particles.

        Args:
            other (Particle): Other particle.
            x (float): Width.
            y (float): Height.
        
        Returns:
            ti.math.vec2: Wrap around distance between the two particles.
        """
        dx = self.pos[0] - other.pos[0]
        dy = self.pos[1] - other.pos[1]
        if abs(dx) > x / 2:  # x-axis
            dx = x - abs(dx)
            if self.pos[0] > other.pos[0]:
                dx = -dx
        if abs(dy) > y / 2:  # y-axis
            dy = y - abs(dy)
            if self.pos[1] > other.pos[1]:
                dy = -dy
        return ti.Vector([dx, dy])

    # @ti.func
    # def dist_wrap(self, other, x, y):
    #     dx = self.pos[0] - other.pos[0]
    #     dy = self.pos[1] - other.pos[1]
    #     # Wrap around for the x-axis
    #     if abs(dx) > x / 2:
    #         dx = x - abs(dx)
    #         if self.pos[0] < other.pos[0]:
    #             dx = -dx
    #     # Wrap around for the y-axis
    #     if abs(dy) > y / 2:
    #         dy = y - abs(dy)
    #         if self.pos[1] < other.pos[1]:
    #             dy = -dy
    #     return ti.Vector([dx, dy])
    # @ti.func
    # def dist_wrap(self, other, width, height):
    #     # Compute the element-wise absolute difference
    #     self_abs = ti.abs(self.pos)
    #     other_abs = ti.abs(other.pos)
    #     delta = self_abs - other_abs
    #     # Check if wrapping around is shorter for both the x and y components
    #     if delta[0] > width / 2:
    #         delta[0] = width - delta[0]
    #     if delta[1] > height / 2:
    #         delta[1] = height - delta[1]
    #     # Correct the signs if necessary
    #     if self.pos[0] > other.pos[0] and delta[0] > 0:
    #         delta[0] = -delta[0]
    #     if self.pos[1] > other.pos[1] and delta[1] > 0:
    #         delta[1] = -delta[1]
    #     return delta
    @ti.func
    def randomise(self, x, y):
        """Randomise the particle's position and velocity.

        Args:
            x (ti.f32): Width.
            y (ti.f32): Height.
        """
        self.randomise_pos(x, y)
        self.randomise_vel()

    @ti.func
    def randomise_pos(self, x, y):
        """Randomise the particle's position.

        Args:
            x (ti.f32): Width.
            y (ti.f32): Height.
        """
        self.pos = [x * ti.random(ti.f32), y * ti.random(ti.f32)]

    @ti.func
    def randomise_vel(self):
        """Randomise the particle's velocity."""
        self.vel = [2 * (ti.random(ti.f32) - 0.5), 2 * (ti.random(ti.f32) - 0.5)]


@ti.data_oriented
class Particles:
    """Particle system."""
    def __init__(self, tolvera, **kwargs):
        """Initialise the particle system.

        Args:
            tolvera (Tolvera): Tolvera instance.
            **kwargs: Keyword arguments (currently there are none).
        """
        self.tv = tolvera
        self.kwargs = kwargs
        self.n = self.tv.pn
        self.p_per_s = self.tv.p_per_s
        self._speed = ti.field(ti.f32, shape=())
        self._speed[None] = 1.0
        self.substep = self.tv.substep
        self.field = Particle.field(shape=(self.n))
        # TODO: These should be possible with State
        # self.pos = State(self.tv, {
        #     'x': (0., self.tv.x),
        #     'y': (0., self.tv.y),
        # }, shape=(self.n,), osc=('get'), name='particles_pos')
        self.C = CONSTS({"COLL_RAD": (ti.f32, 10.0)})
        self.tv.s.collisions_p = {
            'state': {
                'collision': (ti.i32, 0, 1),
                'dpos': (ti.math.vec2, 0., 1.),
                'dvel': (ti.math.vec2, 0., 1.),
            },
            'shape': self.n,
        }
        self.tmp_pos = ti.Vector.field(2, ti.f32, shape=(self.n))
        self.tmp_vel = ti.Vector.field(2, ti.f32, shape=(self.n))
        self.tmp_pos_species = ti.Vector.field(2, ti.f32, shape=(self.p_per_s))
        self.tmp_vel_species = ti.Vector.field(2, ti.f32, shape=(self.p_per_s))
        self.tmp_vel_stats = ti.Vector.field(1, ti.f32, shape=(7))
        self.active_indexes = ti.field(ti.i32, shape=(self.n))
        self.active_count = ti.field(ti.i32, shape=())
        self.init()

    def init(self):
        """Initialise the particle system."""
        self.assign_species()
        self.randomise()

    @ti.kernel
    def assign_species(self):
        """Assign species to particles."""
        for i in range(self.n):
            self.field[i].species = i % self.tv.species

    def _randomise(self):
        """Randomise the particle system (Python scope)."""
        self.randomise()

    @ti.kernel
    def randomise(self):
        """Randomise the particle system (Taichi scope)."""
        for i in range(self.n):
            si = self.field[i].species
            s = self.tv.s.species[si]
            # FIXME: ugly
            # c = self.tv.species_consts
            species = si
            active = 1.0
            pos = [self.tv.x * ti.random(ti.f32), self.tv.y * ti.random(ti.f32)]
            vel = [2 * (ti.random(ti.f32) - 0.5), 2 * (ti.random(ti.f32) - 0.5)]
            size = (
                ti.random(ti.f32) * s.size * self.tv.species_consts.MAX_SIZE
                + self.tv.species_consts.MIN_SIZE
            )
            speed = (
                ti.random(ti.f32) * s.speed * self.tv.species_consts.MAX_SPEED
                + self.tv.species_consts.MIN_SPEED
            )
            mass = ti.random(ti.f32) * s.mass * self.tv.species_consts.MAX_MASS
            self.field[i] = Particle(
                species=species,
                pos=pos,
                vel=vel,
                active=active,
                mass=mass,
                size=size,
                speed=speed,
            )

    @ti.kernel
    def update(self):
        """Update the particle system."""
        j = 0
        for i in range(self.n):
            if self.field[i] == 0.0: continue
            self.toroidal_wrap(i)
            self.limit_speed(i)
            # self.detect_collisions(i, self.C.COLL_RAD)
            # self.update_prev(i)
            self.active_indexes[j] = i
            j += 1
        self.active_count[None] = j

    @ti.func
    def toroidal_wrap(self, i: ti.i32):
        """Toroidal wrap a particle.

        Args:
            i (ti.i32): Particle index.
        """
        p = self.field[i]
        if p.pos[0] > self.tv.x:
            self.field[i].pos[0] = 0.0
        if p.pos[0] < 0.0:
            self.field[i].pos[0] = self.tv.x
        if p.pos[1] > self.tv.y:
            self.field[i].pos[1] = 0.0
        if p.pos[1] < 0.0:
            self.field[i].pos[1] = self.tv.y

    @ti.func
    def limit_speed(self, i: ti.i32):
        """Limit the speed of a particle.

        Args:
            i (ti.i32): Particle index.
        """
        p = self.field[i]
        s = self.tv.s.species[p.species]
        # FIXME: ugly
        sp = (
            s.speed * self.tv.species_consts.MAX_SPEED
            + self.tv.species_consts.MIN_SPEED
        )
        if p.vel.norm() > s.speed:
            self.field[i].vel = p.vel.normalized() * sp * self._speed[None]

    @ti.func
    def detect_collisions(self, i: ti.i32, radius: ti.f32):
        """Detect collisions between particles.

        TODO: Merge deltas into @ti.dataclass, or reimplement Particle.field as tv.s?
        TODO: Multiple collision states? Collided, Colliding, etc.
        TODO: Detect collisions between external objects.

        Args:
            i (ti.i32): Particle index.
            radius (ti.f32): Collision radius.
        """
        for j in range(self.n):
            p1, p2 = self.tv.p.field[i], self.tv.p.field[j]
            if p2.active == 0: continue
            dist = p1.pos - p2.pos
            if dist.norm() < radius:
                pdist = p1.ppos - p2.ppos
                dpos = ti.abs(pdist - dist)
                dvel = ti.abs((p1.pvel - p2.pvel) - (p1.vel - p2.vel))
                self.tv.s.collisions_p[i].dpos = dpos
                self.tv.s.collisions_p[i].dvel = dvel
                if pdist.norm() > radius:
                    self.tv.s.collisions_p[i].collision = 1
                else:
                    self.tv.s.collisions_p[i].collision = 0

    @ti.func
    def update_prev(self, i: ti.i32):
        """Update the previous position and velocity of a particle.

        Args:
            i (ti.i32): Particle index.
        """
        self.field[i].ppos = self.field[i].pos
        self.field[i].pvel = self.field[i].vel

    @ti.kernel
    def activity_decay(self):
        """Decay the activity of the particles."""
        for i in range(self.active_count[None]):
            idx = self.active_indexes[i]
            self.field[idx].active *= self.field[i].decay

    def process(self):
        """Process the particle system."""
        for i in range(self.substep):
            self.update()

    @ti.kernel
    def set_total_active(self, total: ti.i32):
        """Set the total number of active particles.

        Args:
            total (ti.i32): Total active particles.
        """
        for i in range(self.field.shape[0]):
            if i >= total:
                self.field[i].active = 0
            else:
                self.field[i].active = 1

    @ti.kernel
    def set_total_active_amount(self, total: ti.i32, amount: ti.f32):
        """Set the total number of active particles.

        Args:
            total (ti.i32): Total active particles.
            amount (ti.f32): Amount of activity.
        """
        for i in range(self.field.shape[0]):
            if i >= total:
                self.field[i].active = 0
            else:
                self.field[i].active = amount

    @ti.kernel
    def set_species_total_active(self, i: ti.i32, total: ti.i32):
        """Set the total number of active particles for a species.

        Args:
            i (ti.i32): Species index.
            total (ti.i32): Total active particles.
        """
        for j in range(self.field.shape[0]):
            if self.field[j].species == i:
                if j >= total:
                    self.field[j].active = 0
                else:
                    self.field[j].active = 1
    
    @ti.kernel
    def set_species_total_active_amount(self, i: ti.i32, total: ti.i32, amount: ti.f32):
        """Set particle activity amount of a species.

        Args:
            i (ti.i32): Species index.
            total: (ti.i32): Total number of active particles.
            amount (ti.i32): Amount of activity.
        """
        for j in range(self.field.shape[0]):
            if self.field[j].species == i:
                if j >= total:
                    self.field[j].active = 0
                else:
                    self.field[j].active = amount

    def set_pos(self, i, x, y):
        self.field[i].pos = [x, y]

    def set_vel(self, i, x, y):
        self.field[i].vel = [x, y]

    def set_speed(self, i, s):
        self.field[i].speed = s

    def set_size(self, i, s):
        self.field[i].size = s

    def get_pos(self, i):
        return self.field[i].pos.to_numpy().tolist()

    def get_vel(self, i):
        return self.field[i].vel.to_numpy().tolist()

    def get_pos_all_1d(self):
        self._get_pos_all()
        return self.tmp_pos.to_numpy().flatten().tolist()

    def get_pos_all_2d(self):
        self._get_pos_all()
        return self.tmp_pos.to_numpy().tolist()

    def get_vel_all_1d(self):
        self._get_vel_all()
        return self.tmp_vel.to_numpy().flatten().tolist()

    def get_vel_all_2d(self):
        self._get_vel_all()
        return self.tmp_vel.to_numpy().tolist()

    @ti.kernel
    def _get_pos_all(self):
        # for i in range(self.active_count[None]):
        #     idx = self.active_indexes[i]
        #     p = self.field[idx]
        #     self.tmp_pos[i] = p.pos / [self.tv.x, self.tv.y]
        # TODO: Only send active particle positions...? Or inactive=-1?
        for i in range(self.n):
            p = self.field[i]
            # if p.active > 0.0: # causes IML shape assertion error
            self.tmp_pos[i] = p.pos / [self.tv.x, self.tv.y]
            # else:
            #     self.tmp_pos[i] = [0.0,0.0] # ???

    @ti.kernel
    def _get_vel_all(self):
        for i in range(self.n):
            p = self.field[i]
            if p.active > 0.0:
                self.tmp_vel[i] = p.vel

    def get_pos_species_1d(self, species: int):
        self._get_pos_species()
        return self.tmp_pos_species.to_numpy().flatten().tolist()

    def get_pos_species_2d(self, species: int):
        if species > self.tv.species - 1:
            return
        self._get_pos_species(species)
        return self.tmp_pos_species.to_numpy().tolist()

    @ti.kernel
    def _get_pos_species(self, i: ti.i32):
        for j in range(self.n):
            si = j % self.tv.species
            p = self.field[j]
            if i == si and p.active > 0.0:
                species_index = (j - i) // self.tv.species
                pos = p.pos / [self.tv.x, self.tv.y]
                self.tmp_pos_species[species_index] = pos

    def get_vel_species_1d(self, species: int):
        self._get_vel_species(species)
        return self.tmp_vel_species.to_numpy().flatten().tolist()

    def get_vel_species_2d(self, species: int):
        self._get_vel_species(species)
        return self.tmp_vel_species.to_numpy().tolist()

    @ti.kernel
    def _get_vel_species(self, i: ti.i32):
        for j in range(self.n):
            si = j % self.tv.species
            p = self.field[j]
            if i == si and p.active > 0.0:
                species_index = (j - i) // self.tv.species
                vel = p.vel / [self.tv.x, self.tv.y]
                self.tmp_vel_species[species_index] = vel

    def get_vel_stats_species_1d(self, species):
        self._species_velocity_statistics(species)
        return self.tmp_vel_stats.to_numpy().flatten().tolist()

    @ti.kernel
    def _species_velocity_statistics(self, i: ti.i32):
        """
        Centre of Mass Velocity: This is the average velocity of all particles in the species.
        Relative Velocity: This is the average velocity of all particles in the species relative to the centre of mass velocity.
        Angular Momentum: This is the sum of the angular momentum of all particles, which is given by mass * velocity * radius for each particle.
        Kinetic Energy: This is the sum of the kinetic energy of all particles, which is given by 0.5 * mass * velocity^2 for each particle.
        Temperature: In statistical mechanics, the temperature of a system of particles is related to the average kinetic energy of the particles.
        """
        centre_of_mass_velocity = ti.Vector([0.0, 0.0])
        relative_velocity = ti.Vector([0.0, 0.0])
        angular_momentum = ti.Vector([0.0])
        kinetic_energy = ti.Vector([0.0])
        for j in range(self.n):
            if self.field[j].species == i:
                v = self.field[j].vel
                p = self.field[j].pos
                m = self.field[j].mass
                centre_of_mass_velocity += v
                relative_velocity += v  # - centre_of_mass_velocity
                angular_momentum += m * ti.math.cross(v, p)
                kinetic_energy += 0.5 * m * v.norm_sqr()
        centre_of_mass_velocity = centre_of_mass_velocity / self.n_per_species
        relative_velocity = (
            relative_velocity - centre_of_mass_velocity * self.n_per_species
        ) / self.n_per_species
        temperature = 2.0 * kinetic_energy / (self.particles_per_species * 1.380649e-23)
        self.tmp_vel_stats[0] = centre_of_mass_velocity[0]
        self.tmp_vel_stats[1] = centre_of_mass_velocity[1]
        self.tmp_vel_stats[2] = relative_velocity[0]
        self.tmp_vel_stats[3] = relative_velocity[1]
        self.tmp_vel_stats[4] = angular_momentum[0]
        self.tmp_vel_stats[5] = kinetic_energy[0]
        self.tmp_vel_stats[6] = temperature[0]

    def reset(self):
        """Reset the particle system."""
        self.init()

    def speed(self, speed: float = None):
        """Get or set the speed of the particle system.

        Args:
            speed (float, optional): Speed. Defaults to None.
        
        Returns:
            float: Speed.
        """
        if speed is not None:
            self._speed[None] = 1 / (speed + 0.0001)
        else:
            return self._speed[None]

    def __call__(self):
        """Call will process the particle system."""
        self.process()
