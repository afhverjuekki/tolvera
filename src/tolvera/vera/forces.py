"""Force functions for particles.

This module contains functions for applying forces to particles.
It includes functions for moving, attracting, repelling and gravitating particles.
It also includes variations of these functions for specific species of particles.
"""

import taichi as ti

from ..particles import Particle

__all__ = [
    "move",
    "attract",
    "_attract",
    "attract_species",
    "_attract_species",
    "attract_particle",
    "repel",
    "repel_species",
    "gravitate",
    "gravitate_species",
    "noise",
    "centripetal",
    "centripetal_particle",
]


@ti.kernel
def move(particles: ti.template(), weight: ti.f32):
    """Move the particles.

    Args:
        particles (ti.template): Particles.
    """
    for i in range(particles.field.shape[0]):
        if particles.field[i].active == 0:
            continue
        p1 = particles.field[i]
        particles.field[i].pos += p1.vel * p1.speed * p1.active * weight


@ti.kernel
def attract(particles: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
    """Attract the particles to a position.

    Args:
        particles (ti.template): Particles.
        pos (ti.math.vec2): Attraction position.
        mass (ti.f32): Attraction mass.
        radius (ti.f32): Attraction radius.
    """
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        particles.field[i].vel += attract_particle(p, pos, mass, radius)

@ti.func
def _attract(particles: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
    """Attract the particles to a position.

    Args:
        particles (ti.template): Particles.
        pos (ti.math.vec2): Attraction position.
        mass (ti.f32): Attraction mass.
        radius (ti.f32): Attraction radius.
    """
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        particles.field[i].vel += attract_particle(p, pos, mass, radius)


@ti.kernel
def attract_species(
    particles: ti.template(),
    pos: ti.math.vec2,
    mass: ti.f32,
    radius: ti.f32,
    species: ti.i32,
):
    """Attract the particles of a given species to a position.

    Args:
        particles (ti.template): Particles.
        pos (ti.math.vec2): Attraction position.
        mass (ti.f32): Attraction mass.
        radius (ti.f32): Attraction radius.
        species (ti.i32): Species index.
    """
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        if p.species != species:
            continue
        particles.field[i].vel += attract_particle(p, pos, mass, radius)

@ti.func
def _attract_species(
    particles: ti.template(),
    pos: ti.math.vec2,
    mass: ti.f32,
    radius: ti.f32,
    species: ti.i32,
):
    """Attract the particles of a given species to a position.

    Args:
        particles (ti.template): Particles.
        pos (ti.math.vec2): Attraction position.
        mass (ti.f32): Attraction mass.
        radius (ti.f32): Attraction radius.
        species (ti.i32): Species index.
    """
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        if p.species != species:
            continue
        particles.field[i].vel += attract_particle(p, pos, mass, radius)

@ti.func
def attract_particle(
    p: Particle, pos: ti.math.vec2, mass: ti.f32, radius: ti.f32
) -> ti.math.vec2:
    """Attract a particle to a position.

    Args:
        particles (Particle): Individual particle.
        pos (ti.math.vec2): Attraction position.
        mass (ti.f32): Attraction mass.
        radius (ti.f32): Attraction radius.

    Returns:
        ti.math.vec2: Attraction velocity.
    """
    target_distance = (pos - p.pos).norm()
    vel = ti.Vector([0.0, 0.0])
    if target_distance < radius:
        factor = (radius - target_distance) / radius
        vel = (pos - p.pos).normalized() * mass * factor
    return vel


@ti.kernel
def repel(particles: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
    """Repel the particles from a position.

    Args:
        particles (ti.template): Particles.
        pos (ti.math.vec2): Repulsion position.
        mass (ti.f32): Repulsion mass.
        radius (ti.f32): Repulsion radius.
    """
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        particles.field[i].vel += repel_particle(p, pos, mass, radius)


@ti.kernel
def repel_species(
    particles: ti.template(),
    pos: ti.math.vec2,
    mass: ti.f32,
    radius: ti.f32,
    species: ti.i32,
):
    """Repel the particles of a given species from a position.

    Args:
        particles (ti.template): Particles.
        pos (ti.math.vec2): Repulsion position.
        mass (ti.f32): Repulsion mass.
        radius (ti.f32): Repulsion radius.
        species (ti.i32): Species index.
    """
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        if p.species != species:
            continue
        particles.field[i].vel += repel_particle(p, pos, mass, radius)


@ti.func
def repel_particle(
    p: Particle, pos: ti.math.vec2, mass: ti.f32, radius: ti.f32
) -> ti.math.vec2:
    """Repel a particle from a position.

    Args:
        p (Particle): Individual particle.
        pos (ti.math.vec2): Repulsion position.
        mass (ti.f32): Repulsion mass.
        radius (ti.f32): Repulsion radius.

    Returns:
        ti.math.vec2: Repulsion velocity.
    """
    target_distance = (pos - p.pos).norm()
    vel = ti.Vector([0.0, 0.0])
    if target_distance < radius:
        factor = (target_distance - radius) / radius
        vel = (pos - p.pos).normalized() * mass * factor
    return vel


@ti.kernel
def gravitate(particles: ti.template(), G: ti.f32, radius: ti.f32):
    """Gravitate the particles.

    Args:
        particles (ti.template): Particles.
        G (ti.f32): Gravitational constant.
        radius (ti.f32): Gravitational radius.
    """
    for i, j in ti.ndrange(particles.field.shape[0], particles.field.shape[0]):
        if i == j:
            continue
        p1 = particles.field[i]
        p2 = particles.field[j]
        if (p2.pos - p1.pos).norm() > radius:
            continue
        particles.field[i].vel += gravitation(p1, p2, G)


@ti.kernel
def gravitate_species(
    particles: ti.template(), G: ti.f32, radius: ti.f32, species: ti.i32
):
    """Gravitate the particles of a given species.

    Args:
        particles (ti.template): Particles.
        G (ti.f32): Gravitational constant.
        radius (ti.f32): Gravitational radius.
        species (ti.i32): Species index.
    """
    for i, j in ti.ndrange(particles.field.shape[0], particles.field.shape[0]):
        if i == j:
            continue
        p1 = particles.field[i]
        p2 = particles.field[j]
        if p1.species != species or p2.species != species:
            continue
        if (p2.pos - p1.pos).norm() > radius:
            continue
        particles.field[i].vel += gravitation(p1, p2, G)


@ti.func
def gravitation(p1: Particle, p2: Particle, G: ti.f32) -> ti.math.vec2:
    """Calculate the gravitational force between two particles.

    Args:
        p1 (Particle): Particle 1.
        p2 (Particle): Particle 2.
        G (ti.f32): Gravitational constant.

    Returns:
        ti.math.vec2: Gravitational force.
    """
    r = p2.pos - p1.pos
    distance = r.norm() + 1e-5
    force_direction = r.normalized()
    force_magnitude = G * p1.mass * p2.mass / (distance**2)
    force = force_direction * force_magnitude
    return force / p1.mass


@ti.kernel
def noise(particles: ti.template(), weight: ti.f32):
    """Add noise to the particles.

    Args:
        particles (ti.template): Particles.
        weight (ti.f32): Noise weight.
    """
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        particles.field[i].vel += (ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * weight)
        particles.field[i].pos += p.vel * p.speed * p.active

@ti.kernel
def centripetal(particles: ti.template(), centre: ti.math.vec2, direction: ti.i32, weight: ti.f32):
    """Apply a centripetal force to the particles.

    Args:
        particles (ti.template): Particles.
        centre (ti.math.vec2): Centripetal centre.
        direction (ti.i32): Centripetal direction.
        weight (ti.f32): Centripetal weight.
    """
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        particles.field[i].vel += centripetal_particle(p, centre, direction, weight)

@ti.func
def centripetal_particle(p: ti.template(), centre: ti.math.vec2, direction: ti.i32, weight: ti.f32) -> ti.math.vec2:
    """Apply a centripetal force to a particle.

    Args:
        p (Particle): Individual particle.
        centre (ti.math.vec2): Centripetal centre.
        direction (ti.i32): Centripetal direction.
        weight (ti.f32): Centripetal weight.

    Returns:
        ti.math.vec2: Centripetal velocity.
    """
    r = p.pos - centre
    if direction == 0:
        r = -r
    v_perp = ti.Vector([-r[1], r[0]])
    norm = v_perp.norm() + 1e-5
    v_perp_normalized = v_perp / norm
    speed = p.vel.norm()
    new_vel = v_perp_normalized * speed * weight
    return new_vel
