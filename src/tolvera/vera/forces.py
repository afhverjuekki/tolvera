import taichi as ti

from ..particles import Particle, Particles

__all__ = [
    "move",
    "attract",
    "attract_species",
    "repel",
    "repel_species",
    "gravitate",
    "gravitate_species",
]


@ti.kernel
def move(particles: ti.template()):
    for i in range(particles.field.shape[0]):
        if particles.field[i].active == 0:
            continue
        p1 = particles.field[i]
        particles.field[i].pos += particles.field[i].vel * p1.speed * p1.active


@ti.kernel
def attract(particles: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        particles.field[i].vel += _attract(p, pos, mass, radius)


@ti.kernel
def attract_species(
    particles: ti.template(),
    pos: ti.math.vec2,
    mass: ti.f32,
    radius: ti.f32,
    species: ti.i32,
):
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        if p.species != species:
            continue
        particles.field[i].vel += _attract(p, pos, mass, radius)


@ti.func
def _attract(
    p: Particle, pos: ti.math.vec2, mass: ti.f32, radius: ti.f32
) -> ti.math.vec2:
    target_distance = (pos - p.pos).norm()
    vel = ti.Vector([0.0, 0.0])
    if target_distance < radius:
        factor = (radius - target_distance) / radius
        vel = (pos - p.pos).normalized() * mass * factor
    return vel


@ti.kernel
def repel(particles: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        particles.field[i].vel += _repel(p, pos, mass, radius)


@ti.kernel
def repel_species(
    particles: ti.template(),
    pos: ti.math.vec2,
    mass: ti.f32,
    radius: ti.f32,
    species: ti.i32,
):
    for i in range(particles.field.shape[0]):
        p = particles.field[i]
        if p.active == 0:
            continue
        if p.species != species:
            continue
        particles.field[i].vel += _repel(p, pos, mass, radius)


@ti.func
def _repel(
    p: Particle, pos: ti.math.vec2, mass: ti.f32, radius: ti.f32
) -> ti.math.vec2:
    target_distance = (pos - p.pos).norm()
    vel = ti.Vector([0.0, 0.0])
    if target_distance < radius:
        factor = (target_distance - radius) / radius
        vel = (pos - p.pos).normalized() * mass * factor
    return vel


@ti.kernel
def gravitate(particles: ti.template(), G: ti.f32, radius: ti.f32):
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
    r = p2.pos - p1.pos
    distance = r.norm() + 1e-5
    force_direction = r.normalized()
    force_magnitude = G * p1.mass * p2.mass / (distance**2)
    force = force_direction * force_magnitude
    return force / p1.mass
