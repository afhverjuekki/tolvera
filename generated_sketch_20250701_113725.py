"""

Dynamically generated Tölvera sketch.
Timestamp: 2025-07-01 11:37:25.875842
"""
import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    tv = Tolvera(**{'particles': 100, 'px': 'pixels', 'gpu': 'metal'})


    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 1.0
            tv.p.field[i].pos = ti.Vector([ti.random() * tv.x, ti.random() * tv.y])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].species = i % 1
            tv.p.field[i].size = 5.0

    init_particles()
    tv.s.species.field[0].rgba = [1.0, 0.3, 0.3, 1.0]
    tv.s.species.field[1].rgba = [0.3, 1.0, 0.3, 1.0]
    tv.s.species.field[2].rgba = [0.3, 0.3, 1.0, 1.0]

    # --- Generated Expert Functions ---
    @ti.func
    def expert_move_to_right(tv: ti.template(), i: ti.i32) -> ti.math.vec2:
        # Force that pushes particle to the right.
        force_strength = 0.05
        return ti.Vector([force_strength, 0.0])

    # --- Generated Integration Kernel ---
    @ti.kernel
    def apply_all_experts(tv: ti.template(), dt: ti.f32):
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                # Initialise a total force for the particle
                total_force = ti.Vector([0.0, 0.0])
    
                # Accumulate forces from all experts
                total_force += expert_move_to_right(tv, i) * 5.00
    
                # Apply the final combined force to the particle
                tv.p.field[i].vel += total_force * dt
                # Apply simple damping to prevent runaway speeds
                tv.p.field[i].vel *= 0.98
                # Update particle position based on new velocity
                tv.p.field[i].pos += tv.p.field[i].vel * dt


    @tv.render
    def render():
        # This calls the dynamically generated kernel
        if callable(apply_all_experts):
             apply_all_experts(tv, 0.016)

        tv.px.clear()
        tv.px.particles(tv.p, tv.s.species)
        return tv.px

    print("Running generated sketch...")
    run(render)

if __name__ == "__main__":
    main()
