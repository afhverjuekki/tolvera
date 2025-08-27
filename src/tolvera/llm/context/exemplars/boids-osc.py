"""
Boids Flocking Simulation in Tölvera with OSC Integration
Control flocking parameters and send out sonification data via OSC.
V3: Tracks a "rogue" boid (furthest from center) for direct visual/sonic mapping.
V4: Added smoothing for OSC data and visual highlighting of rogue boids.
"""

import taichi as ti
from tolvera import Tolvera, run
import numpy as np

def main(**kwargs):
    """Main function for Boids flocking simulation with OSC."""
    # === Configuration ===
    kwargs.setdefault('species', 3)
    kwargs.setdefault('particles', 800)
    kwargs.setdefault('width', 1920)
    kwargs.setdefault('height', 1080)
    
    learn_mode = kwargs.get('learn_mode', 0)
    
    tv = Tolvera(**kwargs)
    
    # === Particle Initialization ===
    @ti.kernel
    def init_particles_flocks():
        particles_per_species = tv.pn // tv.sn
        remaining = tv.pn % tv.sn
        particle_idx = 0
        for species_id in range(tv.sn):
            count = particles_per_species + (1 if species_id < remaining else 0)
            cluster_center = ti.Vector([
                200.0 + (tv.x - 400.0) * ti.random(),
                200.0 + (tv.y - 400.0) * ti.random()
            ])
            for i in range(count):
                if particle_idx < tv.pn:
                    angle = ti.random() * 2.0 * 3.14159
                    radius = ti.random() * 100.0
                    tv.p.field[particle_idx].active = 1.0
                    tv.p.field[particle_idx].pos = cluster_center + ti.Vector([
                        ti.cos(angle) * radius,
                        ti.sin(angle) * radius
                    ])
                    vel_angle = ti.random() * 2.0 * 3.14159
                    vel_mag = 50.0 + ti.random() * 50.0
                    tv.p.field[particle_idx].vel = ti.Vector([
                        ti.cos(vel_angle) * vel_mag,
                        ti.sin(vel_angle) * vel_mag
                    ])
                    tv.p.field[particle_idx].size = 4.0
                    tv.p.field[particle_idx].mass = 1.0
                    tv.p.field[particle_idx].speed = 1.0
                    tv.p.field[particle_idx].species = species_id
                    particle_idx += 1
    
    init_particles_flocks()
    
    colors = [[0.9, 0.7, 0.2, 1.0], [0.8, 0.2, 0.8, 1.0], [0.2, 0.7, 0.9, 1.0]]
    for i in range(min(tv.sn, len(colors))):
        tv.s.species.field[i].rgba = colors[i]
    
    # === State Initialization ===
    # Smoothing factor - adjust between 0.0 (no smoothing) and 1.0 (heavy smoothing)
    SMOOTHING_FACTOR = 0.0  # Higher = smoother but more lag
    
    tv.s.set('llm_global', {
        'state': {
            'perception_radius': (ti.f32, 20.0, 200.0),
            'separation_radius': (ti.f32, 10.0, 50.0),
            'max_speed': (ti.f32, 50.0, 300.0),
            'max_force': (ti.f32, 100.0, 500.0),
        }, 'shape': 1, 'osc': ('get', 'set')
    })
    tv.s.set('llm_species', {
        'state': {
            'separation_weight': (ti.f32, 0.5, 3.0),
            'alignment_weight': (ti.f32, 0.5, 2.0),
            'cohesion_weight': (ti.f32, 0.5, 2.0),
            'inter_species_avoidance': (ti.f32, 0.0, 5.0),
            # Rogue boid metrics storage
            'rogue_pos': (ti.math.vec2, 0.0, max(tv.x, tv.y)),
            'rogue_vel_mag': (ti.f32, 0.0, 500.0),
            'particle_count': (ti.i32, 0, tv.pn),
            # Smoothed metrics for OSC output
            'smooth_x': (ti.f32, 0.0, 1.0),
            'smooth_y': (ti.f32, 0.0, 1.0), 
            'smooth_vel': (ti.f32, 0.0, 1.0),
            'prev_x': (ti.f32, 0.0, 1.0),
            'prev_y': (ti.f32, 0.0, 1.0),
            'prev_vel': (ti.f32, 0.0, 1.0),
        }, 'shape': tv.sn, 'osc': ('get', 'set')
    })
    tv.s.llm_global.field[0].perception_radius = 80.0
    tv.s.llm_global.field[0].separation_radius = 30.0
    tv.s.llm_global.field[0].max_speed = 200.0
    tv.s.llm_global.field[0].max_force = 300.0
    
    @ti.kernel
    def init_species_params():
        for s in range(tv.sn):
            tv.s.llm_species.field[s].separation_weight = 1.5 + ti.random() * 0.5
            tv.s.llm_species.field[s].alignment_weight = 1.0 + ti.random() * 0.5
            tv.s.llm_species.field[s].cohesion_weight = 1.0 + ti.random() * 0.5
            tv.s.llm_species.field[s].inter_species_avoidance = 2.0 + ti.random() * 1.0
    init_species_params()

    # === Force Experts and Integration Kernels ===
    @ti.func
    def separation_force(p,i):
        force,count=ti.math.vec2(0.0),0
        for j in range(tv.pn):
            if i!=j and tv.p.field[j].active>0:
                diff=p.pos-tv.p.field[j].pos;dist=diff.norm()
                if dist>0 and dist<tv.s.llm_global.field[0].separation_radius: force+=diff/dist/dist;count+=1
        if count>0:
            force/=count;force_norm=force.norm()
            if force_norm>0: force=(force/force_norm)*tv.s.llm_global.field[0].max_speed;force-=p.vel;force*=tv.s.llm_species.field[p.species].separation_weight
        return force
    @ti.func
    def alignment_force(p,i):
        avg_vel,count,steer=ti.math.vec2(0.0),0,ti.math.vec2(0.0)
        for j in range(tv.pn):
            if i!=j and tv.p.field[j].active>0 and tv.p.field[j].species==p.species:
                dist=(tv.p.field[j].pos-p.pos).norm()
                if dist>0 and dist<tv.s.llm_global.field[0].perception_radius: avg_vel+=tv.p.field[j].vel;count+=1
        if count>0:
            avg_vel/=count;avg_vel_norm=avg_vel.norm()
            if avg_vel_norm>0: avg_vel=(avg_vel/avg_vel_norm)*tv.s.llm_global.field[0].max_speed;steer=avg_vel-p.vel;steer*=tv.s.llm_species.field[p.species].alignment_weight
        return steer
    @ti.func
    def cohesion_force(p,i):
        center,count,steer=ti.math.vec2(0.0),0,ti.math.vec2(0.0)
        for j in range(tv.pn):
            if i!=j and tv.p.field[j].active>0 and tv.p.field[j].species==p.species:
                dist=(tv.p.field[j].pos-p.pos).norm()
                if dist>0 and dist<tv.s.llm_global.field[0].perception_radius: center+=tv.p.field[j].pos;count+=1
        if count>0:
            center/=count;desired=center-p.pos;desired_norm=desired.norm()
            if desired_norm>0: desired=(desired/desired_norm)*tv.s.llm_global.field[0].max_speed;steer=desired-p.vel;steer*=tv.s.llm_species.field[p.species].cohesion_weight
        return steer
    @ti.func
    def inter_species_avoidance(p,i):
        force,count,result=ti.math.vec2(0.0),0,ti.math.vec2(0.0)
        avoidance_radius=tv.s.llm_global.field[0].perception_radius*0.7
        for j in range(tv.pn):
            if i!=j and tv.p.field[j].active>0 and tv.p.field[j].species!=p.species:
                diff=p.pos-tv.p.field[j].pos;dist=diff.norm()
                if dist>0 and dist<avoidance_radius: force+=diff/dist/dist;count+=1
        if count>0: force/=count;result=force*tv.s.llm_species.field[p.species].inter_species_avoidance*100.0
        return result
    @ti.func
    def wander_force(p,i):
        angle=(ti.random()-0.5+i*0.001)*0.5
        return ti.math.vec2(ti.cos(angle),ti.sin(angle))*20.0
    @ti.kernel
    def apply_all_experts():
        dt=0.032;max_force=tv.s.llm_global.field[0].max_force;max_speed=tv.s.llm_global.field[0].max_speed
        for i in range(tv.pn):
            if tv.p.field[i].active>0:
                p=tv.p.field[i];total_force=ti.math.vec2(0.0)
                total_force+=separation_force(p,i);total_force+=alignment_force(p,i)
                total_force+=cohesion_force(p,i);total_force+=inter_species_avoidance(p,i)
                total_force+=wander_force(p,i)
                force_norm=total_force.norm()
                if force_norm>max_force:total_force=(total_force/force_norm)*max_force
                acceleration=total_force/p.mass if p.mass>0 else total_force
                new_vel=p.vel+acceleration*dt;vel_norm=new_vel.norm()
                if vel_norm>max_speed:new_vel=(new_vel/vel_norm)*max_speed
                tv.p.field[i].vel=new_vel;tv.p.field[i].pos+=new_vel*p.speed*dt

    # === Intermediate Fields for Metric Calculation ===
    _avg_pos = ti.Vector.field(2, dtype=ti.f32, shape=tv.sn)
    _p_count = ti.field(ti.i32, shape=tv.sn)
    _rogue_idx = ti.field(ti.i32, shape=tv.sn)
    _max_dist_sq = ti.field(ti.f32, shape=tv.sn)

    # === Kernel to find and store Rogue Boid data ===
    @ti.kernel
    def calculate_species_metrics():
        # --- PASS 1: Calculate Average Position ---
        for s in range(tv.sn):
            _avg_pos[s] = ti.math.vec2(0.0)
            _p_count[s] = 0
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                s = tv.p.field[i].species
                _avg_pos[s] += tv.p.field[i].pos
                _p_count[s] += 1
        for s in range(tv.sn):
            if _p_count[s] > 0:
                _avg_pos[s] /= _p_count[s]
            tv.s.llm_species.field[s].particle_count = _p_count[s]

        # --- PASS 2: Find the Rogue Boid (furthest from average) ---
        for s in range(tv.sn):
            _rogue_idx[s] = -1
            _max_dist_sq[s] = -1.0
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                s = tv.p.field[i].species
                dist_sq = (tv.p.field[i].pos - _avg_pos[s]).norm_sqr()
                if dist_sq > _max_dist_sq[s]:
                    _max_dist_sq[s] = dist_sq
                    _rogue_idx[s] = i
        
        # --- FINAL: Store the Rogue Boid's data ---
        for s in range(tv.sn):
            if _rogue_idx[s] != -1:
                rogue_boid = tv.p.field[_rogue_idx[s]]
                tv.s.llm_species.field[s].rogue_pos = rogue_boid.pos
                tv.s.llm_species.field[s].rogue_vel_mag = rogue_boid.vel.norm()
            else: # If no boids of this species exist
                tv.s.llm_species.field[s].rogue_pos = ti.math.vec2(0.0)
                tv.s.llm_species.field[s].rogue_vel_mag = 0.0

    # === Smoothing kernel for OSC data ===
    @ti.kernel
    def smooth_species_metrics():
        for s in range(tv.sn):
            # Get current raw values (normalized)
            current_x = tv.s.llm_species.field[s].rogue_pos.x / tv.x
            current_y = tv.s.llm_species.field[s].rogue_pos.y / tv.y
            current_vel = tv.s.llm_species.field[s].rogue_vel_mag / tv.s.llm_global.field[0].max_speed
            
            # Apply exponential smoothing
            tv.s.llm_species.field[s].smooth_x = (
                SMOOTHING_FACTOR * tv.s.llm_species.field[s].prev_x + 
                (1.0 - SMOOTHING_FACTOR) * current_x
            )
            tv.s.llm_species.field[s].smooth_y = (
                SMOOTHING_FACTOR * tv.s.llm_species.field[s].prev_y + 
                (1.0 - SMOOTHING_FACTOR) * current_y
            )
            tv.s.llm_species.field[s].smooth_vel = (
                SMOOTHING_FACTOR * tv.s.llm_species.field[s].prev_vel + 
                (1.0 - SMOOTHING_FACTOR) * current_vel
            )
            
            # Store current values as previous for next frame
            tv.s.llm_species.field[s].prev_x = tv.s.llm_species.field[s].smooth_x
            tv.s.llm_species.field[s].prev_y = tv.s.llm_species.field[s].smooth_y
            tv.s.llm_species.field[s].prev_vel = tv.s.llm_species.field[s].smooth_vel

    # === OSC Receivers ===
    @tv.osc.map.receive_args(
        perception=(80.0,20.0,200.0),separation=(30.0,10.0,50.0),
        max_speed=(200.0,50.0,300.0),max_force=(300.0,100.0,500.0),count=1)
    def global_params(p:float,s:float,ms:float,mf:float):
        g=tv.s.llm_global.field[0];g.perception_radius=p;g.separation_radius=s;g.max_speed=ms;g.max_force=mf
    @tv.osc.map.receive_args(
        species_id=(0,0,tv.sn-1),sep_w=(1.5,0.5,3.0),align_w=(1.0,0.5,2.0),
        cohere_w=(1.0,0.5,2.0),avoid_w=(2.0,0.0,5.0),count=1)
    def species_params(id:int,sw:float,aw:float,cw:float,avw:float):
        if 0<=id<tv.sn:
            s=tv.s.llm_species.field[id];s.separation_weight=sw;s.alignment_weight=aw
            s.cohesion_weight=cw;s.inter_species_avoidance=avw

    # === Smoothed OSC Senders for Rogue Boid Data ===
    if tv.sn > 0:
        if learn_mode == 1 or learn_mode == 0:
            @tv.osc.map.send_args(val=(0.5,0,1), send_mode='broadcast', name='metrics/0/x', count=2)
            def send_s0_x() -> list[float]:
                return [np.clip(tv.s.llm_species.field[0].smooth_x, 0, 1)]
        if learn_mode == 2 or learn_mode == 0:
            @tv.osc.map.send_args(val=(0.5,0,1), send_mode='broadcast', name='metrics/0/y', count=2)
            def send_s0_y() -> list[float]:
                return [np.clip(tv.s.llm_species.field[0].smooth_y, 0, 1)]
        if learn_mode == 3 or learn_mode == 0:
            @tv.osc.map.send_args(val=(0.5,0,1), send_mode='broadcast', name='metrics/0/vel', count=2)
            def send_s0_vel() -> list[float]:
                return [np.clip(tv.s.llm_species.field[0].smooth_vel, 0, 1)]
    if tv.sn > 1:
        if learn_mode == 4 or learn_mode == 0:
            @tv.osc.map.send_args(val=(0.5,0,1), send_mode='broadcast', name='metrics/1/x', count=2)
            def send_s1_x() -> list[float]:
                return [np.clip(tv.s.llm_species.field[1].smooth_x, 0, 1)]
        if learn_mode == 5 or learn_mode == 0:
            @tv.osc.map.send_args(val=(0.5,0,1), send_mode='broadcast', name='metrics/1/y', count=2)
            def send_s1_y() -> list[float]:
                return [np.clip(tv.s.llm_species.field[1].smooth_y, 0, 1)]
        if learn_mode == 6 or learn_mode == 0:
            @tv.osc.map.send_args(val=(0.5,0,1), send_mode='broadcast', name='metrics/1/vel', count=2)
            def send_s1_vel() -> list[float]:
                return [np.clip(tv.s.llm_species.field[1].smooth_vel, 0, 1)]
    if tv.sn > 2:
        if learn_mode == 7 or learn_mode == 0:
            @tv.osc.map.send_args(val=(0.5,0,1), send_mode='broadcast', name='metrics/2/x', count=2)
            def send_s2_x() -> list[float]:
                return [np.clip(tv.s.llm_species.field[2].smooth_x, 0, 1)]
        if learn_mode == 8 or learn_mode == 0:
            @tv.osc.map.send_args(val=(0.5,0,1), send_mode='broadcast', name='metrics/2/y', count=2)
            def send_s2_y() -> list[float]:
                return [np.clip(tv.s.llm_species.field[2].smooth_y, 0, 1)]
        if learn_mode == 9 or learn_mode == 0:
            @tv.osc.map.send_args(val=(0.5,0,1), send_mode='broadcast', name='metrics/2/vel', count=2)
            def send_s2_vel() -> list[float]:
                return [np.clip(tv.s.llm_species.field[2].smooth_vel, 0, 1)]

    @ti.kernel
    def draw_visuals():
        # Highlight rogue boids with bright rings
        for s in range(tv.sn):
            if _rogue_idx[s] != -1:  # If there's a valid rogue boid for this species
                rogue_boid = tv.p.field[_rogue_idx[s]]
                if rogue_boid.active > 0:
                    # Get rogue boid position
                    x = ti.cast(rogue_boid.pos.x, ti.i32)
                    y = ti.cast(rogue_boid.pos.y, ti.i32)
                    
                    # Draw a bright white ring around the rogue boid
                    ring_color = ti.Vector([1.0, 1.0, 1.0, 1.0])  # Bright white
                    inner_radius = ti.cast(rogue_boid.size + 3, ti.i32)
                    outer_radius = ti.cast(rogue_boid.size + 6, ti.i32)
                    
                    # Draw outer circle (filled)
                    tv.px.circle(x, y, outer_radius, ring_color, 1)
                    # Draw inner circle (filled with background/black to create ring effect)
                    tv.px.circle(x, y, inner_radius, ti.Vector([0.0, 0.0, 0.0, 1.0]), 1)
                    
                    # Add a small cross marker at the center
                    cross_size = 2
                    tv.px.line(x - cross_size, y, x + cross_size, y, ring_color)
                    tv.px.line(x, y - cross_size, x, y + cross_size, ring_color)
    
    @tv.render
    def _():
        tv.px.diffuse(0.98)
        calculate_species_metrics()
        smooth_species_metrics()
        draw_visuals()
        apply_all_experts()
        tv.p()
        tv.px.particles(tv.p, tv.s.species())
        return tv.px

if __name__ == "__main__":
    run(main)