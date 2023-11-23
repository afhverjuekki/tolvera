# Examples

The full collection of examples can be found in [iil-examples](https://github.com/Intelligent-Instruments-Lab/iil-examples/tolvera).
Here are some simple examples:

## Rendering motionless particles

This is the most basic syntax for creating a Tölvera scene. When run, this scene will render a window using default options, and display some motionless particles:

```py
from tolvera import Tolvera

def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        pass
```

## Making particles behave

We can make the particles behave by passing them to a "vera", moving or flocking them based on their properties:

```py
from tolvera import Tolvera

def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        tv.v.move(tv.p) # Moving
        # tv.v.flock(tv.p) # Flocking
```

## Controlling "vera" via OSC input

Tölvera will create an OSC server and client when `osc=True` is passed: 

```py
from tolvera import Tolvera

def main(**kwargs):
    tv = Tolvera(osc=True, **kwargs)

    @tv.render
    def _():
        tv.v.flock(tv.p)
```

<!-- TODO: describe OSC input API -->
`tv.v.flock` can now be controlled via OSC in a number of different ways.

The full OSC options and their defaults are:

```py
{
    'osc': False,
    'host': '127.0.0.1',
    'client': '127.0.0.1',
    'client_name': 'tolvera',
    'receive_port': 5001,
    'send_port': 5000,
    'osc_debug': False,
}
```

## OSC output API

<!-- TODO -->
TBC.

## Adding custom behaviour with Tiachi kernels

We can use Taichi [kernels and functions](https://docs.taichi-lang.org/docs/kernel_function) in Tölvera to add custom behaviours. In this example we add an attractor kernel, and draw it with `tv.px.circle`:

```py
from tolvera import Tolvera
import taichi as ti

def main(**kwargs):
    tv = Tolvera(**kwargs)

    @ti.kernel
    def attract(tv: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
        for i in range(tv.p.n):
            p = tv.p.field[i]
            if p.active == 0: continue
            target_distance = (pos-p.pos).norm()
            if target_distance < radius:
                factor = (radius-target_distance)/radius
                tv.p.field[i].vel += (pos-p.pos).normalized() * mass * factor
        tv.px.circle(pos[0], pos[1], radius, [1.,1.,1.,1.])

    @tv.render
    def _():
        tv.v.move(tv.p)
        attract(tv, [tv.x/2, tv.y/2], 10., 300.)
```

## Live Coding with Sardine

[Sardine](https://sardine.raphaelforment.fr/) is a Python-based live coding environment which can be used to live code Tölvera (currently tested in [Visual Studio Code](https://code.visualstudio.com/) only):

```py
from sardine_core.run import *
from tolvera import Tolvera

def main(**kwargs):
    tv = Tolvera(**kwargs)

    @ti.kernel
    def attract(tv: ti.template(), pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
        for i in range(tv.p.n):
            p = tv.p.field[i]
            if p.active == 0: continue
            target_distance = (pos-p.pos).norm()
            if target_distance < radius:
                factor = (radius-target_distance)/radius
                tv.p.field[i].vel += (pos-p.pos).normalized() * mass * factor
        tv.px.circle(pos[0], pos[1], radius, [1.,1.,1.,1.])
    attract_kernel = attract # workaround for redefining kernels

    # State that can be used inside Taichi scope
    attract_state = tv.State({
        'mass':   (0.,10.), # name: (min, max)
        'radius': (0.,1000.),
    }, tv.o.species) # field shape is int -> (int, int)

    # Sardine @swim function running at control rate
    @swim
    def control(p=4, i=0):
        attract_state.field[0,0].mass = P('1 1.2', i)
        attract_state.field[0,0].radius = P('1900 100', i)
        again(control, p=1/2, i=i+1)

    # Sardine @swim function running at render rate
    @swim
    def render(p=0.5, i=0):
        tv.v.move(tv.p)
        attract_kernel(tv, 
            [tv.x/2, tv.y/2], 
            attract_state.field[0,0].mass, 
            attract_state.field[0,0].radius)
        tv.show(tv.p) # used instead of @tv.render
        again(render, p=1/64, i=i+1)
```

Note that Sardine is not installed by Tölvera, and needs to be installed separately.

## Interactive machine learning

<!-- TODO -->