# Tölvera

[Tölvera](https://tolvera.is) is a library for exploring musical performance with artificial life (ALife) and self-organising systems. The word is an Icelandic [kenning](https://en.wikipedia.org/wiki/Kenning):

- Tölva = computer, from tala (number) + völva (prophetess)
- Vera = being
- Tölvera = number being

Tölvera is written in [Taichi](https://www.taichi-lang.org/), a domain-specific language embedded in Python.

This is experimental software and everything is currently subject to change.

Join us on the Tölvera [Discord](https://discord.gg/ER7tWds9vM).

## Contents
- [Showcase](#showcase)
- [Features](#features)
- [Examples](#examples)
- [Install](#install)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
- [Community](#community)
- [Roadmap](#roadmap)
- [Citing](#citing)
- [Inspiration](#inspiration)
- [Funding](#funding)

## Showcase

TBC Updated YouTube Playlist.

## Features

- `tv.v`: a collection of "vera" (beings) including Move, Flock, Slime and Swarm, with more being continuously added. Vera can be combined and composed in various ways.
- `tv.p`: extensible particle system.
- `tv.s`: multi-species system where each species has a unique relationship with every other species, including itself.
- `tv.px`: drawing library including various shapes and blend modes.
- `tv.osc`: Open Sound Control (OSC) via [iipyper](), including automated export of OSC schemas to JSON, XML, Pure Data (Pd), Max/MSP (SuperCollider TBC).
- `tv.iml`: Interactive Machine Learning via [iml]().
- `tv.ti`: Taichi-based simulation and rendering engine. Can be run "headless" (without graphics).

## Examples

The full collection of examples can be found in [examples](/examples).
Here are some simple examples:

### Rendering motionless particles

This is the most basic syntax for creating a Tölvera scene. When run, this scene will render a window using default options, and display some motionless particles:

```py
from tolvera import Tolvera

def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        pass
```

### Making particles behave

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

### Controlling "vera" via OSC input

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

### OSC output API

<!-- TODO -->
TBC.

### Adding custom behaviour with Tiachi kernels

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

### Live Coding with Sardine

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

### Interactive machine learning

<!-- TODO -->

## Install

Taichi [supports numerous operating systems and backends](https://docs.taichi-lang.org/docs/hello_world#supported-systems-and-backends).
If you plan on using Vulkan for graphics (recommended for macOS), you may need to [install the Vulkan SDK](https://docs.taichi-lang.org/docs/hello_world#supported-systems-and-backends) first and restart your machine.

Tölvera is [registered on PyPI](https://pypi.org/project/tolvera) and can be installed via a Python package manager such as `pip`:

```sh
pip install tolvera
```

You can check if Tölvera installed correctly by running an example:

```sh
python -m tolvera --example=test
```

## Usage

See [examples](/examples) and [docs](/docs).

## Development

To develop Tölvera's internals, it is recommended to clone this repository and install the package in editable mode:

```sh
git clone https://github.com/jarmitage/tolvera # (or clone your own fork)
cd tolvera
pip install -e tolvera
```

## Contributing

We welcome Pull Requests across all areas of the project:
- Addressing [Issues](/issues)
- Adding features (see [Issues](/issues) and [Discussion](/discussion))
- [Examples](/examples)
- [Tests](/tests)
- [Documentation](/docs)

## Community

To discuss Tölvera with developers and other users:
- Use GitHub [Issues](/issues) to report bugs and make specific feature requests.
- Use GitHub [Discussions](/discussions) to share ideas and ask questions.
- Use [Discord](https://discord.gg/ER7tWds9vM) for everything else.

Across the project, we follow the [Algorave Code of Conduct](https://github.com/Algorave/algoraveconduct). Please get in touch if you experience or witness any issues of conduct.

## Roadmap

There is no official roadmap, see [Discussion](/discussion).

## Citing

Tölvera is being written about and used in a number of contexts (see [references.bib](/references.bib)), here are a few recent examples:

```bib
```

## Inspiration

- [SwissGL](https://swiss.gl)
- [Lenia](https://chakazul.github.io/lenia.html)
- Particle Life (attributed to various, see for example [Clusters](https://www.ventrella.com/Clusters/))
- [Journey to the Microcosmos](https://www.youtube.com/@journeytomicro)
- [Complexity Explorables](https://www.complexity-explorables.org/)

## Funding

Tölvera is being developed at the [Intelligent Instruments Lab](https://iil.is) at the [Iceland University of the Arts](https://lhi.is). The Intelligent Instruments project (INTENT) is funded by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 101001848).
