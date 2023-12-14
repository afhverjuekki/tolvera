import taichi as ti

from tolvera import Tolvera
from tolvera.state import State

tv = Tolvera(cpu=True)

state = State(
    tv,
    {
        "a": (ti.i32, 0, 10),
        "b": (ti.f32, 0.0, 1.0),
        "c": (ti.math.vec2, (0.0, 0.0), (1.0, 1.0)),
    },
    (2, 1),
)

# state.field
