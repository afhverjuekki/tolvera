import taichi as ti

from .utils import CONSTS


class Species:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.kwargs = kwargs
        self.n = self.tv.sn
        # FIXME: hack
        self.tv.species_consts = CONSTS(
            {
                "MIN_SIZE": (ti.f32, 2.0),
                "MAX_SIZE": (ti.f32, 5.0),
                "MIN_SPEED": (ti.f32, 0.2),
                "MAX_SPEED": (ti.f32, 3.0),
                "MAX_MASS": (ti.f32, 1.0),
            }
        )
        self.tv.s.species = (
            {
                "size": (ti.f32, 0.0, 1.0),
                "speed": (ti.f32, 0.0, 1.0),
                "mass": (ti.f32, 0.0, 1.0),
                # 'decay': (ti.f32, .9, .999),
                "rgba": (ti.math.vec4, 0.0, 1.0),
            },
            self.n,
            "set",
            "set",
        )

    def randomise(self):
        self.tv.s.species.randomise()
