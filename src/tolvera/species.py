"""Species class."""

import taichi as ti

from .utils import CONSTS


class Species:
    """Species in Tölvera.

    Species are implemented as a State with attributes for `size`, `speed`, `mass` and
    `colour` (rgba), and with a length determined by the number of species in the
    Tölvera instance (`tv.sn`). The attributes are normalised and scaled by species the 
    `species_consts` attribute. They are initialised with random values.

    Rather than accessing this class directly, access is typically via the State
    attributes via the Tölvera instance, via e.g. `tv.s.species.field[i].size`.
    """
    def __init__(self, tolvera, **kwargs) -> None:
        """Initialise Species

        Args:
            tolvera (Tolvera): Tolvera instance.
            **kwargs: Keyword arguments. 
        """
        self.tv = tolvera
        self.kwargs = kwargs
        self.n = self.tv.sn
        self.tv.species_consts = CONSTS(
            {
                "MIN_SIZE": (ti.f32, 2.0),
                "MAX_SIZE": (ti.f32, 5.0),
                "MIN_SPEED": (ti.f32, 0.2),
                "MAX_SPEED": (ti.f32, 2.0),
                "MAX_MASS": (ti.f32, 1.0),
            }
        )
        self.tv.s.species = (
            {
                "size": (ti.f32, 0.0, 1.0),
                "speed": (ti.f32, 0.0, 1.0),
                "mass": (ti.f32, 0.0, 1.0),
                "rgba": (ti.math.vec4, 0.0, 1.0),
            },
            self.n,
            "set",
            "set",
        )

    def randomise(self):
        """Randomise species."""
        self.tv.s.species.randomise()
