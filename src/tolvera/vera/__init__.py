from . import forces
from .flock import Flock
from .reaction_diffusion import ReactionDiffusion
from .slime import Slime


class Vera:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.add_forces_to_self()
        self.flock = Flock(tolvera, **kwargs)
        self.slime = Slime(tolvera, **kwargs)
        self.rd = ReactionDiffusion(tolvera, **kwargs)

    def add_forces_to_self(self):
        for force in forces.__all__:
            setattr(self, force, getattr(forces, force))

    def randomise(self):
        self.flock.randomise()
        self.slime.randomise()
        self.rd.randomise()
