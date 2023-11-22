from .slime import Slime
from .flock import Flock
from .move import Move
from .reaction_diffusion import ReactionDiffusion

class Vera:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.move = Move(tolvera)
        self.flock = Flock(tolvera)
        self.slime = Slime(tolvera)
        self.rd = ReactionDiffusion(tolvera)
    def randomise(self):
        self.flock.randomise()
        self.slime.randomise()
        self.rd.randomise()
