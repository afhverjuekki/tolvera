"""The Vera module provides a wrapper for all available forces and behaviours."""

from . import forces
from .flock import Flock
from .reaction_diffusion import ReactionDiffusion
from .slime import Slime
from .particle_life import ParticleLife
from .swarmalators import Swarmalators
from .gol import GOL

class Vera:
    """The Vera class provides a wrapper for all available forces and behaviours,
    that can be applied to a Tolvera entities such as the Particle system."""
    def __init__(self, tolvera, **kwargs) -> None:
        """Initialise the Vera class.
        
        Args:
            tolvera (Tolvera): A Tolvera instance.
            **kwargs: Keyword arguments passed to the Vera.
        """
        self.tv = tolvera
        self.add_forces_to_self()
        self.flock = Flock(tolvera, **kwargs)
        self.slime = Slime(tolvera, **kwargs)
        self.rd = ReactionDiffusion(tolvera, **kwargs)
        self.plife = ParticleLife(tolvera, **kwargs)
        self.swarm = Swarmalators(tolvera, **kwargs)
        self.gol = GOL(tolvera, **kwargs)

    def add_forces_to_self(self):
        """Add all forces to the Vera instance."""
        for force in forces.__all__:
            setattr(self, force, getattr(forces, force))

    def randomise(self):
        """Randomise all forces and behaviours."""
        self.flock.randomise()
        self.slime.randomise()
        self.rd.randomise()
