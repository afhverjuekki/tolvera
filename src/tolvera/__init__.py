import fire

from .context import TolveraContext
from .particles import *
from .patches import *
from .pixels import *
from .state import StateDict
from .utils import *
from .vera import Vera


class Tolvera:
    """
    Tölvera class which contains all Tölvera components;
    Particles, Species, Vera, and Pixels.
    Multiple Tölvera instances share the same TölveraContext.
    """

    def __init__(self, **kwargs):
        """
        Initialize and setup Tölvera with given keyword arguments.

        Args:
            **kwargs: Keyword arguments for setup and initialization.
        """
        self.kwargs = kwargs
        self.name = kwargs.get("name", "Tölvera")
        self.name_clean = clean_name(self.name)
        if "ctx" not in kwargs:
            self.init_context(**kwargs)
        else:
            self.share_context(kwargs["ctx"])
        self.setup(**kwargs)
        print(f"[{self.name}] Initialization and setup complete.")

    def init_context(self, **kwargs):
        context = TolveraContext(**kwargs)
        self.share_context(context)

    def share_context(self, context):
        if len(context.get_names()) == 0:
            print(f"[{self.name}] Sharing context '{context.name}'.")
        else:
            print(
                f"[{self.name}] Sharing context '{context.name}' with {context.get_names()}."
            )
        self.ctx = context
        self.x = context.x
        self.y = context.y
        self.ti = context.ti
        self.show = context.show
        self.canvas = context.canvas
        self.osc = context.osc
        self.iml = context.iml
        self.render = context.render
        self.cleanup = context.cleanup
        self.cv = context.cv

    def setup(self, **kwargs):
        """
        Setup Tölvera with given keyword arguments.
        This can be called multiple throughout the lifetime of Tölvera.

        Args:
            **kwargs: Keyword arguments for setup.
        """
        self._speed = kwargs.get("speed", 1)  # global timebase
        self.particles = kwargs.get("particles", 1024)
        self.species = kwargs.get("species", 4)
        self.pn = self.particles
        self.sn = self.species
        self.p_per_s = self.particles // self.species
        self.substep = kwargs.get("substep", 1)
        self.evaporate = kwargs.get("evaporate", 0.95)
        self.s = StateDict(self)
        self.px = Pixels(self, **kwargs)
        self._species = Species(self, **kwargs)
        self.p = Particles(self, **kwargs)
        self.v = Vera(self, **kwargs)
        if self.osc is not False:
            self.add_to_osc_map()
        self.ctx.add(self)
        print(f"[{self.name}] Setup complete.")

    def randomise(self):
        """
        Randomize particles, species, and Vera.
        """
        self.p.randomise()
        self.s.species.randomise()
        self.v.randomise()

    def reset(self, **kwargs):
        """
        Reset Tölvera with given keyword arguments.
        This will call setup() with given keyword arguments, but not init().

        Args:
            **kwargs: Keyword arguments for reset.
        """
        print(f"[{self.name}] Resetting self with kwargs={kwargs}...")
        if kwargs is not None:
            self.kwargs = kwargs
        self.setup()

    def speed(self, speed: float = None):
        if speed is not None:
            self._speed = speed
            self.p.speed(speed)
        return self._speed

    def add_to_osc_map(self):
        """
        Add top-level Tölvera functions to OSC map.
        """
        setter_name = f"{self.name_clean}_set"
        getter_name = f"{self.name_clean}_get"
        self.osc.map.receive_args_inline(setter_name + "_randomise", self.randomise)
        # self.osc.map.receive_args_inline(setter_name+'_reset', self.reset) # TODO: kwargs?
        self.osc.map.receive_args_inline(
            setter_name + "_particles_randomise", self.p._randomise
        )  # TODO: move inside Particles

        @self.osc.map.receive_args(speed=(1, 0, 100), count=1)
        def tolvera_set_speed(speed: float):
            self.speed(speed)


def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        tv.p()


if __name__ == "__main__":
    fire.Fire(main)
