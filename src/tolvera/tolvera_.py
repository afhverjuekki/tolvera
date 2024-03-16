"""
Example:
    This example demonstrates the basic usage of Tölvera.
    It will display a window with a black background.
    ```py
    from tolvera import Tolvera, run

    def main(**kwargs):
        tv = Tolvera(**kwargs)

        @tv.render
        def _():
            return tv.px

    if __name__ == '__main__':
        run(main)
    ```

Example:
    Here's an annotated version of the above example:
    ```py
    # First, we import Tolvera and run() from tolvera.
    from tolvera import Tolvera, run

    # Then, we define a main function which takes in keyword arguments 
    # (kwargs) from the command line.
    def main(**kwargs):
        # Inside the main function, we initialise a Tolvera instance 
        # with the given keyword arguments.
        tv = Tolvera(**kwargs)

        # We use the render() decorator to render the pixels.
        # This function can be named anything. 
        # It will run in a loop until the user exits the program.
        @tv.render
        def _():
            # render() must return Pixels. Often, these pixels will be 
            # the pixels of the Tolvera instance, accessed with tv.px.
            return tv.px

    # Finally, we call run() with the main function as the argument.
    if __name__ == '__main__':
        run(main)
    ```

When Tolvera is run, messages will be printed to the console.
These messages inform the user of the status of Tolvera,
during initialisation, setup, and running.
"""

from fire import Fire as run

from .context import TolveraContext
from .particles import *
from .patches import *
from .pixels import *
from .utils import *
from .vera import Vera

class Tolvera:
    """Tolvera main class.

    Attributes:
        `name` (str): Name of Tölvera instance. 
        `ctx` (TolveraContext): Shared TolveraContext.
        `speed` (float): Global timebase speed.
        `pn` (int): Number of particles.
        `sn` (int): Number of species.
        `p_per_s` (int): Number of particles per species.
        `substep` (int): Number of substeps per frame.
        `iml`: Dict of IML instances via anguilla.
        `cv`: computer vision integration via OpenCV.
        `osc`: OSC via iipyper.
        `ti`: Taichi (graphics backend).
    """

    def __init__(self, **kwargs):
        """
        Initialise and setup Tölvera with given keyword arguments.

        Args:
            name (str): Name of Tölvera instance. Defaults to "Tölvera".
            ctx (TolveraContext): TolveraContext to share. Defaults to None.
            see also kwargs for Tolvera.setup().
        """
        self.kwargs = kwargs
        self.name = kwargs.get("name", "Tölvera")
        self.name_clean = clean_name(self.name)
        if "ctx" not in kwargs:
            self.init_context(**kwargs)
        else:
            self.share_context(kwargs["ctx"])
        self.setup(**kwargs)
        print(f"[{self.name}] Initialisation and setup complete.")

    def init_context(self, **kwargs):
        """Initiliase TölveraContext with given keyword arguments.

        Args:
            **kwargs: Keyword arguments for TölveraContext.
        """
        context = TolveraContext(**kwargs)
        self.share_context(context)

    def share_context(self, context):
        """Share TölveraContext with another Tölvera instance.

        Args:
            context: TölveraContext to share.
        """
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
        self.s = context.s
        self.iml = context.iml
        self.render = context.render
        self.cleanup = context.cleanup
        self.cv = context.cv
        self.hands = context.hands
        self.pose = context.pose
        self.face = context.face
        self.face_mesh = context.face_mesh

    def setup(self, **kwargs):
        """
        Setup Tölvera with given keyword arguments.
        This can be called multiple throughout the lifetime of a Tölvera instance.

        Args:
            **kwargs: Keyword arguments for setup.
                speed (float): Global timebase speed. Defaults to 1.
                particles (int): Number of particles. Defaults to 1024.
                species (int): Number of species. Defaults to 4.
                substep (int): Number of substeps per frame. Defaults to 1.
            See also kwargs for Pixels, Species, Particles, and Vera.
        """
        self._speed = kwargs.get("speed", 1)  # global timebase
        self.particles = kwargs.get("particles", 1024)
        self.species = kwargs.get("species", 4)
        if self.particles < self.species:
            self.species = self.particles
        self.pn = self.particles
        self.sn = self.species
        self.p_per_s = self.particles // self.species
        self.substep = kwargs.get("substep", 1)
        self.px = Pixels(self, **kwargs)
        self._species = Species(self, **kwargs)
        self.p = Particles(self, **kwargs)
        self.speed(self._speed)
        self.v = Vera(self, **kwargs)
        if self.osc is not False:
            self.add_to_osc_map()
        if self.cv is not False:
            if self.hands:
                self.hands.px = self.px
            if self.pose:
                self.pose.px = self.px
            if self.face:
                self.face.px = self.px
            if self.face_mesh:
                self.face_mesh.px = self.px
        self.ctx.add(self)
        print(f"[{self.name}] Setup complete.")

    def randomise(self):
        """
        Randomise particles, species, and Vera.
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
        """Set or get global timebase speed."""
        if speed is not None:
            self._speed = speed
            self.p.speed(speed)
        return self._speed

    def add_to_osc_map(self):
        """
        Add top-level Tölvera functions to OSCMap.
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
            """Set global timebase speed."""
            self.speed(speed)
