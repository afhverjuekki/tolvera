# External packages
from iipyper.state import _lock
import fire
from sys import exit

# Wrapped external packages
from ._taichi import Taichi
from ._osc import OSC
from ._iml import IMLDict
# from ._cv import CV

# Tölvera components
from .patches import *
from .utils import *
from .particles import *
from .pixels import *
from .vera import Vera
from .state import StateDict

'''
TODO: render
    proper decorator with args, kwargs
    async runner based on sardine @swim?
TODO: global state load/save via utils.ti_serialize, k:v store
'''

class TolveraContext:
    """
    Context for sharing between multiple Tölvera instances.
    Context includes Taichi, OSC, IML and CV.
    All Tölvera instances share the same context and are added to a dict.
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.init(**kwargs)
    def init(self, **kwargs):
        """
        Initialize wrapped external packages with given keyword arguments.
        This only happens once when Tölvera is first initialized.

        Args:
            **kwargs: Keyword arguments for component initialization.
        """
        self.name = 'Tölvera Context'
        self.name_clean = clean_name(self.name)
        print(f"[{self.name}] Initializing context...")
        self.i = 0
        self.x = kwargs.get('x', 1920)
        self.y = kwargs.get('y', 1080)
        self.ti = Taichi(self, **kwargs)
        self.canvas = Pixels(self, **kwargs)
        self.osc = kwargs.get('osc', False)
        self.iml = kwargs.get('iml', False)
        self.cv  = kwargs.get('cv', False)
        if self.osc:
            self.osc = OSC(self, **kwargs)
        if self.iml:
            self.iml = IMLDict(self)
        # if self.cv:
        #     self.cv = CV(self, **kwargs)
        self._cleanup_fns = []
        self.tolveras = {}
        print(f"[{self.name}] Context initialization complete.")
    def run(self, f=None, **kwargs):
        """
        Run Tölvera with given render function and keyword arguments.
        This function will run inside a locked thread until KeyboardInterrupt/exit.
        It runs the render function, updates the OSC map (if enabled), and shows the pixels.

        Args:
            f: Function to run.
            **kwargs: Keyword arguments for function.
        """
        print(f"[{self.name}] Running with render function {f.__name__}...")
        while self.ti.window.running:
            with _lock:
                if f is not None: 
                    self.canvas = f(**kwargs)
                if self.osc is not False: 
                    self.osc.map()
                self.ti.show(self.canvas)
                self.i += 1
    def stop(self):
        """
        Run cleanup functions and exit.
        """
        print(f"\n[{self.name}] Stopping {self.name}...")
        for f in self._cleanup_fns: 
            print(f"\n[{self.name}] Running cleanup function {f.__name__}...")
            f()
        print(f"\n[{self.name}] Exiting {self.name}...")
        exit(0)
    def render(self, f=None, **kwargs):
        """Render Tölvera with given function and keyword arguments.

        Args:
            f (function, optional): Function to run. Defaults to None.
        """
        try: self.run(f, **kwargs)
        except KeyboardInterrupt: self.stop()
    def cleanup(self, f=None):
        """
        Decorator for cleanup functions based on iipyper.
        Make functions run on KeyBoardInterrupt (before exit).
        Cleanup functions must be defined before render is called!

        Args:
            f: Function to cleanup.

        Returns:
            Decorator function if f is None, else decorated function.
        """
        print(f"\n[{self.name}] Adding cleanup function {f.__name__}...")
        def decorator(f):
            self._cleanup_fns.append(f)
            return f
        if f is None: # return a decorator
            return decorator
        else: #bare decorator case; return decorated function
            return decorator(f)
    def add_tolvera(self, tolvera):
        """
        Add Tölvera to context.

        Args:
            tolvera (Tolvera): Tölvera to add.
        """
        assert isinstance(tolvera, Tolvera), f"tolvera must be of type Tolvera, not {type(tolvera)}."
        print(f"[{self.name}] Adding tolvera='{tolvera.name}' to context.")
        self.tolveras[tolvera.name] = tolvera
    def get_tolvera_by_name(self, name):
        """
        Get Tölvera by name.

        Args:
            name (str): Name of Tölvera to get.

        Returns:
            Tölvera: Tölvera with given name.
        """
        return self.tolveras[name]
    def get_tolvera_names(self):
        """
        Get names of all Tölveras in context.

        Returns:
            list: List of Tölvera names.
        """
        return list(self.tolveras.keys())
    def delete_tolvera(self, name):
        """
        Delete Tölvera by name.

        Args:
            name (str): Name of Tölvera to delete.
        """
        print(f"[{self.name}] Deleting tolvera='{name}' from context.")
        del self.tolveras[name]

'''
TODO: test reset
TODO: add attractors
TODO: combined OSC setter(s) for species+flock+slime+rd etc..?
TODO: make tv.p.n and tv.s.n into 0d fields
'''

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
        self.name = kwargs.get('name', 'Tölvera')
        self.name_clean = clean_name(self.name)
        if 'context' not in kwargs:
            self.init_context(**kwargs)
        else:
            self.share_context(kwargs['context'])
        self.setup(**kwargs)
        print(f"[{self.name}] Initialization and setup complete.")
    def init_context(self, **kwargs):
        context = TolveraContext(**kwargs)
        self.share_context(context)
    def share_context(self, context):
        names = context.get_tolvera_names()
        if len(names) == 0:
            print(f"[{self.name}] Sharing context '{context.name}'.")
        else:
            print(f"[{self.name}] Sharing context '{context.name}' with {context.get_tolvera_names()}.")
        self.ctx = context
        self.x       = context.x
        self.y       = context.y
        self.ti      = context.ti
        self.canvas  = context.canvas
        self.osc     = context.osc
        self.iml     = context.iml
        self.render  = context.render
        self.cleanup = context.cleanup
        self.cv      = context.cv
    def setup(self, **kwargs):
        """
        Setup Tölvera with given keyword arguments.
        This can be called multiple throughout the lifetime of Tölvera.

        Args:
            **kwargs: Keyword arguments for setup.
        """
        self.particles = kwargs.get('particles', 1024)
        self.species   = kwargs.get('species', 4)
        self.pn = self.particles
        self.sn = self.species
        self.p_per_s   = self.particles // self.species
        self.substep   = kwargs.get('substep', 1)
        self.evaporate = kwargs.get('evaporate', 0.95)
        self.s = StateDict(self)
        self.px = Pixels(self, **kwargs)
        self._species = Species(self, **kwargs)
        # self.p = Particles(self, self.s, **kwargs)
        self.p = Particles(self, **kwargs)
        self.v = Vera(self, **kwargs)
        # TODO: Useful?
        # self.v = dotdict({
        #     'move':  Move(self),
        #     'flock': Flock(self),
        #     'slime': Slime(self),
        #     'rd':    ReactionDiffusion(self),
        # })
        if self.osc is not False: self.add_to_osc_map()
        self.ctx.add_tolvera(self)
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
    def add_to_osc_map(self):
        """
        Add top-level Tölvera functions to OSC map.
        """
        setter_name = f"{self.name_clean}_set"
        getter_name = f"{self.name_clean}_get"
        self.osc.map.receive_args_inline(setter_name+'_randomise', self.randomise)
        # self.osc.map.receive_args_inline(setter_name+'_reset', self.reset) # TODO: kwargs?
        self.osc.map.receive_args_inline(setter_name+'_particles_randomise', self.p._randomise) # TODO: move inside Particles


def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        tv.p()

if __name__ == '__main__':
    fire.Fire(main)
