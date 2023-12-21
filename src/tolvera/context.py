"""Tölvera context for sharing between multiple Tölvera instances."""

from sys import exit

from iipyper.state import _lock

from ._taichi import Taichi
from .cv import CV
from .iml import IMLDict
from .osc.osc import OSC
from .patches import *
from .pixels import *
from .utils import *


class TolveraContext:
    """
    Context for sharing between multiple Tölvera instances.
    Context includes Taichi, OSC, IML and CV.
    All Tölvera instances share the same context and are added to a dict.
    """

    def __init__(self, **kwargs) -> None:
        """Initialise Tölvera context with given keyword arguments."""
        self.kwargs = kwargs
        self.init(**kwargs)

    def init(self, **kwargs):
        """
        Initialise wrapped external packages with given keyword arguments.
        This only happens once when Tölvera is first initialised.

        Args:
            **kwargs: Keyword arguments for component initialisation.
                x (int): Width of canvas.
                y (int): Height of canvas.
                osc (bool): Enable OSC.
                iml (bool): Enable IML.
                cv (bool): Enable CV.
                see also kwargs for Taichi, OSC, IMLDict, and CV.
        """
        self.name = "Tölvera Context"
        self.name_clean = clean_name(self.name)
        print(f"[{self.name}] Initializing context...")
        self.i = 0
        self.x = kwargs.get("x", 1920)
        self.y = kwargs.get("y", 1080)
        self.ti = Taichi(self, **kwargs)
        self.show = self.ti.show
        self.canvas = Pixels(self, **kwargs)
        self.osc = kwargs.get("osc", False)
        self.iml = kwargs.get("iml", False)
        self.cv = kwargs.get("cv", False)
        if self.osc:
            self.osc = OSC(self, **kwargs)
        if self.iml:
            self.iml = IMLDict(self)
        if self.cv:
            self.cv = CV(self, **kwargs)
        self._cleanup_fns = []
        self.tolveras = {}
        print(f"[{self.name}] Context initialisation complete.")

    def run(self, f=None, **kwargs):
        """
        Run Tölvera with given render function and keyword arguments.
        This function will run inside a locked thread until KeyboardInterrupt/exit.
        It runs the render function, updates the OSC map (if enabled), and shows the pixels.

        Args:
            f: Function to run.
            **kwargs: Keyword arguments for function.
        """
        if f is not None:
            print(f"[{self.name}] Running with render function {f.__name__}...")
        else:
            print(f"[{self.name}] Running with no render function...")
        while self.ti.window.running:
            with _lock:
                if f is not None:
                    self.canvas = f(**kwargs)
                if self.osc is not False:
                    self.osc.map()
                if self.iml is not False:
                    self.iml()
                if self.cv is not False:
                    self.cv()
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
        try:
            self.run(f, **kwargs)
        except KeyboardInterrupt:
            self.stop()

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
            """Decorator that appends function to cleanup functions."""
            self._cleanup_fns.append(f)
            return f

        if f is None:  # return a decorator
            return decorator
        else:  # bare decorator case; return decorated function
            return decorator(f)

    def add(self, tolvera):
        """
        Add Tölvera to context.

        Args:
            tolvera (Tolvera): Tölvera to add.
        """
        print(f"[{self.name}] Adding tolvera='{tolvera.name}' to context.")
        self.tolveras[tolvera.name] = tolvera

    def get_by_name(self, name):
        """
        Get Tölvera by name.

        Args:
            name (str): Name of Tölvera to get.

        Returns:
            Tölvera: Tölvera with given name.
        """
        return self.tolveras[name]

    def get_names(self):
        """
        Get names of all Tölveras in context.

        Returns:
            list: List of Tölvera names.
        """
        return list(self.tolveras.keys())

    def remove(self, name):
        """
        Remove Tölvera by name.

        Args:
            name (str): Name of Tölvera to delete.
        """
        print(f"[{self.name}] Deleting tolvera='{name}' from context.")
        del self.tolveras[name]
