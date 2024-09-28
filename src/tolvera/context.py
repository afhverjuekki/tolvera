"""
`TolveraContext` is a shared context or environment for `Tolvera` instances.
It is created automatically when a `Tolvera` instance is created, if one 
does not already exist. It manages the integration of packages for graphics, 
computer vision, communications protocols, and more. If multiple `Tolvera` 
instances are created, they must share the same `TolveraContext`.


Example:
    `TolveraContext` can be created manually, and shared with multiple `Tolvera`
    instances. Note that only one `render` function can be used at a time.
    ```py
    from tolvera import TolveraContext, Tolvera, run

    def main(**kwargs):
        ctx = TolveraContext(**kwargs)

        tv1 = Tolvera(ctx=ctx, **kwargs)
        tv2 = Tolvera(ctx=ctx, **kwargs)

        @tv1.render
        def _():
            return tv2.px

    if __name__ == '__main__':
        run(main)
    ```

Example:
    `TolveraContext` can also be created automatically, and still shared.
    ```py
    from tolvera import Tolvera, run

    def main(**kwargs):
        tv1 = Tolvera(**kwargs)
        tv2 = Tolvera(ctx=tv1.ctx, **kwargs)

        @tv1.render
        def _():
            return tv2.px

    if __name__ == '__main__':
        run(main)
    ```
"""

from sys import exit

from iipyper.state import _lock

from .taichi_ import Taichi
from .cv import CV
from .mp import *
from .iml import IMLDict
from .osc.osc import OSC
from .patches import *
from .pixels import *
from .utils import *
from .state import StateDict

class TolveraContext:
    """
    Context for sharing between multiple Tölvera instances.
    Context includes Taichi, OSC, IML and CV.
    All Tölvera instances share the same context and are managed as a dict.

    Attributes:
        kwargs (dict): Keyword arguments for context.
        name (str): Name of context.
        name_clean (str): 'Cleaned' name of context.
        i (int): Frame counter.
        x (int): Width of canvas.
        y (int): Height of canvas.
        ti (Taichi): Taichi instance.
        canvas (Pixels): Pixels instance.
        osc (OSC): OSC instance.
        iml (IML): IML instance.
        cv (CV): CV instance.
        _cleanup_fns (list): List of cleanup functions.
        tolveras (dict): Dict of Tölvera instances.
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
            x (int): Width of canvas. Default: 1920.
            y (int): Height of canvas. Default: 1080.
            osc (bool): Enable OSC. Default: False.
            iml (bool): Enable IML. Default: False.
            cv (bool): Enable CV. Default: False.
            see also kwargs for Taichi, OSC, IMLDict, and CV.
        """
        self.name = "Tölvera Context"
        self.name_clean = clean_name(self.name)
        print(f"[{self.name}] Initializing context...")
        self.x = kwargs.get("x", 1920)
        self.y = kwargs.get("y", 1080)
        self.ti = Taichi(self, **kwargs)
        self.i = ti.field(ti.i32, ())
        self.show = self.ti.show
        self.canvas = Pixels(self, **kwargs)
        self.s = StateDict(self)
        self.osc = kwargs.get("osc", False)
        self.iml = kwargs.get("iml", False)
        self.cv = kwargs.get("cv", False)
        self.hands = kwargs.get("hands", False)
        self.pose = kwargs.get("pose", False)
        self.face = kwargs.get("face", False)
        self.face_mesh = kwargs.get("face_mesh", False)
        if self.osc:
            self.osc = OSC(self, **kwargs)
        if self.iml:
            self.iml = IMLDict(self)
        if self.cv:
            self.cv = CV(self, **kwargs)
            if self.hands:
                self.hands = MPHands(self, **kwargs)
            if self.pose:
                self.pose = MPPose(self, **kwargs)
            if self.face:
                self.face = MPFace(self, **kwargs)
            if self.face_mesh:
                self.face_mesh = MPFaceMesh(self, **kwargs)
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
            # print(kwargs)
            # exit()
            # gui = kwargs.get('gui', None)
            # if gui is not None:
            #     gui()
            # with self.ti.gui.sub_window("Sub Window", 0.1, 0.1, 0.2, 0.2) as w:
            #     w.text("text")
            with _lock:
                self.step(f, **kwargs)
    
    def step(self, f, **kwargs):
        [t.p() for t in self.tolveras.values()]
        if f is not None:
            self.canvas = f(**kwargs)
        if self.osc is not False:
            self.osc.map()
        if self.iml is not False:
            self.iml()
        if self.cv is not False:
            self.cv()
        self.ti.show(self.canvas)
        self.i[None] += 1

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
