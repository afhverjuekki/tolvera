"""Taichi class for initialising Taichi and UI."""

import time
from typing import Any

import taichi as ti


class Taichi:
    """Taichi class for initialising Taichi and UI.
    
    This class provides a show method for showing the Taichi canvas.
    It is used by the TolveraContext class to display a window."""
    def __init__(self, context, **kwargs) -> None:
        """Initialise Taichi
        
        Args:
            context (TolveraContext): global TolveraContext instance.
            **kwargs: Keyword arguments:
                gpu (str): GPU architecture to run on. Defaults to "vulkan".
                cpu (bool): Run on CPU. Defaults to False.
                fps (int): FPS limit. Defaults to 120.
                seed (int): Random seed. Defaults to time.time().
                headless (bool): Run headless. Defaults to False.
                name (str): Window name. Defaults to "Tölvera".
        """
        self.ctx = context
        self.kwargs = kwargs
        self.gpu = kwargs.get("gpu", "vulkan")
        self.cpu = kwargs.get("cpu", None)
        self.fps = kwargs.get("fps", 120)
        self.seed = kwargs.get("seed", int(time.time()))
        self.headless = kwargs.get("headless", False)
        self.name = kwargs.get("name", "Tölvera")
        self.init_ti()
        self.init_ui()
        print(f"[Tölvera.Taichi] Taichi initialised with: {vars(self)}")

    def init_ti(self):
        """Initialise Taichi backend on selected architecture."""
        if self.cpu:
            ti.init(arch=ti.cpu, random_seed=self.seed)
            self.gpu = None
            print("[Tölvera.Taichi] Running on CPU")
        else:
            if self.gpu == "vulkan":
                ti.init(arch=ti.vulkan, random_seed=self.seed)
            elif self.gpu == "metal":
                ti.init(arch=ti.metal, random_seed=self.seed)
            elif self.gpu == "cuda":
                ti.init(arch=ti.cuda, random_seed=self.seed)
            else:
                print(f"[Tölvera.Taichi] Invalid GPU: {self.gpu}")
                return False
            print(f"[Tölvera.Taichi] Running on {self.gpu}")

    def init_ui(self):
        """Initialise Taichi UI window and canvas."""
        self.window = ti.ui.Window(
            self.name,
            (self.ctx.x, self.ctx.y),
            fps_limit=self.fps,
            show_window=not self.headless,
        )
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        # if self.3D:
        #   self.scene = self.window.scene() # 3D

    def show(self, px):
        """Show Taichi canvas and show window."""
        self.canvas.set_image(px.px.rgba)
        if not self.headless:
            self.window.show()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call Taichi window show."""
        self.show(*args, **kwds)
