"""Tölvera command line interface."""

import fire

from tolvera import Tolvera

from .sketchbook import main as sketchbook


def help():
    """Print help message."""
    print(
        """
    available subcommands:
        demo: run a simple default Tolvera demo
        sketchbook: set a sketchbook folder
        sketches: list available sketches
        sketch: run a particular sketch (can be a file path or index)
        random: run a random sketch from the sketchbook
        help: print this help message
    """
    )


def demo(**kwargs):
    """Run a simple demo.
    
    Args:
        **kwargs: Keyword arguments for Tölvera.
    """
    print("Running demo...")
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        """Render function that draws flocking particles."""
        tv.px.diffuse(0.99)
        tv.v.flock(tv.p)
        tv.px.particles(tv.p, tv.s.species, "circle")
        return tv.px


def main(**kwargs):
    """Run Tölvera with kwargs.
    
    Args:
        **kwargs: Keyword arguments for Tolvera (see help()).
    """
    if "sketch" in kwargs or "sketches" in kwargs or "sketchbook" in kwargs:
        sketchbook(**kwargs)
    elif "demo" in kwargs:
        demo(**kwargs)
    elif "help" in kwargs:
        help()
        exit()
    else:
        help()
        demo(**kwargs)


if __name__ == "__main__":
    """Run Tölvera."""
    fire.Fire(main)
