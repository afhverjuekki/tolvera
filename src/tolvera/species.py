import taichi as ti

class Species:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.kwargs = kwargs
        self.n = self.tv.sn
        self.tv.s.species = ({
            'size':  (ti.f32, 1., 4.),
            'speed': (ti.f32, 0., 4.),
            'mass':  (ti.f32, 0., 1.),
            'decay': (ti.f32, .9, .999),
            'rgba':  (ti.math.vec4, 0., 1.),
        }, self.n, 'set', 'set')
    def randomise(self):
        self.tv.s.species.randomise()
