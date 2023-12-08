import taichi as ti
from .utils import CONSTS

class Species:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.kwargs = kwargs
        self.n = self.tv.sn
        # FIXME: hack
        self.tv.species_consts = CONSTS({
            'MIN_SIZE': (ti.f32, 1.),
            'MAX_SIZE': (ti.f32, 4.),
            'MIN_SPEED': (ti.f32, 0.2),
            'MAX_SPEED': (ti.f32, 3.),
            'MAX_MASS': (ti.f32, 1.),
        })
        self.tv.s.species = ({
            'size':  (ti.f32, 0., 1.),
            'speed': (ti.f32, 0., 1.),
            'mass':  (ti.f32, 0., 1.),
            # 'decay': (ti.f32, .9, .999),
            'rgba':  (ti.math.vec4, 0.25, 1.),
        }, self.n, 'set', 'set')
        # self.tv.s.species = ({
        #     'size':  (ti.f32, 1., 4.),
        #     'speed': (ti.f32, 0., 4.),
        #     'mass':  (ti.f32, 0., 1.),
        #     'decay': (ti.f32, .9, .999),
        #     'rgba':  (ti.math.vec4, 0., 1.),
        # }, self.n, 'set', 'set')
    def randomise(self):
        self.tv.s.species.randomise()
