import taichi as ti
from tolvera import Tolvera, run
from tolvera.pixels import Pixels

@ti.data_oriented
class PerlinNoise:
    def __init__(self, res):
        self.res = res
        self.gradient = ti.Vector.field(2, dtype=ti.f32, shape=(res, res))
        self._initialize_gradient()

    @ti.kernel
    def _initialize_gradient(self):
        for i, j in ti.ndrange(self.res, self.res):
            angle = ti.random() * 2 * ti.math.pi
            self.gradient[i, j] = ti.Vector([ti.cos(angle), ti.sin(angle)])

    @staticmethod
    @ti.func
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    @ti.func
    def lerp(t, a, b):
        return a + t * (b - a)

    @staticmethod
    @ti.func
    def grad(hash, x, y):
        h = hash & 3
        u = x if h < 2 else y
        v = y if h < 2 else x
        return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)

    @ti.func
    def noise(self, pos: ti.template()) -> ti.f32:
        xi = ti.cast(ti.floor(pos[0]), ti.int32)
        yi = ti.cast(ti.floor(pos[1]), ti.int32)

        xf = pos[0] - xi
        yf = pos[1] - yi

        u = self.fade(xf)
        v = self.fade(yf)

        n00 = self.grad(ti.cast(self.gradient[xi % self.res, yi % self.res].dot(ti.Vector([1.0, 1.0])), ti.int32), xf, yf)
        n01 = self.grad(ti.cast(self.gradient[xi % self.res, (yi + 1) % self.res].dot(ti.Vector([1.0, 1.0])), ti.int32), xf, yf - 1)
        n10 = self.grad(ti.cast(self.gradient[(xi + 1) % self.res, yi % self.res].dot(ti.Vector([1.0, 1.0])), ti.int32), xf - 1, yf)
        n11 = self.grad(ti.cast(self.gradient[(xi + 1) % self.res, (yi + 1) % self.res].dot(ti.Vector([1.0, 1.0])), ti.int32), xf - 1, yf - 1)

        x1 = self.lerp(u, n00, n10)
        x2 = self.lerp(u, n01, n11)

        return self.lerp(v, x1, x2)

    @ti.kernel
    def generate(self, out: ti.template(), t: ti.f32, scale: ti.f32):
        for i, j in out:
            pos = ti.Vector([i / self.res, j / self.res]) * scale + ti.Vector([0.0, t])
            n = (self.noise(pos) + 1) * 0.5
            out[i, j] = ti.Vector([n, n, n, 1.])

def main(**kwargs):
    tv = Tolvera(**kwargs)
    perlin = PerlinNoise(tv.x)
    noise_img = Pixels(tv)
    perlin.generate(noise_img.px.rgba, 0.0, 10.0)

    @ti.kernel
    def draw():
        tv.px.stamp(0, 0, noise_img)

    draw()

    @tv.render
    def _():
        return tv.px

if __name__ == '__main__':
    run(main)
