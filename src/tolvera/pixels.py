import taichi as ti
from taichi.lang.matrix import MatrixField
from taichi.lang.struct import StructField

from .utils import CONSTS

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4


@ti.dataclass
class Pixel:
    g: vec1
    rgb: vec3
    rgba: vec4
    rgba_inv: vec4


@ti.dataclass
class Point:
    x: ti.i32
    y: ti.i32
    rgba: vec4


@ti.dataclass
class Line:
    x0: ti.i32
    y0: ti.i32
    x1: ti.i32
    y1: ti.i32
    rgba: vec4


@ti.dataclass
class Rect:
    x: ti.i32
    y: ti.i32
    width: ti.i32
    height: ti.i32
    rgba: vec4


@ti.dataclass
class Circle:
    x: ti.i32
    y: ti.i32
    radius: ti.i32
    rgba: vec4


@ti.dataclass
class Triangle:
    x0: ti.i32
    y0: ti.i32
    x1: ti.i32
    y1: ti.i32
    x2: ti.i32
    y2: ti.i32
    rgba: vec4


# TODO: ???
@ti.dataclass
class Polygon:
    p: Point


@ti.data_oriented
class Pixels:
    def __init__(self, tolvera, **kwargs):
        self.tv = tolvera
        self.kwargs = kwargs
        self.polygon_mode = kwargs.get("polygon_mode", "crossing")
        self.x = self.tv.x
        self.y = self.tv.y
        self.px = Pixel.field(shape=(self.x, self.y))
        brightness = kwargs.get("brightness", 1.0)
        self.CONSTS = CONSTS(
            {
                "BRIGHTNESS": (ti.f32, brightness),
            }
        )
        self.shape_enum = {
            "point": 0,
            "line": 1,
            "rect": 2,
            "circle": 3,
            "triangle": 4,
            "polygon": 5,
        }

    def set(self, px):
        self.px.rgba = self.rgba_from_px(px)

    def get(self):
        return self.px

    @ti.kernel
    def clear(self):
        self.px.rgba.fill(0)

    @ti.kernel
    def diffuse(self, evaporate: ti.f32):
        for i, j in ti.ndrange(self.x, self.y):
            d = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    dx = (i + di) % self.x
                    dy = (j + dj) % self.y
                    d += self.px.rgba[dx, dy]
            d *= evaporate / 9.0
            self.px.rgba[i, j] = d

    @ti.func
    def background(self, r, g, b):
        bg = ti.Vector([r, g, b, 1.0])
        self.rect(0, 0, self.x, self.y, bg)

    @ti.func
    def point(self, x: ti.i32, y: ti.i32, rgba: vec4):
        self.px.rgba[x, y] = rgba

    @ti.func
    def points(self, x: ti.template(), y: ti.template(), rgba: vec4):
        for i in ti.static(range(len(x))):
            self.point(x[i], y[i], rgba)

    @ti.func
    def rect(self, x: ti.i32, y: ti.i32, w: ti.i32, h: ti.i32, rgba: vec4):
        # TODO: fill arg
        # TODO: gradients, lerp with ti.math.mix(x, y, a)
        for i, j in ti.ndrange(w, h):
            self.px.rgba[x + i, y + j] = rgba

    @ti.func
    def line(self, x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, rgba: vec4):
        """
        Bresenham's line algorithm
        TODO: thickness
        TODO: anti-aliasing
        TODO: should lines wrap around (as two lines)?
        """
        dx = ti.abs(x1 - x0)
        dy = ti.abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                self.px.rgba[x, y] = rgba
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                self.px.rgba[x, y] = rgba
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        self.px.rgba[x, y] = rgba

    @ti.func
    def circle(self, x: ti.i32, y: ti.i32, r: ti.i32, rgba: vec4):
        for i in range(r + 1):
            d = ti.sqrt(r**2 - i**2)
            d_int = ti.cast(d, ti.i32)
            # TODO: parallelise ?
            for j in range(d_int):
                self.px.rgba[x + i, y + j] = rgba
                self.px.rgba[x + i, y - j] = rgba
                self.px.rgba[x - i, y - j] = rgba
                self.px.rgba[x - i, y + j] = rgba

    @ti.func
    def circles(self, x: ti.template(), y: ti.template(), r: ti.template(), rgba: vec4):
        for i in ti.static(range(len(x))):
            self.circle(x[i], y[i], r[i], rgba)

    @ti.func
    def triangle(self, a, b, c, rgba: vec4):
        # TODO: fill arg
        x = ti.Vector([a[0], b[0], c[0]])
        y = ti.Vector([a[1], b[1], c[1]])
        self.polygon(x, y, rgba)

    @ti.func
    def polygon(self, x: ti.template(), y: ti.template(), rgba: vec4):
        # TODO: fill arg
        # after http://www.dgp.toronto.edu/~mac/e-stuff/point_in_polygon.py
        x_min, x_max = ti.cast(x.min(), ti.i32), ti.cast(x.max(), ti.i32)
        y_min, y_max = ti.cast(y.min(), ti.i32), ti.cast(y.max(), ti.i32)
        l = len(x)
        for i, j in ti.ndrange(x_max - x_min, y_max - y_min):
            p = [x_min + i, y_min + j]
            if self._is_inside(p, x, y, l) != 0:
                # TODO: abstract out, weight?
                """
                x-1,y-1  x,y-1  x+1,y-1
                x-1,y    x,y    x+1,y
                x-1,y+1  x,y+1  x+1,y+1
                """
                _x, _y = p[0], p[1]
                self.px.rgba[_x - 1, _y - 1] = rgba
                self.px.rgba[_x - 1, _y] = rgba
                self.px.rgba[_x - 1, _y + 1] = rgba

                self.px.rgba[_x, _y - 1] = rgba
                self.px.rgba[_x, _y] = rgba
                self.px.rgba[_x, _y + 1] = rgba

                self.px.rgba[_x + 1, _y - 1] = rgba
                self.px.rgba[_x + 1, _y] = rgba
                self.px.rgba[_x + 1, _y + 1] = rgba

    @ti.func
    def _is_inside(self, p, x, y, l):
        is_inside = 0
        if self.polygon_mode == "crossing":
            is_inside = self._is_inside_crossing(p, x, y, l)
        elif self.polygon_mode == "winding":
            is_inside = self._is_inside_winding(p, x, y, l)
        return is_inside

    @ti.func
    def _is_inside_crossing(self, p, x, y, l):
        n = 0
        v0, v1 = ti.Vector([0.0, 0.0]), ti.Vector([0.0, 0.0])
        for i in range(l):
            i1 = i + 1 if i < l - 1 else 0
            v0, v1 = [x[i], y[i]], [x[i1], y[i1]]
            if (v0[1] <= p[1] and v1[1] > p[1]) or (v0[1] > p[1] and v1[1] <= p[1]):
                vt = (p[1] - v0[1]) / (v1[1] - v0[1])
                if p[0] < v0[0] + vt * (v1[0] - v0[0]):
                    n += 1
        return n % 2

    @ti.func
    def _is_inside_winding(self, p, x, y, l):
        n = 0
        v0, v1 = ti.Vector([0.0, 0.0]), ti.Vector([0.0, 0.0])
        for i in range(l):
            i1 = i + 1 if i < l - 1 else 0
            v0, v1 = [x[i], y[i]], [x[i1], y[i1]]
            if v0[1] <= p[1] and v1[1] > p[1] and (v0 - v1).cross(p - v1) > 0:
                n += 1
            elif v1[1] <= p[1] and (v0 - v1).cross(p - v1) < 0:
                n -= 1
        return n

    @ti.kernel
    def flip_x(self):
        """
        Invert image in x-axis.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba_inv[i, j] = self.px.rgba[self.x - 1 - i, j]

    @ti.kernel
    def flip_y(self):
        """
        Flip image in y-axis.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba_inv[i, j] = self.px.rgba[i, self.y - 1 - j]

    @ti.kernel
    def invert(self):
        """
        Invert image.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba_inv[i, j] = 1.0 - self.px.rgba[i, j]

    @ti.kernel
    def decay(self, evaporate: ti.f32):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] *= evaporate

    def blend_add(self, px: ti.template()):
        self._blend_add(self.rgba_from_px(px))

    @ti.kernel
    def _blend_add(self, rgba: ti.template()):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] += rgba[i, j]

    def blend_sub(self, px: ti.template()):
        self._blend_sub(self.rgba_from_px(px))

    @ti.kernel
    def _blend_sub(self, rgba: ti.template()):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] -= rgba[i, j]

    def blend_mul(self, px: ti.template()):
        self._blend_mul(self.rgba_from_px(px))

    @ti.kernel
    def _blend_mul(self, rgba: ti.template()):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] *= rgba[i, j]

    def blend_div(self, px: ti.template()):
        self._blend_div(self.rgba_from_px(px))

    @ti.kernel
    def _blend_div(self, rgba: ti.template()):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] /= rgba[i, j]

    def blend_min(self, px: ti.template()):
        self._blend_min(self.rgba_from_px(px))

    @ti.kernel
    def _blend_min(self, rgba: ti.template()):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.min(self.px.rgba[i, j], rgba[i, j])

    def blend_max(self, px: ti.template()):
        self._blend_max(self.rgba_from_px(px))

    @ti.kernel
    def _blend_max(self, rgba: ti.template()):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.max(self.px.rgba[i, j], rgba[i, j])

    def blend_diff(self, px: ti.template()):
        self._blend_diff(self.rgba_from_px(px))

    @ti.kernel
    def _blend_diff(self, rgba: ti.template()):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.abs(self.px.rgba[i, j] - rgba[i, j])

    def blend_diff_inv(self, px: ti.template()):
        self._blend_diff_inv(self.rgba_from_px(px))

    @ti.kernel
    def _blend_diff_inv(self, rgba: ti.template()):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.abs(rgba[i, j] - self.px.rgba[i, j])

    def blend_mix(self, px: ti.template(), a: ti.f32):
        self._blend_mix(self.rgba_from_px(px))

    @ti.kernel
    def _blend_mix(self, rgba: ti.template(), amount: ti.f32):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.math.mix(self.px.rgba[i, j], rgba[i, j], amount)

    @ti.kernel
    def blur(self, radius: ti.i32):
        """
        Box blur
        """
        for i, j in ti.ndrange(self.x, self.y):
            d = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    dx = (i + di) % self.x
                    dy = (j + dj) % self.y
                    d += self.px.rgba[dx, dy]
            d /= (radius * 2 + 1) ** 2
            self.px.rgba[i, j] = d

    def particles(
        self, particles: ti.template(), species: ti.template(), shape="circle"
    ):
        shape = self.shape_enum[shape]
        self._particles(particles, species, shape)

    @ti.kernel
    def _particles(self, particles: ti.template(), species: ti.template(), shape: int):
        for i in range(self.tv.p.n):
            p = particles.field[i]
            s = species[p.species]
            if p.active == 0.0:
                continue
            px = ti.cast(p.pos[0], ti.i32)
            py = ti.cast(p.pos[1], ti.i32)
            vx = ti.cast(p.pos[0] + p.vel[0] * 20, ti.i32)
            vy = ti.cast(p.pos[1] + p.vel[1] * 20, ti.i32)
            rgba = s.rgba * self.CONSTS.BRIGHTNESS
            if shape == 0:
                self.point(px, py, rgba)
            elif shape == 1:
                self.line(px, py, vx, vy, rgba)
            elif shape == 2:
                side = int(s.size) * 2
                self.rect(px, py, side, side, rgba)
            elif shape == 3:
                self.circle(px, py, p.size, rgba)
            elif shape == 4:
                a = p.pos
                b = p.pos + 1
                c = a + b
                self.triangle(a, b, c, rgba)
            # elif shape == 5:
            #     self.polygon(px, py, rgba)

    def rgba_from_px(self, px):
        if isinstance(px, Pixels):
            return px.px.rgba
        elif isinstance(px, StructField):
            return px.rgba
        elif isinstance(px, MatrixField):
            return px
        else:
            try:
                return px.px.px.rgba
            except:
                raise TypeError(f"Cannot find pixel field in {type(px)}")

    @ti.kernel
    def update(self):
        pass

    def reset(self):
        self.clear()

    def __call__(self):
        return self.get()

    @ti.func
    def rgba_inv(self):  # -> vec3:
        # TODO: rgba_inv
        pass

    # TODO: Normalise positions to [0,1] range?
    @ti.func
    def pos_to_px(self, pos: ti.math.vec2) -> ti.math.vec2:
        return pos * [self.tv.x, self.tv.y]

    @ti.func
    def px_to_pos(self, px: ti.math.vec2) -> ti.math.vec2:
        return px / [self.tv.x, self.tv.y]
