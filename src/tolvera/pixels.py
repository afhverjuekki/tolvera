"""Pixels module.

Example:
    Draw a red rectangle in the centre of the screen.
    ```py
    import taichi as ti
    from tolvera import Tolvera, run

    def main(**kwargs):
        tv = Tolvera(**kwargs)

        @ti.kernel
        def draw():
            w = 100
            tv.px.rect(tv.x/2-w/2, tv.y/2-w/2, w, w, ti.Vector([1., 0., 0., 1.]))

        @tv.render
        def _():
            tv.p()
            draw()
            return tv.px

    if __name__ == '__main__':
        run(main)
    ```
"""

import taichi as ti
from typing import Any
from taichi.lang.matrix import MatrixField
from taichi.lang.struct import StructField
from taichi.lang.field import ScalarField

from .utils import CONSTS

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4


@ti.dataclass
class Pixel:
    rgba: vec4


@ti.data_oriented
class Pixels:
    """Pixels class for drawing pixels to the screen.

    This class is used to draw pixels to the screen. It contains methods for drawing
    points, lines, rectangles, circles, triangles, and polygons. It also contains
    methods for blending pixels together, flipping pixels, inverting pixels, and
    diffusing, decaying and clearing pixels.

    It tries to follow a similar API to the Processing library.
    """
    def __init__(self, tolvera, **kwargs):
        """Initialise Pixels

        Args:
            tolvera (Tolvera): Tölvera instance.
            **kwargs: Keyword arguments.
                polygon_mode (str): Polygon mode. Defaults to "crossing".
                brightness (float): Brightness. Defaults to 1.0. 
        """
        self.tv = tolvera
        self.kwargs = kwargs
        self.polygon_mode = kwargs.get("polygon_mode", "crossing")
        self.x = kwargs.get("x", self.tv.x)
        self.y = kwargs.get("y", self.tv.y)
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

    
    def set(self, px: Any):
        """Set pixels.

        Args:
            px (Any): Pixels to set. Can be Pixels, StructField, MatrixField, etc (see rgba_from_px).
        """
        self.px.rgba = self.rgba_from_px(px)

    @ti.kernel
    def k_set(self, px: ti.template()):
        for x, y in ti.ndrange(self.x, self.y):
            self.px.rgba[x, y] = px.px.rgba[x, y]

    @ti.kernel
    def f_set(self, px: ti.template()):
        for x, y in ti.ndrange(self.x, self.y):
            self.px.rgba[x, y] = px.px.rgba[x, y]

    @ti.func
    def stamp(self, x: ti.i32, y: ti.i32, px: ti.template()):
        """Stamp pixels.

        Args:
            x (ti.i32): X position.
            y (ti.i32): Y position.
            px (ti.template): Pixels to stamp.
        """
        for i, j in ti.ndrange(px.px.shape[0], px.px.shape[1]):
            p = px.px.rgba[i, j]
            if p[0]+p[1]+p[2] > 0: # transparency
                self.px.rgba[x + i, y + j] = p

    @ti.kernel
    def from_numpy(self, img: ti.template()):
        for x, y in ti.ndrange(self.x, self.y):
            if img[x, y, 0]+img[x, y, 1]+img[x, y, 2] > 0.:
                self.px.rgba[x, y] = ti.Vector([
                    img[x, y, 0]/255.,
                    img[x, y, 1]/255.,
                    img[x, y, 2]/255.,
                    img[x, y, 3]/255.])

    def from_img(self, path: str):
        img = ti.tools.imread(path)
        img_fld = ti.field(dtype=ti.f32, shape=img.shape)
        img_fld.from_numpy(img)
        self.from_numpy(img_fld)
        return img_fld

    def get(self):
        """Get pixels."""
        return self.px

    @ti.kernel
    def clear(self):
        """Clear pixels."""
        self.px.rgba.fill(0)

    @ti.kernel
    def diffuse(self, evaporate: ti.f32):
        """Diffuse pixels.
        
        Args:
            evaporate (float): Evaporation rate.
        """
        for i, j in ti.ndrange(self.x, self.y):
            d = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    dx = (i + di) % self.x
                    dy = (j + dj) % self.y
                    d += self.px.rgba[dx, dy]
            d *= 0.99 / 9.0
            self.px.rgba[i, j] = d

    @ti.func
    def background(self, r: ti.f32, g: ti.f32, b: ti.f32):
        """Set background colour.

        Args:
            r (ti.f32): Red.
            g (ti.f32): Green.
            b (ti.f32): Blue.
        """
        bg = ti.Vector([r, g, b, 1.0])
        self.rect(0, 0, self.x, self.y, bg)

    @ti.func
    def point(self, x: ti.i32, y: ti.i32, rgba: vec4):
        """Draw point.

        Args:
            x (ti.i32): X position.
            y (ti.i32): Y position.
            rgba (vec4): Colour.
        """
        self.px.rgba[x, y] = rgba

    @ti.func
    def points(self, x: ti.template(), y: ti.template(), rgba: vec4):
        """Draw points with the same colour.

        Args:
            x (ti.template): X positions.
            y (ti.template): Y positions.
            rgba (vec4): Colour.
        """
        for i in ti.static(range(len(x))):
            self.point(x[i], y[i], rgba)

    @ti.func
    def rect(self, x: ti.i32, y: ti.i32, w: ti.i32, h: ti.i32, rgba: vec4):
        """Draw a filled rectangle.

        Args:
            x (ti.i32): X position.
            y (ti.i32): Y position.
            w (ti.i32): Width.
            h (ti.i32): Height.
            rgba (vec4): Colour.
        """
        # TODO: fill arg
        # TODO: gradients, lerp with ti.math.mix(x, y, a)
        for i, j in ti.ndrange(w, h):
            self.px.rgba[x + i, y + j] = rgba

    @ti.kernel
    def stamp(self, x: ti.i32, y: ti.i32, px: ti.template()):
        """Stamp pixels.

        Args:
            x (ti.i32): X position.
            y (ti.i32): Y position.
            px (ti.template): Pixels to stamp.
        """
        self.stamp_f(x, y, px)

    @ti.func
    def stamp_f(self, x: ti.i32, y: ti.i32, px: ti.template()):
        """Stamp pixels.

        Args:
            x (ti.i32): X position.
            y (ti.i32): Y position.
            px (ti.template): Pixels to stamp.
        """
        for i, j in ti.ndrange(px.px.shape[0], px.px.shape[1]):
            p = px.px.rgba[i, j]
            if p[0]+p[1]+p[2] > 0: # transparency
                self.px.rgba[x + i, y + j] = p

    @ti.func
    def plot(self, x, y, c, rgba):
        """Set the pixel color with blending."""
        self.px.rgba[x, y] = self.px.rgba[x, y] * (1 - c) + rgba * c

    @ti.func
    def ipart(self, x):
        return ti.math.floor(x)

    @ti.func
    def round(self, x):
        return self.ipart(x + 0.5)

    @ti.func
    def fpart(self, x):
        return x - ti.math.floor(x)

    @ti.func
    def rfpart(self, x):
        return 1 - self.fpart(x)

    @ti.func
    def line(self, x0: ti.f32, y0: ti.f32, x1: ti.f32, y1: ti.f32, rgba: vec4):
        """Draw an anti-aliased line using Xiaolin Wu's algorithm."""
        steep = ti.abs(y1 - y0) > ti.abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = y1 - y0
        gradient = dy / dx if dx != 0 else 1.0

        xend = ti.math.round(x0)
        yend = y0 + gradient * (xend - x0)
        xgap = self.rfpart(x0 + 0.5)
        xpxl1 = int(xend)
        ypxl1 = int(self.ipart(yend))
        if steep:
            self.plot(ypxl1, xpxl1, self.rfpart(yend) * xgap, rgba)
            self.plot(ypxl1 + 1, xpxl1, self.fpart(yend) * xgap, rgba)
        else:
            self.plot(xpxl1, ypxl1, self.rfpart(yend) * xgap, rgba)
            self.plot(xpxl1, ypxl1 + 1, self.fpart(yend) * xgap, rgba)

        intery = yend + gradient

        xend = ti.math.round(x1)
        yend = y1 + gradient * (xend - x1)
        xgap = self.fpart(x1 + 0.5)
        xpxl2 = int(xend)
        ypxl2 = int(self.ipart(yend))
        if steep:
            self.plot(ypxl2, xpxl2, self.rfpart(yend) * xgap, rgba)
            self.plot(ypxl2 + 1, xpxl2, self.fpart(yend) * xgap, rgba)
        else:
            self.plot(xpxl2, ypxl2, self.rfpart(yend) * xgap, rgba)
            self.plot(xpxl2, ypxl2 + 1, self.fpart(yend) * xgap, rgba)

        if steep:
            for x in range(xpxl1 + 1, xpxl2):
                self.plot(int(self.ipart(intery)), x, self.rfpart(intery), rgba)
                self.plot(int(self.ipart(intery)) + 1, x, self.fpart(intery), rgba)
                intery += gradient
        else:
            for x in range(xpxl1 + 1, xpxl2):
                self.plot(x, int(self.ipart(intery)), self.rfpart(intery), rgba)
                self.plot(x, int(self.ipart(intery)) + 1, self.fpart(intery), rgba)
                intery += gradient

    # @ti.func
    # def line(self, x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, rgba: vec4):
    #     """Draw a line using Bresenham's algorithm.

    #     Args:
    #         x0 (ti.i32): X start position.
    #         y0 (ti.i32): Y start position.
    #         x1 (ti.i32): X end position.
    #         y1 (ti.i32): Y end position.
    #         rgba (vec4): Colour.

    #     TODO: thickness
    #     TODO: anti-aliasing
    #     TODO: should lines wrap around (as two lines)?
    #     """
    #     dx = ti.abs(x1 - x0)
    #     dy = ti.abs(y1 - y0)
    #     x, y = x0, y0
    #     sx = -1 if x0 > x1 else 1
    #     sy = -1 if y0 > y1 else 1
    #     if dx > dy:
    #         err = dx / 2.0
    #         while x != x1:
    #             self.px.rgba[x, y] = rgba
    #             err -= dy
    #             if err < 0:
    #                 y += sy
    #                 err += dx
    #             x += sx
    #     else:
    #         err = dy / 2.0
    #         while y != y1:
    #             self.px.rgba[x, y] = rgba
    #             err -= dx
    #             if err < 0:
    #                 x += sx
    #                 err += dy
    #             y += sy
    #     self.px.rgba[x, y] = rgba

    @ti.func
    def lines(self, points: ti.template(), rgba: vec4):
        """Draw lines with the same colour.

        Args:
            points (ti.template): Points.
            rgba (vec4): Colour.
        """
        for i in range(points.shape[0] - 1):
            self.line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], rgba)

    @ti.func
    def circle(self, x: ti.i32, y: ti.i32, r: ti.i32, rgba: vec4):
        """Draw a filled circle.

        Args:
            x (ti.i32): X position.
            y (ti.i32): Y position.
            r (ti.i32): Radius.
            rgba (vec4): Colour.
        """
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
        """Draw circles with the same colour.

        Args:
            x (ti.template): X positions.
            y (ti.template): Y positions.
            r (ti.template): Radii.
            rgba (vec4): Colour.
        """
        for i in ti.static(range(len(x))):
            self.circle(x[i], y[i], r[i], rgba)

    @ti.func
    def triangle(self, a, b, c, rgba: vec4):
        """Draw a filled triangle.

        Args:
            a (vec2): Point A.
            b (vec2): Point B.
            c (vec2): Point C.
            rgba (vec4): Colour.
        """
        # TODO: fill arg
        x = ti.Vector([a[0], b[0], c[0]])
        y = ti.Vector([a[1], b[1], c[1]])
        self.polygon(x, y, rgba)

    @ti.func
    def polygon(self, x: ti.template(), y: ti.template(), rgba: vec4):
        """Draw a filled polygon.
        
        Polygons are drawn according to the polygon mode, which can be "crossing" 
        (default) or "winding". First, the bounding box of the polygon is calculated.
        Then, we check if each pixel in the bounding box is inside the polygon. If it
        is, we draw it (along with each neighbour pixel).

        Reference for point in polygon inclusion testing:
        http://www.dgp.toronto.edu/~mac/e-stuff/point_in_polygon.py

        Args:
            x (ti.template): X positions.
            y (ti.template): Y positions.
            rgba (vec4): Colour.
        
        TODO: fill arg
        """
        x_min, x_max = ti.cast(x.min(), ti.i32), ti.cast(x.max(), ti.i32)
        y_min, y_max = ti.cast(y.min(), ti.i32), ti.cast(y.max(), ti.i32)
        l = len(x)
        for i, j in ti.ndrange(x_max - x_min, y_max - y_min):
            p = ti.Vector([x_min + i, y_min + j])
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
    def _is_inside(self, p: vec2, x: ti.template(), y: ti.template(), l: ti.i32):
        """Check if point is inside polygon.

        Args:
            p (vec2): Point.
            x (ti.template): X positions.
            y (ti.template): Y positions.
            l (ti.i32): Number of points.

        Returns:
            int: 1 if inside, 0 if outside.
        """
        is_inside = 0
        if self.polygon_mode == "crossing":
            is_inside = self._is_inside_crossing(p, x, y, l)
        elif self.polygon_mode == "winding":
            is_inside = self._is_inside_winding(p, x, y, l)
        return is_inside

    @ti.func
    def _is_inside_crossing(self, p: vec2, x: ti.template(), y: ti.template(), l: ti.i32):
        """Check if point is inside polygon using crossing number algorithm.

        Args:
            p (vec2): Point.
            x (ti.template): X positions.
            y (ti.template): Y positions.
            l (ti.i32): Number of points.

        Returns:
            int: 1 if inside, 0 if outside.
        """
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
    def _is_inside_winding(self, p: vec2, x: ti.template(), y: ti.template(), l: ti.i32):
        """Check if point is inside polygon using winding number algorithm.

        Args:
            p (vec2): Point.
            x (ti.template): X positions.
            y (ti.template): Y positions.
            l (ti.i32): Number of points.

        Returns:
            int: 1 if inside, 0 if outside.
        """
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
        """Flip image in x-axis."""
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = self.px.rgba[self.x - 1 - i, j]

    @ti.kernel
    def flip_y(self):
        """Flip image in y-axis."""
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = self.px.rgba[i, self.y - 1 - j]

    @ti.kernel
    def invert(self):
        """Invert image."""
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = 1.0 - self.px.rgba[i, j]

    @ti.kernel
    def decay(self, rate: ti.f32):
        """Decay pixels.

        Args:
            rate (ti.f32): decay rate.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] *= rate

    def blend_add(self, px: ti.template()):
        """Blend by adding pixels together (Python scope).

        Args:
            px (ti.template): Pixels to blend with.
        """
        self._blend_add(self.rgba_from_px(px))

    @ti.kernel
    def _blend_add(self, rgba: ti.template()):
        """Blend by adding pixels together (Taichi scope).

        Args:
            rgba (ti.template): Pixels to blend with.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] += rgba[i, j]

    def blend_sub(self, px: ti.template()):
        """Blend by subtracting pixels (Python scope).

        Args:
            px (ti.template): Pixels to blend with.
        """
        self._blend_sub(self.rgba_from_px(px))

    @ti.kernel
    def _blend_sub(self, rgba: ti.template()):
        """Blend by subtracting pixels (Taichi scope).

        Args:
            rgba (ti.template): Pixels to blend with.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] -= rgba[i, j]

    def blend_mul(self, px: ti.template()):
        """Blend by multiplying pixels (Python scope).

        Args:
            px (ti.template): Pixels to blend with.
        """
        self._blend_mul(self.rgba_from_px(px))

    @ti.kernel
    def _blend_mul(self, rgba: ti.template()):
        """Blend by multiplying pixels (Taichi scope).

        Args:
            rgba (ti.template): Pixels to blend with.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] *= rgba[i, j]

    def blend_div(self, px: ti.template()):
        """Blend by dividing pixels (Python scope).

        Args:
            px (ti.template): Pixels to blend with.
        """
        self._blend_div(self.rgba_from_px(px))

    @ti.kernel
    def _blend_div(self, rgba: ti.template()):
        """Blend by dividing pixels (Taichi scope).

        Args:
            rgba (ti.template): Pixels to blend with.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] /= rgba[i, j]

    def blend_min(self, px: ti.template()):
        """Blend by taking the minimum of each pixel (Python scope).

        Args:
            px (ti.template): Pixels to blend with.
        """
        self._blend_min(self.rgba_from_px(px))

    @ti.kernel
    def _blend_min(self, rgba: ti.template()):
        """Blend by taking the minimum of each pixel (Taichi scope).

        Args:
            rgba (ti.template): Pixels to blend with.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.min(self.px.rgba[i, j], rgba[i, j])

    def blend_max(self, px: ti.template()):
        """Blend by taking the maximum of each pixel (Python scope).

        Args:
            px (ti.template): Pixels to blend with.
        """
        self._blend_max(self.rgba_from_px(px))

    @ti.kernel
    def _blend_max(self, rgba: ti.template()):
        """Blend by taking the maximum of each pixel (Taichi scope).

        Args:
            rgba (ti.template): Pixels to blend with.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.max(self.px.rgba[i, j], rgba[i, j])

    def blend_diff(self, px: ti.template()):
        """Blend by taking the difference of each pixel (Python scope).

        Args:
            px (ti.template): Pixels to blend with.
        """
        self._blend_diff(self.rgba_from_px(px))

    @ti.kernel
    def _blend_diff(self, rgba: ti.template()):
        """Blend by taking the difference of each pixel (Taichi scope).

        Args:
            rgba (ti.template): Pixels to blend with.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.abs(self.px.rgba[i, j] - rgba[i, j])

    def blend_diff_inv(self, px: ti.template()):
        """Blend by taking the inverse difference of each pixel (Python scope).

        Args:
            px (ti.template): Pixels to blend with.
        """
        self._blend_diff_inv(self.rgba_from_px(px))

    @ti.kernel
    def _blend_diff_inv(self, rgba: ti.template()):
        """Blend by taking the inverse difference of each pixel (Taichi scope).

        Args:
            rgba (ti.template): Pixels to blend with.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.abs(rgba[i, j] - self.px.rgba[i, j])

    def blend_mix(self, px: ti.template(), amount: ti.f32):
        """Blend by mixing pixels (Python scope).

        Args:
            px (ti.template): Pixels to blend with.
            amount (ti.f32): Amount to mix.
        """
        self._blend_mix(self.rgba_from_px(px), amount)

    @ti.kernel
    def _blend_mix(self, rgba: ti.template(), amount: ti.f32):
        """Blend by mixing pixels (Taichi scope).

        Args:
            rgba (ti.template): Pixels to blend with.
            amount (ti.f32): Amount to mix.
        """
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i, j] = ti.math.mix(self.px.rgba[i, j], rgba[i, j], amount)

    @ti.kernel
    def blur(self, radius: ti.i32):
        """Blur pixels.

        Args:
            radius (ti.i32): Blur radius.
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
        """Draw particles.

        Args:
            particles (ti.template): Particles.
            species (ti.template): Species.
            shape (str, optional): Shape. Defaults to "circle".
        """
        shape = self.shape_enum[shape]
        self._particles(particles, species, shape)

    @ti.kernel
    def _particles(self, particles: ti.template(), species: ti.template(), shape: int):
        """Draw particles.

        Args:
            particles (ti.template): Particles.
            species (ti.template): Species.
            shape (int): Shape enum value.
        """
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
        """Get rgba from pixels.

        Args:
            px (Any): Pixels to get rgba from.

        Raises:
            TypeError: If pixel field cannot be found.

        Returns:
            MatrixField: RGBA matrix field.
        """
        if isinstance(px, Pixels):
            return px.px.rgba
        elif isinstance(px, StructField):
            return px.rgba
        elif isinstance(px, MatrixField):
            return px
        elif isinstance(px, ScalarField):
            return px
        else:
            try:
                return px.px.px.rgba
            except:
                raise TypeError(f"Cannot find pixel field in {type(px)}")

    def __call__(self):
        """Call returns pixels."""
        return self.get()
