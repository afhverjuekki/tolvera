"""
Game of Life based on Taichi example

"In memory of John Horton Conway (1937 - 2020)"

https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/game_of_life.py
"""

"""
TODO: supply explicit width and height
TODO: parse rules from string
TODO: represent rules using tv.s
TODO: draw with gaps in grid
TODO: alpha?
TODO: multispecies
TODO: torus mode?
"""

import taichi as ti
# from tolvera.utils import CONSTS
from ..pixels import Pixels

@ti.data_oriented
class GOL:
    def __init__(self, tolvera, **kwargs) -> None:
        """Game of Life based on Taichi example.

        Args:
            tolvera (Tolvera): A Tolvera instance.
            gol_n (int, optional): The number of cells. Defaults to 64.
            gol_speed (float, optional): The speed factor. Defaults to 1.
            gol_cell_size (int, optional): The size of each cell. Defaults to 8.
            gol_B (list, optional): The birth rules. Defaults to [3].
            gol_S (list, optional): The survival rules. Defaults to [2, 3].
            gol_alive_c (list, optional): The colour of alive cells. Defaults to [1.0, 1.0, 1.0, 1.0].
            gol_dead_c (list, optional): The colour of dead cells. Defaults to [0.0, 0.0, 0.0, 1.0].
            gol_random (float, optional): The randomisation factor. Defaults to 0.8.
        """
        self.tv = tolvera
        self.kwargs = kwargs
        # self.CONSTS = CONSTS({"C": (ti.f32, 300.0)})
        self.n = kwargs.get('gol_n', 64)
        self.speed = ti.field(ti.f32, shape=())
        self.speed[None] = kwargs.get('gol_speed', 1)
        self.substep = ti.field(ti.i32, shape=())
        self.cell_size = kwargs.get('gol_cell_size', 8)
        self.img_size = self.n * self.cell_size
        self.w = self.h = self.img_size
        self.tv.s.gol_cells = {
            'state': {
                'alive': (ti.i32, 0, 1),
                'count': (ti.i32, 0, 8),
            }, 
            'shape': (self.n, self.n),
            'randomise': True
        }
        self.px = Pixels(self.tv, x=self.img_size, y=self.img_size)
        # https://www.conwaylife.com/wiki/Cellular_automaton#Rules
        self.B = kwargs.get('gol_B', [3]) # [2]
        self.S = kwargs.get('gol_S', [2, 3]) # [0]
        # self.tv.s.gol_rules = {
        #     "state": {
        #         "birth": (ti.i32, 0, 10),
        #         "survival": (ti.i32, 0, 10),
        #     }, "shape": 1,
        # }
        self.alive_c = kwargs.get('gol_alive_c', [1.0, 1.0, 1.0, 1.0])
        self.dead_c = kwargs.get('gol_dead_c', [0.0, 0.0, 0.0, 1.0])
        self.random = ti.field(ti.f32, shape=())
        self.random[None] = kwargs.get('gol_random', 0.8)
        self.init()

    def init(self):
        self.set_substep()

    def randomise(self):
        """Randomise the rules."""
        self.tv.s.gol_cells.randomise()
        # self.init()

    @ti.kernel
    def set_substep(self):
        if self.speed[None] > 1:
            self.substep[None] = ti.cast(self.speed[None], ti.i32)
        else:
            self.substep[None] = ti.cast(1 / self.speed[None], ti.i32)

    def set_speed(self, speed: ti.f32):
        self.speed[None] = speed
        self.set_substep()

    @ti.func
    def get_alive(self, i, j):
        alive = self.tv.s.gol_cells.field[i,j].alive
        return alive if 0 <= i < self.n and 0 <= j < self.n else 0

    @ti.func
    def get_count(self, i: ti.i32, j: ti.i32) -> ti.i32:
        return (
            self.get_alive(i - 1, j)
            + self.get_alive(i + 1, j)
            + self.get_alive(i, j - 1)
            + self.get_alive(i, j + 1)
            + self.get_alive(i - 1, j - 1)
            + self.get_alive(i + 1, j - 1)
            + self.get_alive(i - 1, j + 1)
            + self.get_alive(i + 1, j + 1)
        )

    @ti.func
    def calc_rule(self, a: ti.i32, c: ti.i32) -> ti.i32:
        if a == 0:
            for t in ti.static(self.B):
                if c == t:
                    a = 1
        elif a == 1:
            a = 0
            for t in ti.static(self.S):
                if c == t:
                    a = 1
        return a

    @ti.func
    def count_neighbours(self):
        for i, j in self.tv.s.gol_cells.field:
            self.tv.s.gol_cells.field[i,j].count = self.get_count(i, j)

    @ti.func
    def update_alive(self):
        for i, j in self.tv.s.gol_cells.field:
            cell = self.tv.s.gol_cells.field[i,j]
            self.tv.s.gol_cells.field[i,j].alive = self.calc_rule(cell.alive, cell.count)

    @ti.func
    def cell_from_point(pos: ti.math.vec2):
        """
        if gx <= x < gx + gw and gy <= y < gy + gh:
            i, j = int((x - gx) / gc), int((y - gy) / gc)
        """
        pass

    @ti.func
    def fill_area(self, x: ti.i32, y: ti.i32, w: ti.i32, h: ti.i32, alive: ti.i32):
        for i, j in ti.ndrange(w, h):
            self.tv.s.gol_cells.field[x + i, y + j].alive = alive

    @ti.kernel
    def run(self):
        self.count_neighbours()
        self.update_alive()

    @ti.kernel
    def draw(self):
        c = ti.Vector(self.dead_c)
        for i, j in self.tv.s.gol_cells.field:
            cell = self.tv.s.gol_cells.field[i,j]
            if cell.alive == 1:
                c = ti.Vector(self.alive_c)
            else:
                c = ti.Vector(self.dead_c)
            self.px.rect(i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size, c)

    def step(self):
        self.px.clear()
        if self.speed[None] > 1:
            for _ in range(self.substep[None]):
                self.run()
        else:
            if self.tv.ctx.i[None] % self.substep[None] == 0:
                self.run()
        self.draw()

    def __call__(self, *args, **kwds):
        self.step()
        return self.px
