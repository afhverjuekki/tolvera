import taichi as ti
import taichi.math as tm

from ..particles import Particles
from ..pixels import Pixels
from ..utils import CONSTS

@ti.data_oriented
class Hraun:
    def __init__(self, tolvera, **kwargs) -> None:
        """Lava flow simulation based on hraunvera."""
        self.tv = tolvera
        self.kwargs = kwargs

        vents = kwargs.get('vents', 1)

        self.CONSTS = CONSTS({
            "MIN_THICKNESS": (ti.f32, 0.0),
            "MAX_THICKNESS": (ti.f32, 10.0),
            "MIN_EMISSION_RATE": (ti.f32, 0.0),
            "MAX_EMISSION_RATE": (ti.f32, 1.0),
            "MIN_VISCOSITY": (ti.f32, 1.0),
            "MAX_VISCOSITY": (ti.f32, 10000.0),
            "MIN_TEMPERATURE": (ti.f32, 20.0),
            "MAX_TEMPERATURE": (ti.f32, 1200.0),
            "SOLIDIFICATION_TEMP": (ti.f32, 800.0),
            "MIN_HEIGHT": (ti.f32, 0.0),
            "MAX_HEIGHT": (ti.f32, 1000.0),
            "CRITICAL_FLOW_THICKNESS": (ti.f32, 0.01),
            "MAX_OUTFLOW_FRACTION": (ti.f32, 0.25),
            "FLOW_STABILITY_FACTOR": (ti.f32, 0.5)
        })

        self.tv.s.hraun_vents = {
            'state': {
                'rate': (ti.f32, self.CONSTS.MIN_EMISSION_RATE, self.CONSTS.MAX_EMISSION_RATE),
                'temperature': (ti.f32, self.CONSTS.SOLIDIFICATION_TEMP, self.CONSTS.MAX_TEMPERATURE),
            },
            'shape': (vents,),
            'randomise': False
        }

        self.tv.s.hraun_grid = {
            'state': {
                # Phase 2: Terrain & Flow
                'terrain_height': (ti.f32, self.CONSTS.MIN_HEIGHT, self.CONSTS.MAX_HEIGHT),
                'lava_thickness': (ti.f32, self.CONSTS.MIN_THICKNESS, self.CONSTS.MAX_THICKNESS),

                # Phase 3: Temperature
                'lava_temperature': (ti.f32, self.CONSTS.MIN_TEMPERATURE, self.CONSTS.MAX_TEMPERATURE),

                # Phase 4: Solidification
                'solidified_lava_height': (ti.f32, self.CONSTS.MIN_HEIGHT, self.CONSTS.MAX_HEIGHT),

                # Phase 5: Viscosity (Could potentially be calculated on-the-fly in kernels)
                'viscosity': (ti.f32, self.CONSTS.MIN_VISCOSITY, self.CONSTS.MAX_VISCOSITY),

                # Optional: Active flag for optimization (e.g., only process cells with thickness > critical)
                'active': (ti.i32, 0, 1),
                
                # Intermediate flow vectors (N, S, E, W outflow volume)
                # Shape is (x, y), no buffer dim needed as it's recalculated each step
                'flow_out': (tm.vec4, 0.0, 1.0), # Placeholder range
            },
            'shape': (self.tv.x, self.tv.y, 2),
            'randomise': False
        }
        self.current_buffer = ti.field(dtype=ti.i32, shape=())

        self.vents = Particles(self.tv, n=vents)
        self.grid = Pixels(self.tv, **kwargs)
        # Intermediate field for flow calculation (stores outflow N, S, E, W)
        # self.flow_out = ti.Vector.field(4, dtype=ti.f32, shape=(self.tv.x, self.tv.y))

        # TODO: manage colors in a more Tölvera-like way
        self.max_thickness_color = tm.vec4(kwargs.get('lava_max_c', [1.0, 0.5, 0.0, 1.0])) # Orange/Red for hot lava
        self.min_thickness_color = tm.vec4(kwargs.get('lava_min_c', [0.2, 0.1, 0.1, 1.0])) # Dark grey for cool/thin lava
        self.background_color = tm.vec4(kwargs.get('lava_bg_c', [0.0, 0.0, 0.0, 1.0])) # Black background

        self.init()

    @ti.kernel
    def init(self):
        """Initialise the lava flow simulation.
        """
        self.set_vent(0, tm.vec2(self.tv.x // 2, self.tv.y // 2), 1.0)

    @ti.func
    def set_vent(self, i: ti.i32, pos: tm.vec2, rate: ti.f32):
        """Set a vent.

        Args:
            i (ti.i32): The index of the vent.
            pos (tm.vec2): The position of the vent.
            rate (ti.f32): The rate of lava emission.
        """
        self.vents.field[i].active = 1
        self.vents.field[i].pos = pos
        self.tv.s.hraun_vents.field.rate[i] = rate

    def set_dem(self, dem: ti.template()):
        """Set the DEM heightmap.
        """
        self.set_dem_k(dem.px)

    @ti.kernel
    def set_dem_k(self, dem: ti.template()):
        """Set the DEM heightmap.
        """
        self.set_dem_f(dem)

    @ti.func
    def set_dem_f(self, dem: ti.template()):
        """Set the terrain height.
        """
        # Initialize terrain height in both buffers to be the same
        for x, y in ti.ndrange(self.tv.x, self.tv.y):
            demxy = dem[x, y].rgba
            dem_height = (demxy.x + demxy.y + demxy.z) / 3.0
            self.tv.s.hraun_grid.field.terrain_height[x, y, 0] = dem_height
            self.tv.s.hraun_grid.field.terrain_height[x, y, 1] = dem_height

    @ti.func
    def deactivate_vent(self, i: ti.i32):
        """Deactivate a vent.

        Args:
            i (ti.i32): The index of the vent.
        """
        self.vents.field[i].active = 0
        self.tv.s.hraun_vents.field.rate[i] = 0.0

    @ti.kernel
    def copy_state_to_next_buffer(self, current_buf: ti.i32, next_buf: ti.i32):
        """Kernel to copy relevant state from current buffer to next buffer."""
        for x, y in ti.ndrange(self.tv.x, self.tv.y):
            self.tv.s.hraun_grid.field.lava_thickness[x, y, next_buf] = self.tv.s.hraun_grid.field.lava_thickness[x, y, current_buf]
            self.tv.s.hraun_grid.field.lava_temperature[x, y, next_buf] = self.tv.s.hraun_grid.field.lava_temperature[x, y, current_buf]
            self.tv.s.hraun_grid.field.solidified_lava_height[x, y, next_buf] = self.tv.s.hraun_grid.field.solidified_lava_height[x, y, current_buf]

    @ti.func
    def get_total_height(self, x, y, buf): 
        """Calculate total height (terrain + solidified + lava) for a cell."""
        terrain = self.tv.s.hraun_grid.field.terrain_height[x, y, buf] 
        solid = self.tv.s.hraun_grid.field.solidified_lava_height[x, y, buf]
        lava = self.tv.s.hraun_grid.field.lava_thickness[x, y, buf]
        return terrain + solid + lava

    @ti.func
    def flow(self, current_buf: ti.i32, next_buf: ti.i32):
        """Phase 2: Flow.
        Calculates potential flows and applies them to the next buffer.

        Flow is calculated in two steps:
        1. Calculate potential flows.
        2. Apply flows to the next buffer.

        Args:
            current_buf (ti.i32): The current buffer index.
            next_buf (ti.i32): The next buffer index.
        """
        self.calculate_potential_flows(current_buf)
        self.apply_flows(current_buf, next_buf)

    @ti.func
    def calculate_potential_flows(self, current_buf: ti.i32):
        """Phase 2.1: Calculate potential outflow volume for each cell to its neighbors.
        Stores results in self.flow_out[x, y][neighbor_index].
        neighbor_index: 0=N(+y), 1=S(-y), 2=E(+x), 3=W(-x)
        """
        for x, y in ti.ndrange(self.tv.x, self.tv.y):
            # Access flow_out using the current buffer index
            self.tv.s.hraun_grid.field.flow_out[x, y, current_buf] = ti.Vector([0.0, 0.0, 0.0, 0.0]) # Clear previous flows
            
            h_lava = self.tv.s.hraun_grid.field.lava_thickness[x, y, current_buf]
            if h_lava <= self.CONSTS.CRITICAL_FLOW_THICKNESS:
                continue

            total_height_xy = self.get_total_height(x, y, current_buf)
            
            sum_potential_outflow_dh = 0.0
            potential_dh = ti.Vector([0.0, 0.0, 0.0, 0.0])
            neighbor_offsets = ti.Vector([(0, 1), (0, -1), (1, 0), (-1, 0)])

            # Calculate potential height differences to lower neighbors
            for i in ti.static(range(4)): # N, S, E, W
                nx, ny = x + neighbor_offsets[i,0], y + neighbor_offsets[i,1]
                
                if 0 <= nx < self.tv.x and 0 <= ny < self.tv.y:
                    total_height_n = self.get_total_height(nx, ny, current_buf)
                    dh = total_height_xy - total_height_n
                    if dh > 0:
                        potential_dh[i] = dh
                        sum_potential_outflow_dh += dh

            if sum_potential_outflow_dh <= 1e-6:
                continue # No lower neighbors to flow to

            # Calculate outflow distribution
            available_lava = h_lava * self.CONSTS.MAX_OUTFLOW_FRACTION
            total_calculated_outflow = 0.0
            temp_flow_out = ti.Vector([0.0, 0.0, 0.0, 0.0])

            for i in ti.static(range(4)):
                dh = potential_dh[i]
                if dh > 0:
                    proportion = dh / sum_potential_outflow_dh
                    flow_volume = available_lava * proportion
                    flow_volume = ti.min(flow_volume, dh * self.CONSTS.FLOW_STABILITY_FACTOR)
                    flow_volume = ti.max(0.0, flow_volume)
                    temp_flow_out[i] = flow_volume
                    total_calculated_outflow += flow_volume

            # Scale outflows if total calculated exceeds available lava thickness
            scale_factor = 1.0
            if total_calculated_outflow > h_lava:
                scale_factor = h_lava / total_calculated_outflow
            
            # Store final scaled outflow values in the current buffer
            for i in ti.static(range(4)):
                self.tv.s.hraun_grid.field.flow_out[x, y, current_buf][i] = temp_flow_out[i] * scale_factor

    @ti.func
    def apply_flows(self, current_buf: ti.i32, next_buf: ti.i32):
        """Phase 2.2: Update lava thickness in the next buffer based on calculated flows.
        Reads self.flow_out and writes to lava_thickness in next_buf.
        """
        for x, y in ti.ndrange(self.tv.x, self.tv.y):
            h_lava_current = self.tv.s.hraun_grid.field.lava_thickness[x, y, current_buf]
            
            # Sum outflows *from* this cell (x, y), reading from current_buf
            outflow_from_xy = self.tv.s.hraun_grid.field.flow_out[x, y, current_buf].sum()
            
            # Calculate inflow *to* this cell by summing neighbors' outflows towards (x, y), reading from current_buf
            inflow_to_xy = 0.0
            # Inflow from North neighbor (x, y+1) which flows South (index 1)
            nx, ny = x, y + 1
            if 0 <= ny < self.tv.y: inflow_to_xy += self.tv.s.hraun_grid.field.flow_out[nx, ny, current_buf][1]
            # Inflow from South neighbor (x, y-1) which flows North (index 0)
            nx, ny = x, y - 1
            if 0 <= ny < self.tv.y: inflow_to_xy += self.tv.s.hraun_grid.field.flow_out[nx, ny, current_buf][0]
            # Inflow from East neighbor (x+1, y) which flows West (index 3)
            nx, ny = x + 1, y
            if 0 <= nx < self.tv.x: inflow_to_xy += self.tv.s.hraun_grid.field.flow_out[nx, ny, current_buf][3]
            # Inflow from West neighbor (x-1, y) which flows East (index 2)
            nx, ny = x - 1, y
            if 0 <= nx < self.tv.x: inflow_to_xy += self.tv.s.hraun_grid.field.flow_out[nx, ny, current_buf][2]

            net_flow = inflow_to_xy - outflow_from_xy
            new_thickness = h_lava_current + net_flow
            
            # Write final thickness to next buffer, ensuring non-negative
            self.tv.s.hraun_grid.field.lava_thickness[x, y, next_buf] = ti.max(0.0, new_thickness)

    @ti.func
    def emit(self, current_buf: ti.i32, next_buf: ti.i32):
        """Emit lava from active vents.

        Reads `lava_thickness` from `self.tv.s.hraun_grid.field` in the next buffer
        (which already contains the result of the flow calculation).
        Adds the emission rate to this value and writes it back to `next_buf`.

        Args:
            current_buf (ti.i32): The current buffer index (not used for reads here).
            next_buf (ti.i32): The next buffer index.
        """
        for i in self.vents.field:
            if self.vents.field[i].active == 1:
                pos = self.vents.field[i].pos
                x = ti.cast(pos[0], ti.i32) % self.tv.x
                y = ti.cast(pos[1], ti.i32) % self.tv.y
                v_rate = self.tv.s.hraun_vents.field.rate[i]
                # v_temp = self.tv.s.hraun_vents.field.temperature[i] # Temperature emission will be handled later

                # Read current thickness in next_buf (post-flow), add emission, write back
                current_thickness_post_flow = self.tv.s.hraun_grid.field.lava_thickness[x, y, next_buf]
                self.tv.s.hraun_grid.field.lava_thickness[x, y, next_buf] = current_thickness_post_flow + v_rate

    @ti.func
    def cool(self, current_buf: ti.i32, next_buf: ti.i32):
        """Phase 3: Apply cooling to the lava.

        Reads `lava_thickness` and `lava_temperature` from `self.tv.s.hraun_grid.field`.
        Decreases `lava_temperature` based on a cooling rate.
        Ensures temperature does not drop below `MIN_TEMPERATURE` (ambient).

        Phase 6 Enhancement: Cooling rate will depend on `lava_thickness` and temperature
                             (surface vs. interior cooling).
        """
        # Implementation for cooling calculation goes here
        # Reads from current_buf/next_buf(?), writes to next_buf
        pass

    @ti.func
    def solidify(self, current_buf: ti.i32, next_buf: ti.i32):
        """Phase 4: Solidify lava that has cooled sufficiently.

        Reads `lava_temperature`, `lava_thickness`, and `solidified_lava_height`
        from `self.tv.s.hraun_grid.field`.
        Checks if `lava_temperature` is less than or equal to `SOLIDIFICATION_TEMP`.
        If true:
            - Adds `lava_thickness` to `solidified_lava_height`.
            - Resets `lava_thickness` to 0.0.
            - Resets `lava_temperature` to 0.0 (or `MIN_TEMPERATURE`).
        """
        # Implementation for solidification check and update goes here
        # Reads from current_buf/next_buf(?), writes to next_buf
        pass

    @ti.kernel
    def draw(self):
        self.draw_grid()
        self.draw_lava()
        self.draw_vents()

    @ti.func
    def draw_grid(self):
        # Get the index for the current state buffer
        current_buf = self.current_buffer[None]
        # Iterate only over the spatial dimensions
        for i, j in ti.ndrange(self.tv.x, self.tv.y):
            # Read terrain height from the current buffer
            # Terrain height is static, so reading from buffer 0 is also fine
            # if initialized correctly, but using current_buf is consistent.
            h = self.tv.s.hraun_grid.field[i, j, current_buf].terrain_height
            self.grid.px.rgba[i, j] = tm.vec4(h, h, h, 1.0)
    
    @ti.func
    def draw_vents(self):
        for i in self.vents.field:
            pos = self.vents.field[i].pos
            self.grid.circle(pos[0], pos[1], 10, tm.vec4(1.0, 0.0, 0.0, 1.0), 0)
    
    @ti.func
    def draw_lava(self):
        # Get the index for the current state buffer
        current_buf = self.current_buffer[None]
        # Iterate only over the spatial dimensions
        for i, j in ti.ndrange(self.tv.x, self.tv.y):
            # Read lava thickness from the current buffer
            lava_thickness = self.tv.s.hraun_grid.field[i, j, current_buf].lava_thickness
            if lava_thickness > 0.0:
                # Normalize thickness for color mapping (e.g., clamp to 0-1 range for visualization)
                # Adjust the upper limit (1.0 here) as needed for better visual range
                normalized_thickness = ti.min(lava_thickness / 1.0, 1.0) 
                
                # Interpolate between min and max colors
                lava_color = tm.mix(self.min_thickness_color, self.max_thickness_color, normalized_thickness)
                
                # Assuming draw_grid already drew the terrain, only overwrite if lava exists
                self.grid.px.rgba[i, j] = lava_color
            # Note: The old commented-out complex drawing logic would also read from current_buf

    @ti.kernel
    def step(self, current_buf: ti.i32, next_buf: ti.i32):
        """Step the lava flow simulation calculations."""
        # Phase 2: Flow is now handled by these two kernels
        self.flow(current_buf, next_buf)
        
        # Phase 1: Emission
        self.emit(current_buf, next_buf)
        
        # Phase 3: Cooling (Placeholder)
        self.cool(current_buf, next_buf)
        
        # Phase 4: Solidification (Placeholder)
        self.solidify(current_buf, next_buf)

    def __call__(self):
        current_buf = self.current_buffer[None]
        next_buf = 1 - current_buf
        self.copy_state_to_next_buffer(current_buf, next_buf)
        self.step(current_buf, next_buf)
        self.current_buffer[None] = next_buf
        self.draw()
        return self.grid
