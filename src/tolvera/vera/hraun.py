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

        self.dt = ti.field(dtype=ti.f32, shape=())
        self.dt[None] = kwargs.get('hraun_dt', 10)

        vents = kwargs.get('hraun_vents', 1)

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
            "FLOW_STABILITY_FACTOR": (ti.f32, 0.5),
            "COOLING_RATE": (ti.f32, 0.005)
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
        # Colors for temperature visualization
        self.cool_temp_color = tm.vec4(kwargs.get('temp_cool_c', [0.1, 0.1, 0.3, 1.0])) # Cool (ambient) color - Dark Blue/Purple
        self.hot_temp_color = tm.vec4(kwargs.get('temp_hot_c', [1.0, 1.0, 0.8, 1.0]))   # Hot (max temp) color - Bright Yellow/White


        self.init()

    def init(self):
        """Initialise the lava flow simulation.
        """
        vent_loc = tm.vec2(self.tv.x // 2, self.tv.y // 2)
        self.set_vent(0, vent_loc, 1.0, self.CONSTS.MAX_TEMPERATURE)
        self.init_lava_temp()

    @ti.kernel
    def init_lava_temp(self):
        """Initialise the lava temperature.
        """
        for x, y in ti.ndrange(self.tv.x, self.tv.y):
            self.tv.s.hraun_grid.field.lava_temperature[x, y, 0] = self.CONSTS.MIN_TEMPERATURE
            self.tv.s.hraun_grid.field.lava_temperature[x, y, 1] = self.CONSTS.MIN_TEMPERATURE

    @ti.kernel
    def set_vent(self, i: ti.i32, pos: tm.vec2, rate: ti.f32, temp: ti.f32):
        """Set a vent's properties.

        Args:
            i (ti.i32): The index of the vent.
            pos (tm.vec2): The position of the vent.
            rate (ti.f32): The rate of lava emission.
            temp (ti.f32): The temperature of emitted lava.
        """
        self.vents.field[i].active = 1
        self.vents.field[i].pos = pos
        self.tv.s.hraun_vents.field.rate[i] = rate
        self.tv.s.hraun_vents.field.temperature[i] = temp

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

    @ti.kernel
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
        """Phase 1: Emit lava and heat from active vents.

        Updates thickness and temperature in the next buffer based on vent emission.
        Temperature is set directly to vent temperature.

        Args:
            current_buf (ti.i32): The current buffer index (not directly used).
            next_buf (ti.i32): The next buffer index for reading/writing.
        """
        for i in self.vents.field:
            if self.vents.field[i].active == 1:
                pos = self.vents.field[i].pos
                x = ti.cast(pos[0], ti.i32) % self.tv.x
                y = ti.cast(pos[1], ti.i32) % self.tv.y
                
                v_rate = self.tv.s.hraun_vents.field.rate[i] * self.dt[None] # Scale rate by dt
                v_temp = self.tv.s.hraun_vents.field.temperature[i]
                
                # Read existing thickness from next_buf (post-flow)
                h_existing = self.tv.s.hraun_grid.field.lava_thickness[x, y, next_buf]

                # Calculate new thickness
                h_new = h_existing + v_rate

                # Set temperature directly to vent temperature
                T_new = v_temp
                
                # Ensure temperature reverts to ambient if thickness becomes negligible
                # (though h_new should be > 0 if v_rate > 0)
                if h_new <= self.CONSTS.CRITICAL_FLOW_THICKNESS:
                    T_new = self.CONSTS.MIN_TEMPERATURE
                
                # Write updated thickness and temperature back to next_buf
                self.tv.s.hraun_grid.field.lava_thickness[x, y, next_buf] = h_new
                self.tv.s.hraun_grid.field.lava_temperature[x, y, next_buf] = T_new

    @ti.func
    def calculate_cooling(self, current_buf: ti.i32, next_buf: ti.i32):
        """Phase 3.1: Calculate temperature decrease due to cooling.
        Reads thickness from current_buf, temp from next_buf (already copied).
        Writes updated temperature to next_buf ONLY if thickness is sufficient.
        """
        for x, y in ti.ndrange(self.tv.x, self.tv.y):
            # Use thickness from *before* flow/emit (current_buf) to determine if cooling applies
            h_lava = self.tv.s.hraun_grid.field.lava_thickness[x, y, current_buf]
            
            # Only apply cooling if there was significant lava in the previous step
            if h_lava > self.CONSTS.CRITICAL_FLOW_THICKNESS: 
                current_temp = self.tv.s.hraun_grid.field.lava_temperature[x, y, next_buf] # Read temp from next_buf
                if current_temp > self.CONSTS.MIN_TEMPERATURE:
                    # Simple cooling model: proportional to temp difference with ambient
                    temp_diff = current_temp - self.CONSTS.MIN_TEMPERATURE
                    cooling_amount = self.CONSTS.COOLING_RATE * temp_diff * self.dt[None] 
                    
                    new_temp = current_temp - cooling_amount
                    # Clamp to minimum temperature
                    new_temp = ti.max(self.CONSTS.MIN_TEMPERATURE, new_temp)
                    
                    # Write cooled temperature to next_buf
                    self.tv.s.hraun_grid.field.lava_temperature[x, y, next_buf] = new_temp
            # If h_lava was <= CRITICAL_FLOW_THICKNESS, do nothing to temperature in next_buf.
            # It retains its copied value unless overwritten by emit or affected by flow later? (Flow doesn't affect temp yet).


    @ti.func
    def cool(self, current_buf: ti.i32, next_buf: ti.i32):
        """Phase 3: Apply cooling to the lava.
        Wrapper function for cooling calculations.
        """
        self.calculate_cooling(current_buf, next_buf)


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
        """Main draw kernel."""
        self.draw_grid()
        self.draw_lava_temp() # Visualize temperature by default
        # self.draw_lava_thickness() # Keep for debugging if needed
        # self.draw_vents()

    @ti.func
    def draw_grid(self):
        """Draw the underlying terrain height."""
        current_buf = self.current_buffer[None]
        for i, j in ti.ndrange(self.tv.x, self.tv.y):
            # Read terrain height from the current buffer
            h = self.tv.s.hraun_grid.field.terrain_height[i, j, current_buf]
            # Normalize terrain height assuming input DEM provides values mostly in 0-1 range.
            # Clamp to ensure it's visually representable as grayscale.
            norm_h = tm.clamp(h, 0.0, 1.0) 
            self.grid.px.rgba[i, j] = tm.vec4(norm_h, norm_h, norm_h, 1.0)
    
    @ti.func
    def draw_vents(self):
        """Draw markers for active vents."""
        for i in self.vents.field:
             if self.vents.field[i].active == 1:
                pos = self.vents.field[i].pos
                # Ensure pos is within grid bounds for drawing
                x_pos = ti.cast(pos[0], ti.i32) % self.tv.x
                y_pos = ti.cast(pos[1], ti.i32) % self.tv.y
                # Use a distinct color for vents, e.g., bright red
                self.grid.circle(x_pos, y_pos, 5, tm.vec4(1.0, 0.0, 0.0, 1.0), 0)
    
    @ti.func
    def draw_lava_thickness(self):
        """Draw lava thickness."""
        current_buf = self.current_buffer[None]
        for i, j in ti.ndrange(self.tv.x, self.tv.y):
            lava_thickness = self.tv.s.hraun_grid.field.lava_thickness[i, j, current_buf]
            if lava_thickness > self.CONSTS.CRITICAL_FLOW_THICKNESS: # Use critical thickness for visibility
                # Normalize thickness using a smaller divisor (e.g., 1.0) for better visibility.
                normalized_thickness = ti.min(lava_thickness / 1.0, 1.0) 
                lava_color = tm.mix(self.min_thickness_color, self.max_thickness_color, normalized_thickness)
                self.grid.px.rgba[i, j] = lava_color
            # Pixels with thickness <= CRITICAL_FLOW_THICKNESS retain the grid color drawn by draw_grid

    @ti.func
    def draw_lava_temp(self):
        """Draw lava temperature."""
        current_buf = self.current_buffer[None] # Read the correct current buffer
        for i, j in ti.ndrange(self.tv.x, self.tv.y):
            # Read from current_buf now
            lava_thickness = self.tv.s.hraun_grid.field.lava_thickness[i, j, current_buf] 
            if lava_thickness > self.CONSTS.CRITICAL_FLOW_THICKNESS:
                # Read from current_buf now
                lava_temp = self.tv.s.hraun_grid.field.lava_temperature[i, j, current_buf]

                temp_color = self.cool_temp_color # Default cool color
                if lava_temp > self.CONSTS.SOLIDIFICATION_TEMP: # Check if above solidification temp
                    # Simple threshold visualization
                    # temp_color = tm.vec4(1.0, 0.0, 0.0, 1.0) # Set to pure red if hot
                    # Interpolated visualization
                    normalized_temp = (lava_temp - self.CONSTS.SOLIDIFICATION_TEMP) / (self.CONSTS.MAX_TEMPERATURE - self.CONSTS.SOLIDIFICATION_TEMP)
                    normalized_temp = tm.clamp(normalized_temp, 0.0, 1.0)
                    # Use SOLIDIFICATION_TEMP color as the low end, hot_temp_color as high end
                    # Let's define a solidification temp color, maybe dark red?
                    solid_color = tm.vec4(0.5, 0.0, 0.0, 1.0) 
                    temp_color = tm.mix(solid_color, self.hot_temp_color, normalized_temp)


                self.grid.px.rgba[i, j] = temp_color
            # else: keep background terrain color (drawn by draw_grid)

    @ti.kernel
    def step(self, current_buf: ti.i32, next_buf: ti.i32):
        """Step the lava flow simulation calculations."""
        # Phase 3: Cooling (apply *before* flow/emission temperature updates)
        # Note: Reads thickness from current_buf, temp from next_buf, writes cooled temp to next_buf
        self.cool(current_buf, next_buf) 

        # Phase 2: Flow 
        # Note: Reads current_buf state, writes updated thickness to next_buf
        self.flow(current_buf, next_buf)
        
        # Phase 1: Emission
        # Note: Reads next_buf state (post-flow, post-cool), writes updated thickness & temp to next_buf
        self.emit(current_buf, next_buf) # Pass current_buf for consistency, though unused
                
        # Phase 4: Solidification (Placeholder)
        # Note: Reads next_buf state (post-flow/emit/cool), writes updates to next_buf
        self.solidify(current_buf, next_buf)

    def __call__(self):
        current_buf = self.current_buffer[None]
        next_buf = 1 - current_buf
        # Copy state first, including temperature from previous step
        self.copy_state_to_next_buffer(current_buf, next_buf) 
        # Run simulation steps, operating primarily on next_buf
        self.step(current_buf, next_buf) 
        # Swap buffers for next iteration
        self.current_buffer[None] = next_buf 
        # Draw the state from the *new* current buffer
        self.draw() 
        return self.grid
