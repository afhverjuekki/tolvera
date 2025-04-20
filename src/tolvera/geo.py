import gzip
import taichi as ti
import numpy as np
import rasterio as rio
import cv2 as cv
from tolvera.pixels import Pixels

def get_metadata(file_path):
    """
    Retrieve metadata from a GeoTIFF file.

    Parameters:
    - file_path (str): Path to the GeoTIFF file.

    Returns:
    - metadata (dict): Metadata dictionary containing information like profile, CRS, bounds, etc.
    """
    with rio.open(file_path) as src:
        metadata = src.meta
        metadata['crs'] = src.crs.to_dict() if src.crs else None
        metadata['bounds'] = src.bounds
    
    return metadata

def px_to_geo(file_path, row, col):
    """
    Convert pixel coordinates to geographic coordinates (latitude, longitude).

    Parameters:
    - file_path (str): Path to the GeoTIFF file.
    - row (int): Pixel row index.
    - col (int): Pixel column index.

    Returns:
    - (float, float): Tuple containing latitude and longitude.
    """
    with rio.open(file_path) as src:
        transform = src.transform
        # Convert pixel coordinates to geographic coordinates
        lon, lat = rio.transform.xy(transform, row, col)
    
    return lat, lon

def geo_to_px(file_path, lat, lon):
    """
    Convert geographic coordinates (latitude, longitude) to pixel coordinates.

    Parameters:
    - file_path (str): Path to the GeoTIFF file.
    - lat (float): Latitude.
    - lon (float): Longitude.

    Returns:
    - (int, int): Tuple containing pixel row and column indices.
    """
    with rio.open(file_path) as src:
        transform = src.transform
        # Convert geographic coordinates to pixel coordinates
        row, col = rio.transform.rowcol(transform, lon, lat)
    
    return row, col

def load(file_path, band=1):
    """
    Load a GeoTIFF file.

    Parameters:
    - file_path (str): Path to the GeoTIFF file.
    - band (int): Band index (default: 1).

    Returns:
    - data (numpy.ndarray): Array containing the raster data.
    """
    with rio.open(file_path) as src:
        data = src.read(band)
    
    return data

def load_window(file_path, band=1, row_start=0, row_end=None, col_start=0, col_end=None, norm=True):
    """
    Load a specific window (subset) of a GeoTIFF file.

    Parameters:
    - file_path (str): Path to the GeoTIFF file.
    - band (int): Band index (default: 1).
    - row_start, row_end (int): Start and end indices for rows (inclusive).
    - col_start, col_end (int): Start and end indices for columns (inclusive).

    Returns:
    - window_data (numpy.ndarray): Array containing the window data.
    """
    with rio.open(file_path) as src:
        if row_end is None:
            row_end = src.height - 1
        if col_end is None:
            col_end = src.width - 1
        
        window_data = src.read(band, window=((row_start, row_end + 1), (col_start, col_end + 1)))
    
    if norm:
        window_data = normalise(window_data)

    return window_data

def load_chunks(file_path, band=1, chunk_size=1000):
    """
    Load GeoTIFF data in chunks (tiles).

    Parameters:
    - file_path (str): Path to the GeoTIFF file.
    - band (int): Band index (default: 1).
    - chunk_size (int): Size of each chunk (default: 1000).

    Yields:
    - chunk_data (numpy.ndarray): Array containing the chunk data.
    """
    with rio.open(file_path) as src:
        height = src.height
        width = src.width
        
        for row_start in range(0, height, chunk_size):
            for col_start in range(0, width, chunk_size):
                row_end = min(row_start + chunk_size - 1, height - 1)
                col_end = min(col_start + chunk_size - 1, width - 1)
                
                chunk_data = src.read(band, window=((row_start, row_end + 1), (col_start, col_end + 1)))
                yield chunk_data

def normalise(data):
    """
    Normalise raster data to the range [0, 1].

    Parameters:
    - data (numpy.ndarray): Array containing the raster data.

    Returns:
    - normalised_data (numpy.ndarray): Array containing the normalised data.
    """

    # Ensure the data type is float for the operations
    data = data.astype(np.float32)
    
    # Identify and handle NoData values
    nodata_value = np.nan
    if hasattr(data, 'mask') and hasattr(data, 'fill_value'):
        nodata_value = data.fill_value
    else:
        unique, counts = np.unique(data, return_counts=True)
        nodata_value = unique[np.argmax(counts)]  # Assuming the most frequent value as NoData
    
    print(f"NoData Value: {nodata_value}")

    # Mask the NoData values
    data = np.ma.masked_equal(data, nodata_value).filled(np.nan)
    
    # Handling NaNs or Infinities
    if np.isnan(data).any() or np.isinf(data).any():
        print("Warning: Input data contains NaN or Infinity values.")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs and Infs by converting to 0
    
    # Exclude extreme values
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    print(f"Min: {min_val}, Max: {max_val}")
    
    # Handling edge cases
    if min_val == max_val:
        return np.zeros_like(data)  # All values are the same, return zeros
    
    # Clipping extreme values if necessary
    # For example, clip all values outside the 1st and 99th percentiles
    p1 = np.nanpercentile(data, 1)
    p99 = np.nanpercentile(data, 100)
    clipped_data = np.clip(data, p1, p99)

    min_val_clipped = np.nanmin(clipped_data)
    max_val_clipped = np.nanmax(clipped_data)
    print(f"Clipped Min: {min_val_clipped}, Clipped Max: {max_val_clipped}")
    
    normalised_data = (clipped_data - min_val_clipped) / (max_val_clipped - min_val_clipped)
    
    return normalised_data.astype(np.float32)

@ti.func
def color_map_1d(val, r=0., g=0., b=0.):
    """Weighted colormap"""
    val = 1 - val
    r = ti.max(0, 1 - ti.abs(val - r))
    g = ti.max(0, 1 - ti.abs(val - g))
    b = ti.max(0, 1 - ti.abs(val - b))
    return ti.Vector([r, g, b, 1])

@ti.data_oriented
class Geo:
    def __init__(self, tolvera, **kwargs):
        self.tv = tolvera
        self.kwargs = kwargs
        self.x, self.y = self.tv.x, self.tv.y
        self.data = ti.field(ti.f32, (self.x, self.y))
        self.should_clamp = kwargs.get('clamp', True)
        self.px = Pixels(self.tv)
        self.wx, self.wy = 0,0
        self.init_colors()
        self.zoom = 1.0  # Current zoom level (1.0 = 100%)
        self.view_x = 0.0 # X coordinate of the view center in source pixels
        self.view_y = 0.0 # Y coordinate of the view center in source pixels
        self.src_w, self.src_h = 0, 0 # Dimensions of the source data
        self.disp_w, self.disp_h = self.x, self.y # Display dimensions (already set)
        self.src = None # Initialize src

    def save_npy(self, file_path, data):
        f = gzip.GzipFile(file_path, 'wb')
        np.save(f, data)
        f.close()

    def load_npy(self, file_path):
        f = gzip.GzipFile(file_path, 'rb')
        data = np.load(f)
        f.close()
        return data

    def from_npy(self, file_path):
        self.file_path = file_path
        if self.file_path is None:
            raise ValueError(f"Incorrect or missing file path in {self.file_path}")
        else:
            try:
                print(f"Loading NPY file: {self.file_path}")
                data = self.load_npy(self.file_path)
                self.src = data
                self.src_h, self.src_w = self.src.shape[0], self.src.shape[1]
                self.view_x = self.src_w / 2
                self.view_y = self.src_h / 2
                self.update_view()
                print(f"Finished loading {self.file_path}. Source shape=(w:{self.src_w}, h:{self.src_h})")
            except Exception as e:
                raise ValueError(f"Error loading NPY file: {e}")

    def from_geotiff(self, file_path, band=1, to_numpy=True):
        self.file_path = file_path
        self.band = band
        if self.file_path is None:
            raise ValueError(f"Incorrect or missing file path in {self.file_path}")
        else:
            try:
                print(f"Loading GeoTIFF file: {self.file_path}")
                with rio.open(self.file_path) as src_rio: # Renamed to avoid clash with self.src
                    data = src_rio.read(self.band)
                    data = self.process_geotiff(data)
                    self.src = data
                    self.src_w, self.src_h = src_rio.meta['width'], src_rio.meta['height']
                    self.view_x = self.src_w / 2
                    self.view_y = self.src_h / 2
                    self.update_view()
                    if to_numpy:
                        self.save_npy(file_path.replace('.tif', '.npy.gz'), data)
                print(f"Finished loading {self.file_path}. Source shape=(w:{self.src_w}, h:{self.src_h})")
            except Exception as e:
                raise ValueError(f"Error loading GeoTIFF file: {e}")

    def process_geotiff(self, data):
        data = normalise(data)
        # data = np.fliplr(data)
        # data = np.rot90(data)
        return data

    def init_colors(self):
        self.r, self.g, self.b = ti.field(ti.f32, ()), ti.field(ti.f32, ()), ti.field(ti.f32, ())
        self.set_colors(*(0.1, 0.5, 0.9))

    @ti.kernel
    def set_colors(self, r: ti.f32, g: ti.f32, b: ti.f32):
        self.r[None] = r
        self.g[None] = g
        self.b[None] = b

    @ti.kernel
    def set_px(self):
        for x,y in ti.ndrange(self.data.shape[0], self.data.shape[1]):
            y_inverted = self.disp_h - 1 - y
            self.px.px.rgba[x,y] = color_map_1d(self.data[x, y_inverted], self.r[None], self.g[None], self.b[None])

    def update_view(self):
        """
        Updates the self.data field based on the current view (pan/zoom).
        Extracts the relevant window from self.src, resizes it, and copies to GPU.
        """
        if self.src is None or self.src_w == 0 or self.src_h == 0:
             print("[Geo.update_view] Warning: Source data not loaded or has zero dimensions.")
             self.data.fill(0.0)
             return

        # 1. Calculate source window dimensions based on zoom
        # The size of the window we need to extract from the source (in source pixels)
        src_window_w = self.disp_w / self.zoom
        src_window_h = self.disp_h / self.zoom

        # 2. Calculate source window top-left corner coordinates (float first, then int)
        # Use view_x, view_y as the center of the view
        src_x_start_f = self.view_x - src_window_w / 2
        src_y_start_f = self.view_y - src_window_h / 2

        # Convert to integer indices for slicing
        src_x_start = int(np.floor(src_x_start_f))
        src_y_start = int(np.floor(src_y_start_f))
        # Calculate end coordinates based on integer start and float size
        src_x_end = int(np.ceil(src_x_start_f + src_window_w))
        src_y_end = int(np.ceil(src_y_start_f + src_window_h))

        # 3. Clamp coordinates to source boundaries [0, src_w) and [0, src_h)
        clamped_x_start = max(0, src_x_start)
        clamped_y_start = max(0, src_y_start)
        clamped_x_end = min(self.src_w, src_x_end)
        clamped_y_end = min(self.src_h, src_y_end)

        # Calculate the actual width and height of the clamped region
        clamped_w = clamped_x_end - clamped_x_start
        clamped_h = clamped_y_end - clamped_y_start

        # 4. Extract the clamped window from self.src (NumPy slicing: [row, col] -> [y, x])
        if clamped_w <= 0 or clamped_h <= 0:
             # If the clamped window has no size (view is entirely outside bounds)
             src_window_data = np.zeros((1, 1), dtype=self.src.dtype) # Use a minimal array
             print("[Geo.update_view] Warning: View is outside source data bounds.")
             # We still need to proceed to resize this to fill the display with background
             # Set effective dimensions for pasting calculation
             effective_src_w = 0
             effective_src_h = 0
             paste_x_start = 0
             paste_y_start = 0
        else:
            src_window_data = self.src[clamped_y_start:clamped_y_end, clamped_x_start:clamped_x_end]
            # Calculate where the valid data should start within a theoretical full target window
            paste_x_start = clamped_x_start - src_x_start
            paste_y_start = clamped_y_start - src_y_start
            effective_src_w = clamped_w
            effective_src_h = clamped_h


        # 5. Handle cases where the extracted window needs padding
        # Create a target buffer matching the *calculated* source window size, fill with 0 (or edge color)
        # This buffer will hold the valid data pasted onto it.
        target_window_w = int(np.round(src_window_w))
        target_window_h = int(np.round(src_window_h))

        # Ensure target dimensions are at least 1x1
        target_window_w = max(1, target_window_w)
        target_window_h = max(1, target_window_h)

        # Create the buffer (initially with zeros, could use edge padding later if desired)
        # Use the dtype of the source data
        target_buffer = np.zeros((target_window_h, target_window_w), dtype=self.src.dtype)

        # Calculate where to paste the valid src_window_data into the target_buffer
        paste_x_end = paste_x_start + effective_src_w
        paste_y_end = paste_y_start + effective_src_h

        # Ensure paste coordinates are within the buffer bounds
        paste_x_start_clamped = max(0, paste_x_start)
        paste_y_start_clamped = max(0, paste_y_start)
        paste_x_end_clamped = min(target_window_w, paste_x_end)
        paste_y_end_clamped = min(target_window_h, paste_y_end)

        # Calculate the corresponding slice from src_window_data
        src_slice_x_start = paste_x_start_clamped - paste_x_start
        src_slice_y_start = paste_y_start_clamped - paste_y_start
        src_slice_x_end = src_slice_x_start + (paste_x_end_clamped - paste_x_start_clamped)
        src_slice_y_end = src_slice_y_start + (paste_y_end_clamped - paste_y_start_clamped)


        # Perform the paste if the slices are valid
        if (paste_x_end_clamped > paste_x_start_clamped and
            paste_y_end_clamped > paste_y_start_clamped and
            src_slice_x_end > src_slice_x_start and
            src_slice_y_end > src_slice_y_start and
            src_window_data.size > 0): # Check if src_window_data is not empty

            target_buffer[paste_y_start_clamped:paste_y_end_clamped, paste_x_start_clamped:paste_x_end_clamped] = \
                src_window_data[src_slice_y_start:src_slice_y_end, src_slice_x_start:src_slice_x_end]


        # 6. Resize the potentially padded buffer to display dimensions using OpenCV
        # cv2.resize expects (width, height) for target size
        # Use self.disp_w, self.disp_h which are the display dimensions (self.x, self.y)
        resized_window = cv.resize(target_buffer, (self.disp_w, self.disp_h), interpolation=cv.INTER_LINEAR)

        # 7. Transpose the resized window to match the Taichi field shape (width, height)
        resized_window_transposed = resized_window.T

        # 8. Copy the final transposed and resized window to the Taichi field
        self.data.from_numpy(resized_window_transposed.astype(np.float32)) # Ensure float32 for Taichi field

        self.set_px()

        # Optional: Print debug info
        # print(f"[Geo.update_view] Zoom: {self.zoom:.2f}, Center: ({self.view_x:.1f}, {self.view_y:.1f})")
        # print(f"[Geo.update_view] Src window float: [{src_x_start_f:.1f}:{src_x_start_f+src_window_w:.1f}, {src_y_start_f:.1f}:{src_y_start_f+src_window_h:.1f}]")
        # print(f"[Geo.update_view] Src window int: [{src_x_start}:{src_x_end}, {src_y_start}:{src_y_end}]")
        # print(f"[Geo.update_view] Clamped window: [{clamped_x_start}:{clamped_x_end}, {clamped_y_start}:{clamped_y_end}] Size:({clamped_w}x{clamped_h})")
        # print(f"[Geo.update_view] Target buffer size: ({target_window_w}x{target_window_h}) Paste coords: [{paste_x_start}:{paste_x_end}, {paste_y_start}:{paste_y_end}]")
        # print(f"[Geo.update_view] Resized shape: {resized_window.shape}, Transposed: {resized_window_transposed.shape}, Field shape: {self.data.shape}")

    def pan(self, dx, dy):
        """Pans the view by dx, dy in display pixels."""
        if self.zoom == 0: return # Avoid division by zero

        # Adjust pan distance based on zoom: moving 1 display pixel
        # corresponds to moving 1/zoom source pixels.
        self.view_x += dx / self.zoom
        self.view_y += dy / self.zoom # Assuming positive dy moves view down (screen coords)

        # Optional clamping 
        if self.should_clamp:
            self.clamp()

        self.update_view()

    def zoom_at(self, factor, screen_x, screen_y):
        """Zooms the view by a factor, centered on a screen coordinate."""
        if factor == 1.0: return # No change

        old_zoom = self.zoom
        new_zoom = self.zoom * factor
        # Prevent zooming too far in or out (adjust limits as needed)
        min_zoom = max(self.disp_w / self.src_w, self.disp_h / self.src_h) if self.src_w > 0 and self.src_h > 0 else 0.01 # Cannot zoom out further than fitting the whole image
        max_zoom = 100.0 # Arbitrary max zoom in
        new_zoom = np.clip(new_zoom, min_zoom, max_zoom)

        if new_zoom == old_zoom: return # No change after clipping

        # Calculate the source coordinate (in src pixels) under the cursor *before* zoom
        # screen_x = 0 is left, screen_y = 0 is top
        # view_x, view_y is center of view in src pixels
        # disp_w, disp_h is screen size
        src_x_at_cursor = self.view_x + (screen_x - self.disp_w / 2) / old_zoom
        src_y_at_cursor = self.view_y + (screen_y - self.disp_h / 2) / old_zoom # screen_y=0 is top, source y=0 is top

        # Update zoom first
        self.zoom = new_zoom

        # Calculate the *new* view center needed to keep the src_x/y_at_cursor
        # at the same screen_x/y position *after* zooming.
        self.view_x = src_x_at_cursor - (screen_x - self.disp_w / 2) / self.zoom
        self.view_y = src_y_at_cursor - (screen_y - self.disp_h / 2) / self.zoom

        # Optional clamping (same as pan)
        if self.should_clamp:
            self.clamp()

        self.update_view()

    def clamp(self):
        """Clamps the view center (view_x, view_y) so the viewport stays within source bounds."""
        # Calculate half-width and half-height of the viewport in source coordinates
        margin_w = (self.disp_w / self.zoom) / 2
        margin_h = (self.disp_h / self.zoom) / 2

        # Clamp view_x: min value is margin_w, max value is src_w - margin_w
        # This ensures the viewport [view_x - margin_w, view_x + margin_w] stays within [0, src_w]
        if self.src_w > 0:
             self.view_x = np.clip(self.view_x, margin_w, self.src_w - margin_w)
        else: # Handle case where src_w is 0 or not set
             self.view_x = 0

        # Clamp view_y: min value is margin_h, max value is src_h - margin_h
        # This ensures the viewport [view_y - margin_h, view_y + margin_h] stays within [0, src_h]
        if self.src_h > 0:
             self.view_y = np.clip(self.view_y, margin_h, self.src_h - margin_h)
        else: # Handle case where src_h is 0 or not set
             self.view_y = 0

        # Add small check for invalid dimensions which can cause issues with clip
        if self.src_w <= 2 * margin_w:
            self.view_x = self.src_w / 2
        if self.src_h <= 2 * margin_h:
            self.view_y = self.src_h / 2

    def __call__(self, *args, **kwargs):
        return self.px
