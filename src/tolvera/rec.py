import os
import taichi as ti
from tqdm import tqdm
from .pixels import Pixel

"""
TODO: how to reconcile self.f/self.r with VideoManager.frame_rate?
TODO: how to record in blocks of 512? (1920, 1080, 1024+) hits unsigned int limit
    Possible to use queue method from Audio examples?
TODO: how to prevent/monitor memory overflow?
TODO: Save to npy then convert to mp4?
    vid_np = vid.to_numpy()?
    Separate StateRecorder class?
TODO: bundle with _tolvera? would need to manually add cleanup func
TODO: post-processing
    optional overlay custom text, frames elapsed in corner of video
    custom/override with @ti.kernel/func
"""

@ti.data_oriented
class VideoRecorder:
    def __init__(self, tolvera, **kwargs) -> None:
        """Initialise a video recorder for a Tölvera program.

        Args:
            tolvera (Tolvera): Tölvera instance to record.
            f (int): Number of frames to record. Defaults to 16.
            r (int): Ratio of frames to record (tv.ti.fps/r). Defaults to 4.
            c (int): Frame counter. Defaults to 0.
            w (int): Width of video. Defaults to tv.x.
            h (int): Height of video. Defaults to tv.y.
            output_dir (str): Output directory. Defaults to './output'.
            filename (str): Output filename. Defaults to 'output'.
            automatic_build (bool): Automatically build video. Defaults to True.
            build_mp4 (bool): Build mp4. Defaults to True.
            build_gif (bool): Build gif. Defaults to False.
            clean_frames (bool): Clean frames. Defaults to True.
        """
        self.tv = tolvera
        self.f = kwargs.get('f', 16) # number of frames to record
        self.r = kwargs.get('r', 4) # ratio of frames to record (tv.ti.fps/r)
        self.c = kwargs.get('c', 0) # frame counter
        self.w = kwargs.get('w', self.tv.x) # width
        self.h = kwargs.get('h', self.tv.y) # height
        self.output_dir = kwargs.get('', './output')
        self.filename = kwargs.get('filename', 'output')
        self.automatic_build = kwargs.get('automatic_build', True)
        self.build_mp4 = kwargs.get('build_mp4', True)
        self.build_gif = kwargs.get('build_gif', False)
        self.clean_frames = kwargs.get('clean_frames', True)
        self.video_manager = ti.tools.VideoManager(output_dir=self.output_dir, video_filename=self.filename, width=self.w, height=self.h, framerate=24, automatic_build=False)
        self.vid = Pixel.field(shape=(self.tv.x, self.tv.y, self.f))
        self.px = Pixel.field(shape=(self.tv.x, self.tv.y))
        print(f"[VideoRecorder] {self.w}x{self.h} every {self.r} frames {self.f} times to {self.output_dir}/{self.filename}.")

    @ti.kernel
    def rec(self, i: ti.i32):
        """Record the current frame to the video.

        Args:
            i (ti.i32): Frame index.
        """
        for x, y in ti.ndrange(self.tv.x, self.tv.y):
            self.vid[x, y, i].rgba = self.tv.px.px.rgba[x, y]
    
    @ti.kernel
    def dump(self, i: ti.i32):
        """Dump the current frame to the video.

        Args:
            i (ti.i32): Frame index.
        """
        for x, y in ti.ndrange(self.tv.x, self.tv.y):
            self.px.rgba[x, y] = self.vid[x, y, i].rgba

    def write_frame(self, i: int):
        """Write a frame to the video.

        Args:
            i (int): Frame index.
        """
        self.dump(i)
        self.video_manager.write_frame(self.px.rgba)

    def write(self):
        """Write all frames to the video and build if necessary."""
        print(f"[VideoRecorder] Writing {self.f} frames to {self.filename}")
        for i in tqdm(range(self.f)):
            self.write_frame(i)
        if self.automatic_build:
            print(f"[VideoRecorder] Building {self.filename} with mp4={self.build_mp4} and gif={self.build_gif}")
            self.video_manager.make_video(mp4=self.build_mp4, gif=self.build_gif)
        if self.clean_frames:
            print(f"[VideoRecorder] Cleaning {self.filename} frames")
            self.clean()

    def clean(self):
        """Delete all previous image files in the saved directory.
        
        Fixed version, see https://github.com/taichi-dev/taichi/issues/8533
        """
        for fn in os.listdir(self.video_manager.frame_directory):
            if fn.endswith(".png") and fn in self.video_manager.frame_fns:
                os.remove(f"{self.video_manager.frame_directory}/{fn}")

    def step(self):
        """Record the current frame and increment the frame counter."""
        i = self.tv.ctx.i[None]
        if i % self.r == 0:
            self.rec(self.c)
            self.c += 1
        if i == self.f*self.r:
            self.tv.ctx.stop()

    def __call__(self, *args, **kwds):
        """Record the current frame and increment the frame counter."""
        self.step()
