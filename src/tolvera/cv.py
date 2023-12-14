import cv2
import numpy as np
import taichi as ti

from .particles import Particles
from .pixels import Pixels


@ti.data_oriented
class CV:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.x, self.y = self.tv.x, self.tv.y
        self.px = Pixels(self.tv, **kwargs)
        self.frame_rgb = np.zeros((self.y, self.x, 3), np.uint8)
        self.ggui_fps_limit = kwargs.get("ggui_fps_limit", 120)
        self.substeps = kwargs.get("substeps", 1)
        self.invert = kwargs.get("invert", True)
        self.colormode = kwargs.get("colormode", "rgba")
        self.device = kwargs.get("device", 0)
        self._camera = kwargs.get("camera", False)
        if self._camera:
            self.camera_init()

    def camera_init(self):
        print(f"[{self.tv.name}] Initialising camera device {self.device}...")
        self.camera_capture = cv2.VideoCapture(self.device)
        self.camera_x = self.camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.camera_y = self.camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.camera_fps = self.camera_capture.get(cv2.CAP_PROP_FPS)
        self.camera_substeps = (int)(
            self.ggui_fps_limit / self.camera_fps
        ) * self.substeps
        self.i = 0
        if not self.camera_capture.isOpened():
            print("Cannot open camera")
            exit()

    def camera_read(self):
        ret, self.camera_frame = self.camera_capture.read()
        # print(f"[{self.tv.name}] Camera read: {ret}")
        return self.camera_frame

    def threshold(self, img, thresh=127, max=255, threshold_type="binary"):
        if threshold_type == "binary":
            ret, thresh_img = cv2.threshold(img, thresh, max, cv2.THRESH_BINARY)
        elif threshold_type == "otsu":
            # FIXME: why is this not working?
            """
            > Invalid number of channels in input image:
            >     'VScn::contains(scn)'
            > where
            >     'scn' is 1
            """
            thresh = 0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh_img = cv2.threshold(
                img, thresh, max, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        self.thresh_img = thresh_img
        return thresh_img

    def find_contours(self, thresh):
        img = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = contours
        return contours

    def approx_poly_dp(self, contours, epsilon=0.1):
        polygons = [cv2.approxPolyDP(c, epsilon, True) for c in contours]
        self.polygons = polygons
        return polygons

    def draw_contours(self, contours, color=(255, 255, 255), thickness=5):
        img = np.zeros((self.y, self.x), np.uint8)
        img = cv2.drawContours(img, contours, -1, color, thickness)
        self.contours_img = img
        return img

    def gaussian_blur(self, img, ksize=(25, 25), sigmaX=0):
        img = cv2.GaussianBlur(img, ksize, sigmaX)
        return img

    def resize(self, img, dsize=(1920, 1080), interpolation=cv2.INTER_LINEAR):
        img = cv2.resize(img, dsize, interpolation)
        return img

    def pyr_down(self, img, factor=1):
        for i in range(factor):
            img = cv2.pyrDown(img)
        return img

    def pyr_up(self, img, factor=1):
        for i in range(factor):
            img = cv2.pyrUp(img)
        return img

    def bgr_to_gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def gray_to_bgr(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def invert(self, img):
        img = cv2.bitwise_not(img)
        return img

    @ti.kernel
    def cv_to_px(self, f: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        for i, j in ti.ndrange(self.x, self.y):
            _i, _j = self.y - j, self.x - i
            r = f[_i, _j, 2] / 255
            g = f[_i, _j, 1] / 255
            b = f[_i, _j, 0] / 255
            self.px.px.rgba[i, j] = [r, g, b, 1]

    @ti.kernel
    def px_to_cv(self, px_rgb: ti.template()):
        # TODO: untested
        for i, j in ti.ndrange(self.x, self.y):
            _i, _j = self.y - j, self.x - i
            r, g, b = px_rgb[i, j]
            self.frame_rgb[_i, _j, 2] = r * 255
            self.frame_rgb[_i, _j, 1] = g * 255
            self.frame_rgb[_i, _j, 0] = b * 255

    @ti.kernel
    def img_to_px(self, img: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in ti.ndrange(self.x, self.y):
            _i, _j = self.y - j, self.x - i
            p = img[_i, _j] / 255
            self.px.px.rgba[i, j] = [p, p, p, 1]

    def process(self):
        self.i += 1
        if self.i % self.camera_substeps == 0:
            frame = self.camera_read()
            thresh = self.threshold(frame)
            contours = self.find_contours(thresh)
            polygons = self.approx_poly_dp(contours)
            # img      = self.draw_contours(contours)
            # img      = self.gaussian_blur(img)
            # img      = self.resize(img, dsize=(int(1920/4), int(1080/4)))
            self.cv_to_px(self.camera_frame)

    def cleanup(self):
        self.camera_capture.release()

    def __call__(self, *args, **kwargs):
        self.process()
        return self.px.px.rgba
