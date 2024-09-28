import cv2 as cv
import numpy as np
import taichi as ti

from .pixels import Pixels


@ti.data_oriented
class CV:
    def __init__(self, context, **kwargs) -> None:
        self.ctx = context
        self.x, self.y = self.ctx.x, self.ctx.y
        self.px = Pixels(self.ctx, **kwargs)
        self.frame_rgb = np.zeros((self.y, self.x, 3), np.uint8)
        self.ggui_fps_limit = kwargs.get("ggui_fps_limit", 120)
        self.substeps = kwargs.get("substeps", 2)
        self.colormode = kwargs.get("colormode", "rgba")
        self.device = kwargs.get("device", 0)
        self._camera = kwargs.get("camera", False)
        self._video = kwargs.get("video", False)
        self._videofile = kwargs.get("videofile", None)
        self.cc_frame = np.zeros((self.y, self.x, 3), np.uint8)
        self.cc_frame_f32 = np.zeros((self.y, self.x, 3), np.float32)
        self.diff = np.zeros((self.y, self.x, 3), np.uint8)
        self.diff_p = 0.0
        if self._camera:
            self.capture_init(self.device)
        elif self._video:
            self.capture_init(self._videofile)

    def capture_init(self, filename):
        print(f"[{self.ctx.name}] Initialising video capture with '{filename}'...")
        self.camera_capture = cv.VideoCapture(filename)
        self.camera_x = self.camera_capture.get(cv.CAP_PROP_FRAME_WIDTH)
        self.camera_y = self.camera_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.camera_fps = self.camera_capture.get(cv.CAP_PROP_FPS)
        self.camera_substeps = (int)(
            self.ggui_fps_limit / self.camera_fps
        ) * self.substeps
        self.i = 0
        if not self.camera_capture.isOpened():
            print("[tolvera.CV] Cannot open capture")
            exit()

    def capture_read(self):
        ret, self.cc_frame = self.camera_capture.read()
        if ret:
            self.cc_frame_f32 = self.cc_frame.astype(np.float32)
            return self.cc_frame_f32
        elif self._video:
            self.camera_capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            return self.cc_frame_f32
        else:
            print("[tolvera.CV] Cannot read frame")
            exit()

    def threshold(self, img, thresh=127, max=255, threshold_type="binary"):
        if threshold_type == "binary":
            ret, thresh_img = cv.threshold(img, thresh, max, cv.THRESH_BINARY)
        elif threshold_type == "otsu":
            # FIXME: why is this not working?
            """
            > Invalid number of channels in input image:
            >     'VScn::contains(scn)'
            > where
            >     'scn' is 1
            """
            thresh = 0
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, thresh_img = cv.threshold(
                img, thresh, max, cv.THRESH_BINARY + cv.THRESH_OTSU
            )
        self.thresh_img = thresh_img
        return thresh_img

    def find_contours(self, thresh):
        img = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(
            img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        self.contours = contours
        return contours

    def approx_poly_dp(self, contours, epsilon=0.1):
        polygons = [cv.approxPolyDP(c, epsilon, True) for c in contours]
        self.polygons = polygons
        return polygons

    def draw_contours(self, contours, color=(255, 255, 255), thickness=5):
        img = np.zeros((self.y, self.x), np.uint8)
        img = cv.drawContours(img, contours, -1, color, thickness)
        self.contours_img = img
        return img

    def gaussian_blur(self, img, ksize=(25, 25), sigmaX=0):
        img = cv.GaussianBlur(img, ksize, sigmaX)
        return img

    def resize(self, img, dsize=(1920, 1080), interpolation=cv.INTER_LINEAR):
        img = cv.resize(img, dsize, interpolation)
        return img

    def pyr_down(self, img, factor=1):
        for i in range(factor):
            img = cv.pyrDown(img)
        return img

    def pyr_up(self, img, factor=1):
        for i in range(factor):
            img = cv.pyrUp(img)
        return img

    def bgr_to_gray(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

    def gray_to_bgr(self, img):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        return img

    def invert(self, img):
        img = cv.bitwise_not(img)
        return img

    def abs_diff(self, a, b):
        self.diff = cv.absdiff(a, b)
        diff = self.diff#.astype(np.uint8)
        self.diff_p = (np.count_nonzero(diff) * 100) / diff.size
        print('diff', self.diff_p)
        return self.diff_p

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
            frame = self.capture_read()
            # thresh = self.threshold(frame)
            # contours = self.find_contours(thresh)
            # polygons = self.approx_poly_dp(contours)
            # img      = self.draw_contours(contours)
            # img      = self.gaussian_blur(img)
            # img      = self.resize(img, dsize=(int(1920/4), int(1080/4)))
            # self.cv_to_px(self.camera_frame)
            self.cv_to_px(frame)

    def cleanup(self):
        self.camera_capture.release()

    def __call__(self, *args, **kwargs):
        self.process()
        return self.px
