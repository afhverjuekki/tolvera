import taichi as ti
import cv2
import numpy as np

from .particles import Particles
from .pixels import Pixels

# TODO: OtsuThreshold
# TODO: test kmeans https://docs.opencv.org/4.6.0/d5/d38/group__core__cluster.html
# TODO: contours to Pixels.polygons? hierarchy?
    # https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
    # https://docs.opencv.org/4.x/da/d32/samples_2cpp_2contours2_8cpp-example.html#a24
# TODO: camera input to physarum
# TODO: move camera read to separate thread?
# TODO: dot detection
# TODO: circle_around
# TODO: update main to remove taichi import?

@ti.data_oriented
class CV:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.x, self.y = self.tv.x, self.tv.y
        self.px = Pixels(self.tv, **kwargs)
        self.px_rgba = ti.Vector.field(4, dtype=ti.f32, shape=(self.x,self.y))
        self.px_rgb = ti.Vector.field(3, dtype=ti.f32, shape=(self.x,self.y))
        self.px_g = ti.field(dtype=ti.f32, shape=(1, self.x, self.y))
        self.frame_rgb = np.zeros((self.y,self.x,3), np.uint8)
        self.frame_g = np.zeros((self.y,self.x), np.uint8)
        self.ggui_fps_limit = kwargs.get('ggui_fps_limit', 120)
        self.substeps = kwargs.get('substeps', 2)
        self.invert = kwargs.get('invert', True)
        self.colormode = kwargs.get('colormode', 'rgba')
        self.device = kwargs.get('device', 0)
        self._camera = kwargs.get('camera', False)
        if self._camera:
            self.camera_init()
    def camera_init(self):
        print(f"[{self.tv.name}] Initialising camera device {self.device}...")
        self.camera_capture = cv2.VideoCapture(self.device)
        self.camera_x = self.camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.camera_y = self.camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.camera_fps = self.camera_capture.get(cv2.CAP_PROP_FPS)
        self.camera_substeps = (int)(self.ggui_fps_limit/self.camera_fps)*self.substeps
        self.i = 0
        if not self.camera_capture.isOpened():
            print("Cannot open camera")
            exit()
    def cv2_camera_read(self):
        ret, self.camera_frame = self.camera_capture.read()
        return self.camera_frame
    def cv2_threshold(self, img, thresh=127, max=255, threshold_type='binary'):
        if threshold_type == 'binary':
            ret, thresh_img = cv2.threshold(img, thresh, max, cv2.THRESH_BINARY)
        elif threshold_type == 'otsu':
            # FIXME: why is this not working?
            '''
            > Invalid number of channels in input image:
            >     'VScn::contains(scn)'
            > where
            >     'scn' is 1
            '''
            thresh=0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh_img = cv2.threshold(img, thresh, max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.thresh_img = thresh_img
        return thresh_img
    def cv2_find_contours(self, thresh):
        img = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        return contours
    def cv2_approx_poly_dp(self, contours, epsilon=0.1):
        polygons = [cv2.approxPolyDP(c, epsilon, True) for c in contours]
        self.polygons = polygons
        return polygons
    def cv2_draw_contours(self, contours, color=(255,255,255), thickness=5):
        img = np.zeros((self.y,self.x), np.uint8)
        img = cv2.drawContours(img, contours, -1, color, thickness)
        self.contours_img = img
        return img
    def cv2_gaussian_blur(self, img, ksize=(25,25), sigmaX=0):
        img = cv2.GaussianBlur(img, ksize, sigmaX)
        return img
    def cv2_resize(self, img, dsize=(1920,1080), interpolation=cv2.INTER_LINEAR):
        img = cv2.resize(img, dsize, interpolation)
        return img
    def cv2_pyr_down(self, img, factor=1):
        for i in range(factor):
            img = cv2.pyrDown(img)
        return img
    def cv2_pyr_up(self, img, factor=1):
        for i in range(factor):
            img = cv2.pyrUp(img)
        return img
    def cv2_bgr_to_gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    def cv2_gray_to_bgr(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    def cv2_invert(self, img):
        img = cv2.bitwise_not(img)
        return img
    @ti.kernel
    def frame2px_rgba(self, f: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        for i, j in ti.ndrange(self.x, self.y):
            _i, _j = self.y-j,self.x-i
            r = f[_i,_j,2]/255
            g = f[_i,_j,1]/255
            b = f[_i,_j,0]/255
            self.px_rgba[i,j] = [r,g,b,1]
    @ti.kernel
    def px_rgb2frame(self, px_rgb: ti.template()):
        # TODO: untested
        for i, j in ti.ndrange(self.x, self.y):
            _i, _j = self.y-j,self.x-i
            r,g,b = px_rgb[i,j]
            self.frame_rgb[_i,_j,2] = r*255
            self.frame_rgb[_i,_j,1] = g*255
            self.frame_rgb[_i,_j,0] = b*255
    @ti.kernel
    def frame2px_g(self, f: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in ti.ndrange(self.x, self.y):
            _i, _j = self.y-j-1,self.x-i-1
            p = f[_i,_j]/255
            self.px_g[0,i,j] = p
            # self.px_g[0,i,self.y-1-j] = p # inverted

    @ti.kernel
    def px_g2frame(self, px_g: ti.template()):
        # TODO: untested
        for i, j in ti.ndrange(self.x, self.y):
            _i, _j = self.y-j-1,self.x-i-1
            p = px_g[0,i,j]
            self.frame_g[_i,_j] = p*255
    @ti.kernel
    def img2px_rgba(self, img: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in ti.ndrange(self.x, self.y):
            _i, _j = self.y-j,self.x-i
            p = img[_i,_j]/255
            self.px_rgba[i,j] = [p,p,p,1]
    def process(self):
        self.i += 1
        if self.i % self.camera_substeps == 0:
            frame    = self.cv2_camera_read()
            thresh   = self.cv2_threshold(frame)
            contours = self.cv2_find_contours(thresh)
            polygons = self.cv2_approx_poly_dp(contours)
            img      = self.cv2_draw_contours(contours)
            # img      = self.cv2_gaussian_blur(img)
            # img      = self.cv2_resize(img, dsize=(int(1920/4), int(1080/4)))
            if self.colormode == 'rgba':
                self.frame2px_rgba(frame)
                # self.px.px.rgba = self.px_rgba
            elif self.colormode == 'rgb':
                self.img2px_rgba(img)
            elif self.colormode == 'g':
                self.frame2px_g(img)
            else:
                assert False, f'colormode error: {self.colormode}'
    def get_frame(self):
        return self.px_rgba
    def get_thresh(self):
        return self.thresh_img
    def get_contours(self):
        return self.contours
    def get_polygons(self):
        return self.polygons
    def px_rgb_to_contours(self, px_rgb):
        frame    = self.px_rgb2frame(px_rgb)
        thresh   = self.threshold(frame)
        contours = self.find_contours(thresh)
        # img      = self.draw_contours(contours)
        # self.frame2px_rgb(img)
        return contours
    def cleanup(self):
        self.camera_capture.release()
    def __call__(self, *args, **kwargs):
        self.process()
        if self.colormode == 'rgba':
            return self.px.px
        elif self.colormode == 'rgb':
            return self.px_rgb#.to_numpy()[0]
        elif self.colormode == 'g':
            return self.px_g#.to_numpy()[0]
        else:
            assert False, f'colormode error: {self.colormode}'
