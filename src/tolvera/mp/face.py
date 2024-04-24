import mediapipe as mp
import taichi as ti
import numpy as np
import enum

from ..osc.update import Updater

class FaceKeyPoint(enum.IntEnum):
  """The enum type of the six face detection key points."""
  RIGHT_EYE = 0
  LEFT_EYE = 1
  NOSE_TIP = 2
  MOUTH_CENTER = 3
  RIGHT_EAR_TRAGION = 4
  LEFT_EAR_TRAGION = 5

@ti.dataclass
class FaceConnection:
    a: ti.i32
    b: ti.i32

@ti.data_oriented
class MPFace:
    def __init__(self, context, **kwargs) -> None:
        self.ctx = context
        self.kwargs = kwargs
        self.n_points = 6
        self.max_faces = kwargs.get('max_faces', 4)

        self.config = {
            'min_detection_confidence': kwargs.get('detection_con', .5),
            'model_selection': kwargs.get('model_selection', 0),
        }

        """
        TODO: add bbox as separate tv.s.faces_bbox?
            format: RELATIVE_BOUNDING_BOX
            relative_bounding_box {
                xmin: 0.482601523
                ymin: 0.402242899
                width: 0.162447035
                height: 0.2887941
            }
        """

        self.faces_np = {
            'pxnorm':np.zeros((self.max_faces, self.n_points, 2), np.float32),
            'px':np.zeros((self.max_faces, self.n_points, 2), np.float32),
        }
        self.ctx.s.faces = {
            'state': {
                'pxnorm': (ti.math.vec2, 0.0, 1.0),
                'px': (ti.math.vec2, 0.0, 1.0),
                # 'metres': (ti.math.vec3, 0.0, 1.0), # face_world_landmarks
            },
            'shape': (self.max_faces, self.n_points)
        }

        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(**self.config)
        self.detected = ti.field(ti.i32, shape=())

        self.updater = Updater(self.detect, kwargs.get('face_detect_rate', 10))

    def detect(self, frame=None):
        if frame is None: return
        self.results = self.face.process(frame)
        if self.results.detections is None:
            self.ctx.s.faces.fill(0.)
            self.detected[None] = -1
            return

        if self.results.detections:
            for i, face in enumerate(self.results.detections):
                for j, lm in enumerate(face.location_data.relative_keypoints):
                    pxnorm = np.array([1-lm.x, 1-lm.y])
                    px = np.array([self.ctx.x*(1-lm.x), self.ctx.y*(1-lm.y)])
                    self.faces_np['pxnorm'][i, j] = pxnorm
                    self.faces_np['px'][i, j] = px
        self.ctx.s.faces.set_from_nddict(self.faces_np)
        
        self.detected[None] = len(self.results.detections)
    
    @ti.kernel
    def draw(self):
        if self.detected[None] > 0:
            self.draw_face_lms(5, ti.Vector([1, 1, 1, 1]))

    @ti.func
    def draw_face_lms(self, r, rgba):
        for i, lm in ti.ndrange(self.detected[None], self.n_conns):
            self.draw_lm(i, lm, r, rgba)

    @ti.func
    def draw_lm(self, face: ti.i32, lm: ti.i32, r: ti.i32, rgba: ti.math.vec4):
        px = self.ctx.s.faces[face, lm].px
        cx = ti.cast(px.x, ti.i32)
        cy = ti.cast(px.y, ti.i32)
        self.px.circle(cx, cy, r, rgba)
    
    def landmark_name_from_index(self, index):
        return FaceKeyPoint(index).name
    
    def landmark_index_from_name(self, name):
        return FaceKeyPoint[name].value

    @ti.kernel
    def get_landmark(self, face: ti.i32, landmark: ti.i32) -> ti.math.vec2:
        return self.ctx.s.faces[landmark].px

    def __call__(self, frame):
        self.updater(frame)
