import mediapipe as mp
import taichi as ti
import numpy as np
import enum

from ..osc.update import Updater

class PoseLandmark(enum.IntEnum):
  """The 33 pose landmarks."""
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32

POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
    (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)])

@ti.dataclass
class PoseConnection:
    a: ti.i32
    b: ti.i32

@ti.data_oriented
class MPPose:
    def __init__(self, context, **kwargs) -> None:
        self.ctx = context
        self.kwargs = kwargs
        self.n_conns = len(POSE_CONNECTIONS)

        self.config = {
            'static_image_mode': kwargs.get('static_mode', False),
            'model_complexity': kwargs.get('model_complexity', 1),
            'smooth_landmarks': kwargs.get('smooth_landmarks', True),
            'enable_segmentation': kwargs.get('enable_segmentation', False),
            'smooth_segmentation': kwargs.get('smooth_segmentation', True),
            'min_detection_confidence': kwargs.get('detection_con', .5),
            'min_tracking_confidence': kwargs.get('min_track_con', .5),
        }
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(**self.config)
        self.setup_connections()
        
        self.pose_np = {
            'pxnorm':np.zeros((self.n_conns, 3), np.float32),
            'px':np.zeros((self.n_conns, 2), np.float32),
        }
        self.ctx.s.pose = {
            'state': {
                'pxnorm': (ti.math.vec3, 0.0, 1.0),
                'px': (ti.math.vec2, 0.0, 1.0),
                # 'metres': (ti.math.vec3, 0.0, 1.0), # pose_world_landmarks
            },
            'shape': self.n_conns
        }
        self.detected = ti.field(ti.i32, shape=())

        if self.ctx.iml:
            self.ctx.iml.pose = {
                'type': 'vec2vec',
                'size': ((self.n_conns,2), 1),
            }

        self.updater = Updater(self.detect, kwargs.get('pose_detect_rate', 10))

    def setup_connections(self):
        self.pose_conns = PoseConnection.field(shape=(self.n_conns))
        for i, c in enumerate(POSE_CONNECTIONS):
            self.pose_conns[i] = PoseConnection(c[0], c[1])

    def detect(self, frame=None):
        if frame is None: return
        self.results = self.pose.process(frame)
        if not self.results.pose_landmarks:
            self.ctx.s.pose.fill(0.)
            self.detected[None] = 0
            return

        if self.results.pose_landmarks:
            landmarks = self.results.pose_landmarks.landmark
            for i, lm in enumerate(landmarks):
                pxnorm = np.array([1-lm.x, 1-lm.y, 1-lm.z])
                px = np.array([self.ctx.x*(1-lm.x), self.ctx.y*(1-lm.y)])
                self.pose_np['pxnorm'][i] = pxnorm
                self.pose_np['px'][i] = px
        self.ctx.s.pose.set_from_nddict(self.pose_np)
        
        self.detected[None] = 1

    @ti.kernel
    def draw(self):
        if self.detected[None] == 1:
            self.draw_pose_conns(ti.Vector([1, 1, 1, 1]))
            self.draw_pose_lms(5, ti.Vector([1, 1, 1, 1]))

    @ti.func
    def draw_pose_conns(self, rgba):
        for conn in range(self.n_conns):
            self.draw_conn(conn, rgba)

    @ti.func
    def draw_pose_lms(self, r, rgba):
        for lm in range(self.n_conns):
            self.draw_lm(lm, r, rgba)

    @ti.func
    def draw_conn(self, conn, rgba):
        c = self.pose_conns[conn]
        a = self.ctx.s.pose[c.a].px
        b = self.ctx.s.pose[c.b].px
        ax, ay = ti.cast(a.x, ti.i32), ti.cast(a.y, ti.i32)
        bx, by = ti.cast(b.x, ti.i32), ti.cast(b.y, ti.i32)
        self.px.line(ax, ay, bx, by, rgba)

    @ti.func
    def draw_lm(self, lm: ti.i32, r: ti.i32, rgba: ti.math.vec4):
        px = self.ctx.s.pose[lm].px
        cx = ti.cast(px.x, ti.i32)
        cy = ti.cast(px.y, ti.i32)
        self.px.circle(cx, cy, r, rgba)

    def landmark_name_from_index(self, index):
        return PoseLandmark(index).name

    @ti.kernel
    def get_landmark(self, pose: ti.i32, landmark: ti.i32) -> ti.math.vec2:
        return self.ctx.s.pose[pose, landmark].px

    def __call__(self, frame):
        self.updater(frame)
