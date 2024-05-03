import mediapipe as mp
import taichi as ti
import numpy as np
import enum

from ..osc.update import Updater

from .face_mesh_connections import *

@ti.dataclass
class FaceMeshConnection:
    a: ti.i32
    b: ti.i32

@ti.data_oriented
class MPFaceMesh:
    def __init__(self, context, **kwargs) -> None:
        self.ctx = context
        self.kwargs = kwargs
        # self.n_conns = len(FACEMESH_TESSELATION)
        self.n_conns = len(FACEMESH_CONTOURS)
        self.n_points = 478

        self.config = {
            'static_image_mode': kwargs.get('static_mode', False),
            'max_num_faces': kwargs.get('max_faces', 1),
            'refine_landmarks': kwargs.get('refine_landmarks', False),
            'min_detection_confidence': kwargs.get('detection_con', .5),
            'min_tracking_confidence': kwargs.get('tracking_con', .5),
            # 'output_face_blendshapes': kwargs.get('output_face_blendshapes', False),
            # 'output_facial_transformation_matrixes': kwargs.get('output_facial_transformation_matrixes', False),
        }
        self.mpFaceMesh = mp.solutions.face_mesh
        self.face_mesh = self.mpFaceMesh.FaceMesh(**self.config)
        self.setup_connections()

        self.face_mesh_np = {
            'pxnorm':np.zeros((self.config['max_num_faces'], self.n_points, 3), np.float32),
            'px':np.zeros((self.config['max_num_faces'], self.n_points, 2), np.float32),
        }
        self.ctx.s.face_mesh = {
            'state': {
                'pxnorm': (ti.math.vec3, 0.0, 1.0),
                'px': (ti.math.vec2, 0.0, 1.0),
                # 'metres': (ti.math.vec3, 0.0, 1.0), # multi_face_world_landmarks
            },
            'shape': (self.config['max_num_faces'], self.n_points)
        }

        self.detected = ti.field(ti.i32, shape=())

        self.updater = Updater(self.detect, kwargs.get('face_detect_rate', 10))

    def setup_connections(self):
        self.lips_conns = FaceMeshConnection.field(shape=(len(FACEMESH_LIPS)))
        self.left_eye_conns = FaceMeshConnection.field(shape=(len(FACEMESH_LEFT_EYE)))
        self.left_iris_conns = FaceMeshConnection.field(shape=(len(FACEMESH_LEFT_IRIS)))
        self.left_eyebrow_conns = FaceMeshConnection.field(shape=(len(FACEMESH_LEFT_EYEBROW)))
        self.right_eye_conns = FaceMeshConnection.field(shape=(len(FACEMESH_RIGHT_EYE)))
        self.right_eyebrow_conns = FaceMeshConnection.field(shape=(len(FACEMESH_RIGHT_EYEBROW)))
        self.right_iris_conns = FaceMeshConnection.field(shape=(len(FACEMESH_RIGHT_IRIS)))
        self.face_oval_conns = FaceMeshConnection.field(shape=(len(FACEMESH_FACE_OVAL)))
        self.nose_conns = FaceMeshConnection.field(shape=(len(FACEMESH_NOSE)))
        self.contours_conns = FaceMeshConnection.field(shape=(len(FACEMESH_CONTOURS)))
        # self.irises_conns = FaceMeshConnection.field(shape=(len(FACEMESH_IRISES)))
        # self.tesselation_conns = FaceMeshConnection.field(shape=(len(FACEMESH_TESSELATION)))
        for i in range(self.lips_conns.shape[0]):
            self.lips_conns[i] = FaceMeshConnection(*FACEMESH_LIPS[i])
        for i in range(self.left_eye_conns.shape[0]):
            self.left_eye_conns[i] = FaceMeshConnection(*FACEMESH_LEFT_EYE[i])
        for i in range(self.left_iris_conns.shape[0]):
            self.left_iris_conns[i] = FaceMeshConnection(*FACEMESH_LEFT_IRIS[i])
        for i in range(self.left_eyebrow_conns.shape[0]):
            self.left_eyebrow_conns[i] = FaceMeshConnection(*FACEMESH_LEFT_EYEBROW[i])
        for i in range(self.right_eye_conns.shape[0]):
            self.right_eye_conns[i] = FaceMeshConnection(*FACEMESH_RIGHT_EYE[i])
        for i in range(self.right_eyebrow_conns.shape[0]):
            self.right_eyebrow_conns[i] = FaceMeshConnection(*FACEMESH_RIGHT_EYEBROW[i])
        for i in range(self.right_iris_conns.shape[0]):
            self.right_iris_conns[i] = FaceMeshConnection(*FACEMESH_RIGHT_IRIS[i])
        for i in range(self.face_oval_conns.shape[0]):
            self.face_oval_conns[i] = FaceMeshConnection(*FACEMESH_FACE_OVAL[i])
        for i in range(self.nose_conns.shape[0]):
            self.nose_conns[i] = FaceMeshConnection(*FACEMESH_NOSE[i])
        for i, j in enumerate(FACEMESH_CONTOURS):
            self.contours_conns[i] = FaceMeshConnection(j[0], j[1])
        # for i, j in enumerate(FACEMESH_IRISES):
        #     self.irises_conns[i] = FaceMeshConnection(j[0], j[1])
        # for i, j in enumerate(FACEMESH_TESSELATION):
        #     self.tesselation_conns[i] = FaceMeshConnection(j[0], j[1])

    def detect(self, frame=None):
        if frame is None: return
        self.results = self.face_mesh.process(frame)
        if not self.results.multi_face_landmarks:
            self.ctx.s.face_mesh.fill(0.)
            self.detected[None] = -1
            return

        for i, face in enumerate(self.results.multi_face_landmarks):
            for j, lm in enumerate(face.landmark):
                pxnorm = np.array([1-lm.x, 1-lm.y, 1-lm.z])
                px = np.array([self.ctx.x*(1-lm.x), self.ctx.y*(1-lm.y)])
                self.face_mesh_np['pxnorm'][i, j] = pxnorm
                self.face_mesh_np['px'][i, j] = px
        self.ctx.s.face_mesh.set_from_nddict(self.face_mesh_np)

        self.detected[None] = len(self.results.multi_face_landmarks)

    @ti.kernel
    def draw(self):
        if self.detected[None] > 0:
            self.draw_face_lms(5, ti.Vector([1, 1, 1, 1]))
            self.draw_face_conns(ti.Vector([1, 1, 1, 1]))
    
    @ti.func
    def draw_face_lms(self, r, rgba):
        for i, lm in ti.ndrange(self.detected[None], self.n_points):
            self.draw_lm(i, lm, r, rgba)

    @ti.func
    def draw_face_conns(self, rgba):
        for i, lm in ti.ndrange(self.detected[None], self.n_points):
            self.draw_conn(i, lm, rgba)

    @ti.func
    def draw_conn(self, face, conn, rgba):
        c = self.contours_conns[conn]
        a = self.ctx.s.face_mesh[face, c.a].px
        b = self.ctx.s.face_mesh[face, c.b].px
        ax, ay = ti.cast(a.x, ti.i32), ti.cast(a.y, ti.i32)
        bx, by = ti.cast(b.x, ti.i32), ti.cast(b.y, ti.i32)
        self.px.line(ax, ay, bx, by, rgba)

    @ti.func
    def draw_lm(self, face: ti.i32, lm: ti.i32, r: ti.i32, rgba: ti.math.vec4):
        px = self.ctx.s.face_mesh[face, lm].px
        cx = ti.cast(px.x, ti.i32)
        cy = ti.cast(px.y, ti.i32)
        self.px.circle(cx, cy, r, rgba)

    def __call__(self, frame):
        self.updater(frame)
