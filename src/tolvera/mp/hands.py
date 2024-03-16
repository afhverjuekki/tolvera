import mediapipe as mp
import taichi as ti
import numpy as np
import enum

from ..osc.update import Updater

class HandLandmark(enum.IntEnum):
  """The 21 hand landmarks."""
  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))
HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])

@ti.dataclass
class HandConnection:
    a: ti.i32
    b: ti.i32

@ti.data_oriented
class MPHands:
    def __init__(self, context, **kwargs) -> None:
        self.ctx = context
        self.kwargs = kwargs
        self.n_conns = len(HAND_CONNECTIONS)

        self.config = {
            'static_image_mode': kwargs.get('static_mode', False),
            'max_num_hands': kwargs.get('max_hands', 2),
            'model_complexity': kwargs.get('model_complexity', 1),
            'min_detection_confidence': kwargs.get('detection_con', .5),
            'min_tracking_confidence': kwargs.get('min_track_con', .5),
        }
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(**self.config)
        self.setup_connections()
        
        self.hands_np = {
            'pxnorm':np.zeros((self.config['max_num_hands'], self.n_conns, 3), np.float32),
            'px':np.zeros((self.config['max_num_hands'], self.n_conns, 2), np.float32),
        }
        self.ctx.s.hands = {
            'state': {
                'pxnorm': (ti.math.vec3, 0.0, 1.0),
                'px': (ti.math.vec2, 0.0, 1.0),
                # 'metres': (ti.math.vec3, 0.0, 1.0), # multi_hand_world_landmarks
            },
            'shape': (self.config['max_num_hands'], self.n_conns)
        }
        self.handed = ti.field(ti.i32, shape=(self.config['max_num_hands'])) # 0=l, 1=r
        self.handed.fill(-1)

        if self.ctx.iml:
            self.ctx.iml.hands = {
                'type': 'vec2vec',
                'size': ((self.n_conns,2), 1),
            }

        self.updater = Updater(self.detect, kwargs.get('hands_detect_rate', 10))
    
    def setup_connections(self):
        self.palm_conns = HandConnection.field(shape=(len(HAND_PALM_CONNECTIONS)))
        self.thumb_conns = HandConnection.field(shape=(len(HAND_THUMB_CONNECTIONS)))
        self.index_conns = HandConnection.field(shape=(len(HAND_INDEX_FINGER_CONNECTIONS)))
        self.middle_conns = HandConnection.field(shape=(len(HAND_MIDDLE_FINGER_CONNECTIONS)))
        self.ring_conns = HandConnection.field(shape=(len(HAND_RING_FINGER_CONNECTIONS)))
        self.pinky_conns = HandConnection.field(shape=(len(HAND_PINKY_FINGER_CONNECTIONS)))
        self.hand_conns = HandConnection.field(shape=(len(HAND_CONNECTIONS)))
        for p in range(self.palm_conns.shape[0]):
            self.palm_conns[p] = HandConnection(*HAND_PALM_CONNECTIONS[p])
        for p in range(self.thumb_conns.shape[0]):
            self.thumb_conns[p] = HandConnection(*HAND_THUMB_CONNECTIONS[p])
        for p in range(self.index_conns.shape[0]):
            self.index_conns[p] = HandConnection(*HAND_INDEX_FINGER_CONNECTIONS[p])
        for p in range(self.middle_conns.shape[0]):
            self.middle_conns[p] = HandConnection(*HAND_MIDDLE_FINGER_CONNECTIONS[p])
        for p in range(self.ring_conns.shape[0]):
            self.ring_conns[p] = HandConnection(*HAND_RING_FINGER_CONNECTIONS[p])
        for p in range(self.pinky_conns.shape[0]):
            self.pinky_conns[p] = HandConnection(*HAND_PINKY_FINGER_CONNECTIONS[p])
        for i, p in enumerate(HAND_CONNECTIONS):
            self.hand_conns[i] = HandConnection(p[0], p[1])

    def detect(self, frame=None):
        if frame is None: return
        self.results = self.hands.process(frame)
        if not self.results.multi_hand_landmarks:
            self.ctx.s.hands.fill(0.)
            self.handed.fill(-1)
            return

        for i, hand in enumerate(self.results.multi_hand_landmarks):
            for j, lm in enumerate(hand.landmark):
                pxnorm = np.array([1-lm.x, 1-lm.y, 1-lm.z])
                px = np.array([self.ctx.x*(1-lm.x), self.ctx.y*(1-lm.y)])
                self.hands_np['pxnorm'][i, j] = pxnorm
                self.hands_np['px'][i, j] = px
        self.ctx.s.hands.set_from_nddict(self.hands_np)

        for i, handed in enumerate(self.results.multi_handedness):
            label = handed.classification[0].label
            self.handed[i] = 0 if label == 'Left' else 1

    @ti.kernel
    def draw_hands(self):
        if self.handed[0] > -1:
            for hand in range(self.ctx.s.hands.shape[0]):
                self.draw_hand_conns(hand)
                self.draw_hand_lms(hand)
    
    @ti.kernel
    def draw_hand(self, hand: ti.i32):
        if self.handed[hand] > -1:
            self.draw_hand_conns(hand)
            self.draw_hand_lms(hand)

    @ti.func
    def draw_hand_conns(self, hand):
        for conn in range(self.hand_conns.shape[0]):
            self.draw_conn(hand, conn, ti.Vector([1,1,1,1]))
    
    @ti.func
    def draw_palm_conns(self, hand):
        for conn in range(self.palm_conns.shape[0]):
            self.draw_conn(hand, conn)
    
    @ti.func
    def draw_thumb_conns(self, hand):
        for conn in range(self.thumb_conns.shape[0]):
            self.draw_conn(hand, conn)
    
    @ti.func
    def draw_index_conns(self, hand):
        for conn in range(self.index_conns.shape[0]):
            self.draw_conn(hand, conn)
    
    @ti.func
    def draw_middle_conns(self, hand):
        for conn in range(self.middle_conns.shape[0]):
            self.draw_conn(hand, conn)
    
    @ti.func
    def draw_ring_conns(self, hand):
        for conn in range(self.ring_conns.shape[0]):
            self.draw_conn(hand, conn)
    
    @ti.func
    def draw_pinky_conns(self, hand):
        for conn in range(self.pinky_conns.shape[0]):
            self.draw_conn(hand, conn)

    @ti.func
    def draw_conn(self, hand, conn, rgba):
        c = self.hand_conns[conn]
        a = self.ctx.s.hands[hand, c.a].px
        b = self.ctx.s.hands[hand, c.b].px
        ax, ay = ti.cast(a.x, ti.i32), ti.cast(a.y, ti.i32)
        bx, by = ti.cast(b.x, ti.i32), ti.cast(b.y, ti.i32)
        self.px.line(ax, ay, bx, by, rgba)

    @ti.func
    def draw_hand_lms(self, hand):
        r, rgba = 10, ti.Vector([hand,1-hand,0,1])
        for lm in range(self.hand_conns.shape[0]):
            self.draw_lm(hand, lm, r, rgba)
    
    @ti.func
    def draw_palm_lms(self, hand):
        r, rgba = 10, ti.Vector([1,1,1,1])
        for lm in range(self.palm_conns.shape[0]):
            self.draw_lm(hand, lm, r, rgba)
    
    @ti.func
    def draw_thumb_lms(self, hand):
        r, rgba = 10, ti.Vector([1,1,1,1])
        for lm in range(self.thumb_conns.shape[0]):
            self.draw_lm(hand, lm, r, rgba)
    
    @ti.func
    def draw_index_lms(self, hand):
        r, rgba = 10, ti.Vector([1,1,1,1])
        for lm in range(self.index_conns.shape[0]):
            self.draw_lm(hand, lm, r, rgba)
    
    @ti.func
    def draw_middle_lms(self, hand):
        r, rgba = 10, ti.Vector([1,1,1,1])
        for lm in range(self.middle_conns.shape[0]):
            self.draw_lm(hand, lm, r, rgba)
    
    @ti.func
    def draw_ring_lms(self, hand):
        r, rgba = 10, ti.Vector([1,1,1,1])
        for lm in range(self.ring_conns.shape[0]):
            self.draw_lm(hand, lm, r, rgba)
    
    @ti.func
    def draw_pinky_lms(self, hand):
        r, rgba = 10, ti.Vector([1,1,1,1])
        for lm in range(self.pinky_conns.shape[0]):
            self.draw_lm(hand, lm, r, rgba)

    @ti.func
    def draw_lm(self, hand: ti.i32, lm: ti.i32, r: ti.i32, rgba: ti.math.vec4):
        px = self.ctx.s.hands[hand, lm].px
        cx = ti.cast(px.x, ti.i32)
        cy = ti.cast(px.y, ti.i32)
        self.px.circle(cx, cy, r, rgba)
        # self.px.rect(cx, cy, r, r, rgba)

    def train(self, id:int):
        invec = self.ctx.s.hands.attr_to_vec('pxnorm')
        outvec = self.ctx.iml.hands.random_output()
        self.ctx.iml.hands.add(invec, outvec)
        print(f'[tolvera.mp.MPHands.train] Pairs: {len(self.ctx.iml.hands.pairs)}')

    def recognize(self, id):
        # get all pairs that match id
        # iml.index.metric(iml.embed_input([input]), iml.embed_input(pairs))
        pass

    def landmark_name_from_index(self, index):
        return HandLandmark(index).name
    
    def landmark_index_from_name(self, name):
        return HandLandmark[name].value

    @ti.kernel
    def get_landmark(self, hand: ti.i32, landmark: ti.i32) -> ti.math.vec2:
        return self.ctx.s.hands[hand, landmark].px

    def __call__(self, frame):
        self.updater(frame)
