"""Face detection + tracking (bundled OpenCV Zoo ONNX models) and a velocity-based PTZ auto-framer."""
import collections
import math
import os
import sys
import time

import cv2
import numpy as np

import dshow

TARGET_MODES = ("Largest person", "Closest to center", "Whole group")
LOST_ACTIONS = ("Stay", "Zoom out", "Go home")
OPTICAL_ZOOM_RATIO = 12.0   # zoom factor between zoom min and max (MB Cam12X Pro)
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
TRACK_MODEL = "object_tracking_vittrack_2023sep.onnx"


def resource_path(*parts):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


class Person:
    __slots__ = ("raw", "frame", "head", "kind", "score")

    def __init__(self, raw, frame, head, kind, score):
        self.raw = raw      # detection box (x, y, w, h), normalized 0..1
        self.frame = frame  # composition box (head to chest), normalized
        self.head = head    # head position (x, y), normalized
        self.kind = kind
        self.score = score


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _inside(box, x, y):
    return box[0] <= x <= box[0] + box[2] and box[1] <= y <= box[1] + box[3]


def _iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _face_frame(x, y, w, h):
    """Composition box for a face: head to chest."""
    return x + w / 2 - 1.5 * w, y - 0.5 * h, 3.0 * w, 4.5 * h


def _face_person(x, y, w, h, score):
    return Person(raw=(x, y, w, h), frame=_face_frame(x, y, w, h), head=(x + w / 2, y + h / 2), kind="face",
                  score=score)


class OneEuroFilter:
    """One Euro filter (Casiez, Roussel, Vogel, CHI 2012): strong smoothing while the value is
    steady, little lag while it changes quickly."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        self.dx += self._alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        self.x += self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt) * (x - self.x)
        return self.x


class ConstantVelocityKalman:
    """1-D Kalman filter with a constant-velocity motion model and variable time steps.

    Unlike low-pass smoothing it has no lag for steady motion, and it can predict ahead: between
    measurements (video latency) and through short gaps (coasting), where the velocity is faded
    out so the prediction doesn't run away."""

    def __init__(self, accel_noise, meas_noise, coast_after=0.2, coast_tau=0.5):
        self.q = accel_noise
        self.r = meas_noise ** 2
        self.coast_after, self.coast_tau = coast_after, coast_tau
        self.p = self.v = self.t = self.t_meas = None
        self.P = None   # covariance [[pp, pv], [vp, vv]]

    def update(self, z, t):
        if self.p is None:
            self.p, self.v, self.t, self.t_meas = z, 0.0, t, t
            self.P = [[self.r, 0.0], [0.0, 1.0]]
            return
        dt = t - self.t
        if dt > 0:
            (pp, pv), (vp, vv) = self.P
            q = self.q
            self.p += self.v * dt
            self.P = [[pp + dt * (pv + vp) + dt * dt * vv + q * dt ** 3 / 3, pv + dt * vv + q * dt * dt / 2],
                      [vp + dt * vv + q * dt * dt / 2, vv + q * dt]]
            self.t = t
        (pp, pv), (vp, vv) = self.P
        s = pp + self.r
        k0, k1 = pp / s, vp / s
        y = z - self.p
        self.p += k0 * y
        self.v += k1 * y
        self.P = [[(1 - k0) * pp, (1 - k0) * pv], [vp - k1 * pp, vv - k1 * pv]]
        self.t_meas = t

    def peek(self, t):
        """Predicted value at time t (without changing the filter)."""
        if self.p is None or t <= self.t:
            return self.p
        age0, age1 = self.t - self.t_meas, t - self.t_meas
        straight = max(0.0, min(age1, self.coast_after) - age0)
        fading = max(0.0, age1 - max(age0, self.coast_after))
        return self.p + self.v * straight + self.v * self.coast_tau * (1 - math.exp(-fading / self.coast_tau))


class FaceTracker:
    """YuNet finds faces a few times per second at most; OpenCV's TrackerVit follows the chosen
    face on every frame. Benchmarked on a 752-frame clip from the MB Cam12X Pro: ~4 ms CPU per
    frame (vs ~107 ms for YuNet at full resolution + NanoDet every frame), a box on every frame
    and no size/position jumps."""

    backend = "YuNet face detection + OpenCV VitTrack (ONNX)"
    DETECT_WIDTH = 640
    FAR_WIDTH = 1280
    RECHECK_S = 1.0        # confirm/correct the tracked face this often
    SEARCH_S = 0.25        # look for faces this often while nobody is tracked
    GROUP_S = 0.33         # re-detect everyone this often when framing the whole group
    FAR_SEARCH_S = 1.0     # full-resolution search for small (far away) faces while nobody is tracked
    UNCONFIRMED_S = 4.0    # drop the track when no detection has confirmed it for this long
    MIN_SCORE = 0.3

    def __init__(self):
        self.yunet = cv2.FaceDetectorYN.create(resource_path("models", FACE_MODEL), "", (640, 360), 0.7, 0.3, 50)
        self._vit_params = cv2.TrackerVit_Params()
        self._vit_params.net = resource_path("models", TRACK_MODEL)
        self.mode = TARGET_MODES[0]
        self.hint = None       # predicted head position while the framer coasts: re-acquire the face there
        self.faces = []        # latest detections, normalized (x, y, w, h, score)
        self.target = None     # tracked face, normalized (x, y, w, h)
        self._tracker = None
        self._pick = None
        self._last_detect = self._last_far = self._confirmed = 0.0

    def pick(self, x, y):
        """Follow the face nearest to a clicked point (normalized). True if a face is close enough."""
        for fx, fy, fw, fh, _ in self.faces:
            if _inside(_face_frame(fx, fy, fw, fh), x, y):
                self._pick = (x, y)
                return True
        return False

    def reset(self):
        self._tracker = None
        self.target = None

    def process(self, frame, now):
        """Returns (faces, target) for this frame."""
        H, W = frame.shape[:2]
        group = self.mode == "Whole group"
        if group and self._tracker is not None:
            self.reset()

        if self._tracker is not None:
            ok, (x, y, w, h) = self._tracker.update(frame)
            if ok and self._tracker.getTrackingScore() >= self.MIN_SCORE and now - self._confirmed < self.UNCONFIRMED_S:
                self.target = (x / W, y / H, w / W, h / H)
            else:
                self.reset()

        tracking = self._tracker is not None
        interval = self.RECHECK_S if tracking else (self.GROUP_S if group and self.faces else self.SEARCH_S)
        if self._pick is not None or now - self._last_detect >= interval:
            self._last_detect = now
            small_target = tracking and self.target[3] * self.DETECT_WIDTH * H / W < 20
            faces = self._detect(frame, self.FAR_WIDTH if small_target else self.DETECT_WIDTH)
            if not faces and not tracking and now - self._last_far >= self.FAR_SEARCH_S:
                self._last_far = now
                faces = self._detect(frame, self.FAR_WIDTH)
            self.faces = faces
            if not group:
                self._choose_target(frame, faces, now)
        return self.faces, (None if group else self.target)

    def _detect(self, frame, width):
        H, W = frame.shape[:2]
        width = min(width, W)
        img = frame if width == W else cv2.resize(frame, (width, int(H * width / W)), interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]
        self.yunet.setInputSize((iw, ih))
        _, found = self.yunet.detect(img)
        return [(f[0] / iw, f[1] / ih, f[2] / iw, f[3] / ih, float(f[14])) for f in (found if found is not None else [])]

    def _choose_target(self, frame, faces, now):
        if self._pick is not None:
            pick, self._pick = self._pick, None
            choice = min(faces, key=lambda f: _dist((f[0] + f[2] / 2, f[1] + f[3] / 2), pick), default=None)
        elif self.target is not None:
            best = max(faces, key=lambda f: _iou(f[:4], self.target), default=None)
            if best is None or _iou(best[:4], self.target) < 0.2:
                return   # face not visible right now (turned away?), keep tracking until UNCONFIRMED_S
            choice = best
        elif not faces:
            return
        elif self.hint is not None and min(_dist((f[0] + f[2] / 2, f[1] + f[3] / 2), self.hint) for f in faces) < 0.25:
            choice = min(faces, key=lambda f: _dist((f[0] + f[2] / 2, f[1] + f[3] / 2), self.hint))
        elif self.mode == "Closest to center":
            choice = min(faces, key=lambda f: _dist((f[0] + f[2] / 2, f[1] + f[3] / 2), (0.5, 0.4)))
        else:
            choice = max(faces, key=lambda f: f[2] * f[3])
        if choice is None:
            return
        H, W = frame.shape[:2]
        x, y, w, h = choice[:4]
        box = (int(x * W), int(y * H), max(8, int(w * W)), max(8, int(h * H)))
        self._tracker = cv2.TrackerVit_create(self._vit_params)   # re-initializing corrects size drift
        self._tracker.init(frame, box)
        self.target = tuple(choice[:4])
        self._confirmed = now


def estimate_fov_units(control, zoom_ratio=OPTICAL_ZOOM_RATIO, wide_share=0.35):
    """Rough horizontal field of view in pan units at the current zoom (for click-to-center)."""
    pan = control.ranges.get(dshow.PAN)
    if pan is None:
        return 0.0
    zn = 0.0
    zoom = control.ranges.get(dshow.ZOOM)
    if zoom is not None and control.get(dshow.ZOOM) is not None:
        zn = (control.get(dshow.ZOOM) - zoom.min) / zoom.span
    return pan.span * wide_share / (1 + (zoom_ratio - 1) * zn)


class FramerSettings:
    def __init__(self):
        self.target = TARGET_MODES[0]
        self.framing = 0.55       # share of frame height the composition box should fill
        self.deadzone = 0.08
        self.head_height = 0.38   # desired vertical head position
        self.auto_zoom = True
        self.zoom_aggr = 0.35     # 0 = calm (waits long, small steps) … 1 = eager
        self.max_zoom = 0.7       # share of the zoom range auto zoom may use
        self.speed = 0.6          # relative motor speed (ignored by cameras with a fixed rate)
        self.on_lost = LOST_ACTIONS[0]
        self.lost_timeout = 5.0
        self.face_only = False    # frame just the face (tight close-up) instead of head to chest
        self.pan_right = 1       # absolute-value direction that turns the camera right
        self.tilt_up = 1          # absolute-value direction that tilts the camera up


class AutoFramer:
    """Steers the camera toward the tracked face (or the whole group) with start/stop motion
    commands. Inputs are One-Euro filtered; moves stop early based on the target's image
    velocity and the video latency, so fixed-speed PTZ motors don't overshoot."""

    LEASE = 0.45
    CONFIRM = 3
    PREDICTIVE = True      # Kalman prediction + camera-motion compensation in "Face only"
    COAST_S = 1.0          # keep following the predicted path this long when the face is lost
    LEAD_S = 0.08          # steer toward where the face will be when a command takes effect
    PREDICTIVE_SETTLE = 0.2

    def __init__(self, settings=None):
        self.s = settings or FramerSettings()
        self.reset()

    def reset(self):
        self.box = self.head = None
        self.hits = 0
        self.last_seen = 0.0
        self.lost_handled = True
        self.paused_until = 0.0
        self._filters = None
        self._dir = {}
        self._err = {}
        self._vel = {}
        self._pulses = {}
        self._kf = None             # Kalman filters: world x, world head y, log size
        self._raw = None            # last measured (x, head y) in the picture
        self._shape = (1.0, 0.5)    # box width/height ratio, head offset below the box top (in box heights)
        self._cam_off = {dshow.PAN: 0.0, dshow.TILT: 0.0}   # picture shift caused by our own camera moves
        self._ego_hist = collections.deque(maxlen=64)
        self._ego_t = None
        self.predicting = False
        self._zoom_start = None
        self._zoom_blocked = (0, 0.0)
        self._ez = None
        self._zoom_calm_since = None
        self._zoom_in_after = 0.0
        self._pulse_end = 0.0
        self.zoom_state = "idle"

    def pause(self, seconds=3.0):
        """Manual control takes over; the manual command replaces any tracking motion."""
        self.paused_until = time.monotonic() + seconds
        self._dir.clear()

    def release(self, control):
        for axis, d in self._dir.items():
            if d and control is not None:
                control.move(axis, 0)
        self._dir.clear()

    def update(self, target, faces, frame_ts, control):
        """target: tracked face (x, y, w, h) or None; faces: latest detections (x, y, w, h, score)."""
        now = time.monotonic()
        observed = self._observe(target, faces)
        predictive = self.PREDICTIVE and self.s.face_only and self.s.target != "Whole group"
        if predictive:
            self._advance_ego(now)
        if now < self.paused_until:
            self._kf = None   # manual moves aren't in the camera-motion model: start the prediction fresh

        if observed is None:
            if (predictive and self._kf is not None and self.hits >= self.CONFIRM and now >= self.paused_until
                    and now - self.last_seen < self.COAST_S):
                self.predicting = True
                self._raw = None
                self._predict_box(now)
                self._steer(control, now, now, coasting=True)
                return "Following, predicting movement"
            self.predicting = False
            self.release(control)
            if self.box is None:
                return "Looking for people…"
            if now - self.last_seen > self.s.lost_timeout:
                if not self.lost_handled:
                    self._on_lost(control)
                    self.lost_handled = True
                self.box = self.head = None
                self._filters = self._kf = None
                self.hits = 0
                return "Looking for people…"
            return "Lost sight, waiting a moment"

        self.predicting = False
        if predictive:
            self._kalman_update(observed, frame_ts, now)
        else:
            self._kf = None
            self._smooth(observed, frame_ts)
        self.hits += 1
        self.last_seen = now
        self.lost_handled = False
        if self.hits < self.CONFIRM:
            return "Found someone…"
        if now < self.paused_until:
            self._dir.clear()
            return "Paused while you move the camera"
        self._steer(control, frame_ts, now)
        if self.zoom_state in ("zooming in", "zooming out", "about to zoom in"):
            return f"Following, {self.zoom_state}"
        if self.zoom_state == "waiting: person near the edge":
            return "Following. Can't zoom closer because the person is near the edge of the picture"
        if self.zoom_state == "at max zoom":
            return "Following, zoomed in as far as allowed"
        if self._dir.get(dshow.PAN) or self._dir.get(dshow.TILT):
            return "Following, turning"
        return "Following"

    def _observe(self, target, faces):
        """(center x, box top, box width, box height, head y) of what should be framed."""
        if self.s.target == "Whole group":
            if not faces:
                return None
            boxes = [_face_frame(*f[:4]) for f in faces]
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[0] + b[2] for b in boxes)
            y2 = max(b[1] + b[3] for b in boxes)
            head_y = min(f[1] + f[3] / 2 for f in faces)
            return (x1 + x2) / 2, y1, x2 - x1, y2 - y1, head_y
        if target is None:
            return None
        if self.s.face_only:   # close-up: the face plus a small margin
            fx, fy, fw, fh = target
            return fx + fw / 2, fy - 0.25 * fh, 1.6 * fw, 1.5 * fh, fy + fh / 2
        x, y, w, h = _face_frame(*target)
        return x + w / 2, y, w, h, target[1] + target[3] / 2

    def _smooth(self, observed, t):
        cx, top, w, h, head_y = observed
        if self._filters is not None and self.head is not None and _dist((cx, head_y), self.head) > 0.25:
            self._filters = None   # a different person: jump instead of gliding across the picture
        if self._filters is None:
            # Tuned on tracker output from the MB Cam12X Pro: position lag ~25 ms, size lag ~70 ms,
            # with most of the jitter reduction of much slower settings.
            self._filters = [OneEuroFilter(3.0, 8.0), OneEuroFilter(3.0, 8.0), OneEuroFilter(1.5, 2.0),
                             OneEuroFilter(1.5, 2.0), OneEuroFilter(3.0, 8.0)]
            self._err.clear()
            self._vel.clear()
        fx, ft, fw, fh, fy = self._filters
        cx, top, head_y = fx(cx, t), ft(top, t), fy(head_y, t)
        w, h = math.exp(fw(math.log(max(1e-4, w)), t)), math.exp(fh(math.log(max(1e-4, h)), t))
        self.box = (cx - w / 2, top, w, h)
        self.head = (cx, head_y)

    # ---------------------------------------------------------------- prediction ("Face only")
    def _pulse_speed(self, axis):
        st = self._pulses.get(axis)
        return (st and st["speed"]) or 1.0

    def _advance_ego(self, now):
        """Integrate how far our own bursts have shifted the picture, so the Kalman filters see the
        subject's motion rather than the camera's."""
        if self._ego_t is not None:
            s = self.s
            for axis, direction in ((dshow.PAN, s.pan_right), (dshow.TILT, -s.tilt_up)):
                moving = self._dir.get(axis, 0)
                st = self._pulses.get(axis)
                if moving and st:
                    active = max(0.0, min(now, st["end"]) - self._ego_t)
                    # the picture error changes by -sign(error) * speed while a correcting burst runs
                    self._cam_off[axis] += moving * direction * self._pulse_speed(axis) * active
        self._ego_t = now
        self._ego_hist.append((now, self._cam_off[dshow.PAN], self._cam_off[dshow.TILT]))

    def _ego_at(self, t):
        best = None
        for entry in self._ego_hist:
            if entry[0] <= t:
                best = entry
            else:
                break
        entry = best or (self._ego_hist[0] if self._ego_hist else (t, 0.0, 0.0))
        return entry[1], entry[2]

    def _kalman_update(self, observed, frame_ts, now):
        cx, top, w, h, head_y = observed
        off_x, off_y = self._ego_at(frame_ts)
        wx, wy, ls = cx + off_x, head_y + off_y, math.log(max(1e-4, h))
        if self._kf is not None:
            px, py = self._kf[0].peek(frame_ts), self._kf[1].peek(frame_ts)
            if _dist((wx, wy), (px, py)) > 0.3:
                self._kf = None   # a different face: start over instead of gliding across
        if self._kf is None:
            # q/r picked from a grid on tracker noise measured on the MB Cam12X Pro (std 0.13 % of the
            # frame): 2-3x lower tracking error than the One Euro smoothing for walking, stop-and-go and
            # swaying, and 0.5 s gap predictions 55-75 % closer than holding the last position.
            self._kf = [ConstantVelocityKalman(0.05, 0.006), ConstantVelocityKalman(0.05, 0.006),
                        ConstantVelocityKalman(0.05, 0.015)]
            self._shape = (w / h, (head_y - top) / h)
            for st in self._pulses.values():   # keep the learned image speed, drop stale burst state
                st["measure"] = None
        for kf, z in zip(self._kf, (wx, wy, ls)):
            kf.update(z, frame_ts)
        ratio, head_off = self._shape
        self._shape = (0.8 * ratio + 0.2 * w / h, 0.8 * head_off + 0.2 * (head_y - top) / h)
        self._raw = (cx, head_y)
        self._predict_box(now)

    def _predict_box(self, now):
        t = now + self.LEAD_S
        kx, ky, ks = self._kf
        x = kx.peek(t) - self._cam_off[dshow.PAN]
        y = ky.peek(t) - self._cam_off[dshow.TILT]
        h = math.exp(ks.peek(ks.t))   # size: filtered, not extrapolated
        ratio, head_off = self._shape
        w = h * ratio
        self.box = (x - w / 2, y - head_off * h, w, h)
        self.head = (x, y)

    def _velocity(self, axis, e, ts):
        prev = self._err.get(axis)
        self._err[axis] = (e, ts)
        if prev is not None and ts > prev[1] + 1e-3:
            inst = max(-3.0, min(3.0, (e - prev[0]) / (ts - prev[1])))
            self._vel[axis] = 0.5 * inst + 0.5 * self._vel.get(axis, 0.0)
        return self._vel.get(axis, 0.0)

    def _command(self, control, axis, want):
        if want and getattr(control, "at_limit", lambda p, s: False)(axis, want) and not self._dir.get(axis):
            want = 0
        if want:
            control.move(axis, want, speed=self.s.speed, lease=self.LEASE)
            self._dir[axis] = want
        elif self._dir.get(axis):
            control.move(axis, 0)
            self._dir[axis] = 0

    def _steer(self, control, frame_ts, now, coasting=False):
        s = self.s
        (bx, by, bw, bh), (hx, hy) = self.box, self.head
        ex = bx + bw / 2 - 0.5
        ey = hy - s.head_height
        latency = min(0.6, max(0.0, now - frame_ts)) + 0.1

        axes = []
        if control.supported(dshow.PAN):
            axes.append((dshow.PAN, ex, s.pan_right, s.deadzone))
        if control.supported(dshow.TILT):
            axes.append((dshow.TILT, ey, -s.tilt_up, s.deadzone * 1.25))
        zoom = control.ranges.get(dshow.ZOOM)
        zoomed_in = zoom is not None and control.get(dshow.ZOOM) is not None and (
            control.get(dshow.ZOOM) - zoom.min) / zoom.span > 0.5
        for axis, e, direction, dz in axes:
            self._velocity(axis, e, frame_ts)
            if s.face_only or zoomed_in:
                if self._kf is not None:   # predictive: act on the predicted error, learn from measurements
                    raw = None
                    if self._raw is not None:
                        raw = self._raw[0] - 0.5 if axis == dshow.PAN else self._raw[1] - s.head_height
                    subject_v = self._kf[0 if axis == dshow.PAN else 1].v
                    self._pulse(control, axis, e, raw, frame_ts, direction, dz, now, subject_v,
                                self.PREDICTIVE_SETTLE, 0.25 if coasting else 0.5)
                else:
                    self._pulse(control, axis, e, e, frame_ts, direction, dz, now)
                continue
            e_pred = e + self._vel.get(axis, 0.0) * latency
            current = self._dir.get(axis, 0)
            if current == 0:
                want = direction * (1 if e > 0 else -1) if abs(e) > dz and abs(e_pred) > dz * 0.5 else 0
            else:
                still_needed = abs(e_pred) > dz * 0.3 and direction * (1 if e_pred > 0 else -1) == current
                want = current if still_needed else 0
            self._command(control, axis, want)

        if coasting:   # no reliable size while predicting: hold the zoom
            self._command(control, dshow.ZOOM, 0)
            self.zoom_state = "idle"
        elif s.auto_zoom and control.supported(dshow.ZOOM):
            self._steer_zoom(control, ex, ey, now)
        else:
            self.zoom_state = "idle"

    PULSE_SETTLE = 0.45    # s to wait after a burst so the picture (and video latency) catches up
    PULSE_GAIN = 0.7       # correct only part of the error per burst: undershoot instead of overshoot

    def _pulse(self, control, axis, e, e_raw, frame_ts, direction, dz, now, subject_v=0.0,
               settle=None, max_burst=0.5):
        """Zoomed-in steering: short timed bursts sized from the learned image speed of this axis,
        each followed by a settle pause. A fixed-speed motor moves a tight shot too fast for
        continuous start/stop control.

        e: error to act on (predicted in "Face only"); e_raw: last measured error, used to learn the
        image speed (None while coasting); subject_v: the subject's own velocity (error units/s)."""
        settle = self.PULSE_SETTLE if settle is None else settle
        st = self._pulses.setdefault(axis, {"end": 0.0, "next": 0.0, "speed": None, "measure": None})
        current = self._dir.get(axis, 0)
        want_sign = direction * (1 if e > 0 else -1)
        if current:
            if now >= st["end"] or abs(e) < dz * 0.3 or want_sign != current:
                control.move(axis, 0)
                self._dir[axis] = 0
                m = st["measure"]
                if m is not None and now < m["end"]:   # stopped early: learn from the real duration
                    m["seconds"] = max(0.03, now - m["start"])
                    m["end"] = now
                st["end"] = min(st["end"], now)
                st["next"] = now + settle
            return
        m = st["measure"]
        if m is not None and e_raw is not None and frame_ts >= m["end"] + 0.05:
            # learn how fast the picture moves, from the first frame captured after the burst ended
            st["measure"] = None
            moved = (m["e"] + m["v"] * (frame_ts - m["ts"]) - e_raw) * (1 if m["e"] > 0 else -1)
            if moved > 0.01:
                observed = min(5.0, max(0.1, moved / m["seconds"]))
                st["speed"] = observed if st["speed"] is None else 0.5 * st["speed"] + 0.5 * observed
        if now < st["next"] or abs(e) <= dz:
            return
        if getattr(control, "at_limit", lambda p, s: False)(axis, want_sign):
            return
        aim = e + subject_v * 0.15   # the subject keeps moving while the burst runs
        if (aim > 0) != (e > 0):
            return                   # the subject is already heading back to the center on its own
        seconds = min(max_burst, max(0.06, self.PULSE_GAIN * abs(aim) / (st["speed"] or 1.0)))
        control.move(axis, want_sign, speed=self.s.speed, lease=seconds + 0.05)
        self._dir[axis] = want_sign
        st["end"] = now + seconds
        if e_raw is not None:
            st["measure"] = {"e": e_raw, "start": now, "seconds": seconds, "end": now + seconds, "ts": frame_ts,
                             "v": subject_v}

    def _steer_zoom(self, control, ex, ey, now):
        """Zoom out promptly when the shot is too tight; zoom in in steps once the shot has been
        steady for a moment (tuned by zoom_aggr)."""
        s = self.s
        a = min(1.0, max(0.0, s.zoom_aggr))

        def lerp(calm, eager):
            return calm + (eager - calm) * a

        bx, by, bw, bh = self.box
        ez_now = math.log(max(1e-3, max(bh / max(0.05, s.framing), bw / 0.9)))
        self._ez = ez_now   # inputs are already One-Euro filtered
        cut = bx < 0.01 or bx + bw > 0.99 or by < 0.0
        current = self._dir.get(dshow.ZOOM, 0)

        if current < 0 or ez_now > 0.18 or (cut and ez_now > -0.1):
            keep_out = ez_now > 0.05 or (cut and ez_now > -0.1)
            if keep_out and self._zoom_progress(-1, abs(ez_now), now):
                if current > 0:
                    control.move(dshow.ZOOM, 0)
                self._command(control, dshow.ZOOM, -1)
                self._zoom_calm_since = None
                self._zoom_in_after = now + lerp(3.0, 1.0)   # no zoom-in right after zooming out
                self.zoom_state = "zooming out"
                return
            self._command(control, dshow.ZOOM, 0)
            self.zoom_state = "at widest" if keep_out else "idle"
            return

        zoom = control.ranges[dshow.ZOOM]
        value = control.get(dshow.ZOOM)
        below_max = value is None or value < zoom.min + s.max_zoom * zoom.span
        # Zooming magnifies around the image center: only require that the person stays well
        # inside the picture after the next step (a centered subject is impossible at a tilt limit).
        hx, hy = self.head
        grow = 1.25

        def scaled(v):
            return 0.5 + (v - 0.5) * grow
        fits = 0.1 < scaled(hx) < 0.9 and 0.06 < scaled(hy) < 0.85 and scaled(by) > -0.12
        steady = abs(self._vel.get(dshow.PAN, 0.0)) < 0.35 and abs(self._vel.get(dshow.TILT, 0.0)) < 0.35

        if current > 0:  # a zoom-in step is running
            if now >= self._pulse_end or not fits or ez_now > -0.05 or not below_max:
                control.move(dshow.ZOOM, 0)
                self._dir[dshow.ZOOM] = 0
                self._zoom_in_after = now + lerp(0.6, 0.25)
                self.zoom_state = "settling"
            else:
                self.zoom_state = "zooming in"
            return

        if self._ez >= -lerp(0.2, 0.1):
            reason = "shot fits"
        elif not below_max:
            reason = "at max zoom"
        elif not fits:
            reason = "waiting: person near the edge"
        elif not steady or self._dir.get(dshow.PAN) or self._dir.get(dshow.TILT):
            reason = "waiting: camera or person moving"
        else:
            reason = None
        if reason is not None:
            self._zoom_calm_since = None
            self.zoom_state = "idle" if reason == "shot fits" else reason
            return
        if self._zoom_calm_since is None:
            self._zoom_calm_since = now
        if now < self._zoom_in_after or now - self._zoom_calm_since < lerp(1.2, 0.3):
            self.zoom_state = "about to zoom in"
            return
        # step length scales with how far off the framing is (assumes ~10 % of the range per second)
        needed = -self._ez / math.log(OPTICAL_ZOOM_RATIO) * zoom.span
        step = min(lerp(1.5, 3.0), max(0.25, needed / (0.1 * zoom.span) * lerp(0.7, 0.9)))
        control.move(dshow.ZOOM, 1, speed=s.speed, lease=step + 0.05)
        self._dir[dshow.ZOOM] = 1
        self._pulse_end = now + step
        self.zoom_state = "zooming in"

    def _zoom_progress(self, want, err, now):
        """Back off when zooming doesn't change the framing (zoom already at its limit)."""
        blocked_dir, until = self._zoom_blocked
        if want and want == blocked_dir and now < until:
            return 0
        if not want:
            self._zoom_start = None
            return 0
        if self._zoom_start is None or self._zoom_start[0] != want:
            self._zoom_start = (want, now, err)
        elif now - self._zoom_start[1] > 1.5:
            if self._zoom_start[2] - err < 0.05:
                self._zoom_blocked, self._zoom_start = (want, now + 4.0), None
                return 0
            self._zoom_start = (want, now, err)
        return want

    def _on_lost(self, control):
        if self.s.on_lost == "Zoom out" and control.supported(dshow.ZOOM):
            control.set(dshow.ZOOM, control.ranges[dshow.ZOOM].min)
        elif self.s.on_lost == "Go home":
            for prop in (dshow.PAN, dshow.TILT, dshow.ZOOM):
                if control.supported(prop):
                    control.set(prop, control.ranges[prop].default)


def calibrate(control, video, progress=lambda msg: None):
    """Find which absolute direction turns the camera right/up by pulsing each axis and
    measuring the global image shift with phase correlation. Returns {"pan_right", "tilt_up"}."""
    control.stop_all()
    size = (320, 180)
    window = cv2.createHanningWindow(size, cv2.CV_32F)

    def frame_after(t):
        deadline = time.monotonic() + 3.0
        while video.frame is None or video.ts <= t:
            if time.monotonic() > deadline:
                raise RuntimeError("no video frames, calibration needs the live picture")
            time.sleep(0.02)
        return np.float32(cv2.cvtColor(cv2.resize(video.frame, size, interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2GRAY))

    def pulse(prop, direction, seconds):
        control.move(prop, direction, speed=0.6)
        time.sleep(seconds)
        control.move(prop, 0)

    result = {}
    for prop, key, name in ((dshow.PAN, "pan_right", "pan"), (dshow.TILT, "tilt_up", "tilt")):
        if not control.supported(prop):
            continue
        rng = control.ranges[prop]
        current = control.get(prop)
        d = 1 if current is None or current < (rng.min + rng.max) / 2 else -1   # move away from the nearer limit
        progress(f"Checking which way the camera turns ({name})…")
        pulse(prop, d, 0.25)   # warm-up move lets the controller learn the relative sign
        time.sleep(1.2)
        pulse(prop, -d, 0.25)
        time.sleep(1.2)
        for seconds in (0.35, 0.7, 1.2):
            before = frame_after(time.monotonic())
            pulse(prop, d, seconds)
            time.sleep(1.0)
            after = frame_after(time.monotonic())
            (dx, dy), response = cv2.phaseCorrelate(before, after, window)
            pulse(prop, -d, seconds)
            time.sleep(1.0)
            shift = dx if prop == dshow.PAN else dy
            if abs(shift) >= 4 and response > 0.03:
                break
        else:
            raise RuntimeError(f"{name}: no image motion measured (camera at its limit or scene too plain?)")
        if prop == dshow.PAN:
            result[key] = d if dx < 0 else -d   # content moves left when the camera turns right
        else:
            result[key] = d if dy > 0 else -d   # content moves down when the camera tilts up
    return result
