"""Minimal DirectShow COM bindings for UVC camera controls (IAMCameraControl).

Enumeration order of CLSID_VideoInputDeviceCategory matches OpenCV's CAP_DSHOW
index order, so the same index can be used for capture and control.

Measured on the Yealink MB Cam12X Pro:
  * Set() returns in ~8 ms, but while an ABSOLUTE move is running every further
    request (Set or Get) blocks until the motor arrives (up to ~3 s) and the move
    cannot be preempted.
  * RELATIVE pan/tilt/zoom (KSPROPERTY_CAMERACONTROL_*_RELATIVE, ids 10/11/13)
    start, stop and reverse within ~7 ms. The speed magnitude is ignored; the
    camera moves at a fixed rate (~28 pan units/s wide, slower when zoomed).
So interactive motion and tracking use relative moves; absolute positions are
only used for sliders/presets and read back while the camera is idle.
"""
import ctypes
import threading
import time
from ctypes import POINTER, c_long, c_ulong, c_void_p, wintypes
from dataclasses import dataclass

import comtypes
import comtypes.client
from comtypes import COMMETHOD, GUID, HRESULT, STDMETHOD, IUnknown
from comtypes.automation import VARIANT
from comtypes.persist import IPropertyBag

CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")

# IAMCameraControl / KSPROPERTY_CAMERACONTROL properties
PAN, TILT, ROLL, ZOOM, EXPOSURE, IRIS, FOCUS = range(7)
CONTROL_NAMES = {PAN: "Pan", TILT: "Tilt", ROLL: "Roll", ZOOM: "Zoom",
                 EXPOSURE: "Exposure", IRIS: "Iris", FOCUS: "Focus"}
REL_PROP = {PAN: 10, TILT: 11, ZOOM: 13}
FLAG_AUTO = 0x1
FLAG_MANUAL = 0x2

ABS_UNITS_PER_SEC = 20.0   # used to estimate how long an absolute move blocks the device
EMULATED_STEP_INTERVAL = 0.06
# Relative command sign that INCREASES the absolute value, where known up front.
# Unknown devices start with +1 and learn it from read-backs after each move.
KNOWN_REL_SIGNS = {"mb cam12x": {PAN: -1}}
# Real mechanical limits where the driver reports more than the camera does:
# the MB Cam12X Pro reports tilt -90…45 but never tilts above 0.
KNOWN_LIMITS = {"mb cam12x": {TILT: (None, 0)}}


class IMoniker(IUnknown):
    _iid_ = GUID("{0000000f-0000-0000-C000-000000000046}")


IMoniker._methods_ = [
    # IPersist / IPersistStream (placeholders keep the vtable aligned)
    STDMETHOD(HRESULT, "GetClassID", [c_void_p]),
    STDMETHOD(HRESULT, "IsDirty"),
    STDMETHOD(HRESULT, "Load", [c_void_p]),
    STDMETHOD(HRESULT, "Save", [c_void_p, wintypes.BOOL]),
    STDMETHOD(HRESULT, "GetSizeMax", [c_void_p]),
    # IMoniker
    COMMETHOD([], HRESULT, "BindToObject",
              (["in"], c_void_p, "pbc"), (["in"], c_void_p, "pmkToLeft"),
              (["in"], POINTER(GUID), "riidResult"),
              (["out"], POINTER(POINTER(IUnknown)), "ppvResult")),
    COMMETHOD([], HRESULT, "BindToStorage",
              (["in"], c_void_p, "pbc"), (["in"], c_void_p, "pmkToLeft"),
              (["in"], POINTER(GUID), "riid"),
              (["out"], POINTER(POINTER(IUnknown)), "ppvObj")),
]


class IEnumMoniker(IUnknown):
    _iid_ = GUID("{00000102-0000-0000-C000-000000000046}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Next",
                  (["in"], c_ulong, "celt"),
                  (["out"], POINTER(POINTER(IMoniker)), "rgelt"),
                  (["out"], POINTER(c_ulong), "pceltFetched")),
        STDMETHOD(HRESULT, "Skip", [c_ulong]),
        STDMETHOD(HRESULT, "Reset"),
        STDMETHOD(HRESULT, "Clone", [c_void_p]),
    ]


class ICreateDevEnum(IUnknown):
    _iid_ = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
    _methods_ = [
        COMMETHOD([], HRESULT, "CreateClassEnumerator",
                  (["in"], POINTER(GUID), "clsidDeviceClass"),
                  (["out"], POINTER(POINTER(IEnumMoniker)), "ppEnumMoniker"),
                  (["in"], wintypes.DWORD, "dwFlags")),
    ]


class IAMCameraControl(IUnknown):
    _iid_ = GUID("{C6E13370-30AC-11d0-A18C-00A0C9118956}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetRange",
                  (["in"], c_long, "Property"),
                  (["out"], POINTER(c_long), "pMin"),
                  (["out"], POINTER(c_long), "pMax"),
                  (["out"], POINTER(c_long), "pSteppingDelta"),
                  (["out"], POINTER(c_long), "pDefault"),
                  (["out"], POINTER(c_long), "pCapsFlags")),
        COMMETHOD([], HRESULT, "Set",
                  (["in"], c_long, "Property"),
                  (["in"], c_long, "lValue"),
                  (["in"], c_long, "Flags")),
        COMMETHOD([], HRESULT, "Get",
                  (["in"], c_long, "Property"),
                  (["out"], POINTER(c_long), "lValue"),
                  (["out"], POINTER(c_long), "Flags")),
    ]


class CAUUID(ctypes.Structure):
    _fields_ = [("cElems", c_ulong), ("pElems", POINTER(GUID))]


class ISpecifyPropertyPages(IUnknown):
    _iid_ = GUID("{B196B28B-BAB4-101A-B69C-00AA00341D07}")
    _methods_ = [COMMETHOD([], HRESULT, "GetPages", (["out"], POINTER(CAUUID), "pPages"))]


def _monikers():
    dev_enum = comtypes.client.CreateObject(CLSID_SystemDeviceEnum, interface=ICreateDevEnum)
    enum = dev_enum.CreateClassEnumerator(CLSID_VideoInputDeviceCategory, 0)
    if not enum:
        return
    while True:
        moniker, fetched = enum.Next(1)
        if not fetched or not moniker:
            return
        yield moniker


def _friendly_name(moniker):
    try:
        bag = moniker.BindToStorage(None, None, IPropertyBag._iid_).QueryInterface(IPropertyBag)
        value = bag.Read("FriendlyName", VARIANT(), None)
        return value.value if isinstance(value, VARIANT) else str(value)
    except Exception:
        return "Unknown camera"


def list_video_devices():
    """Return friendly names in DirectShow (= OpenCV CAP_DSHOW) index order."""
    comtypes.CoInitialize()
    try:
        return [_friendly_name(m) for m in _monikers()]
    finally:
        comtypes.CoUninitialize()


def _bind_filter(index):
    for i, moniker in enumerate(_monikers()):
        if i == index:
            return moniker.BindToObject(None, None, IUnknown._iid_)
    raise IndexError(f"No video device at index {index}")


@dataclass
class ControlRange:
    prop: int
    min: int
    max: int
    step: int
    default: int
    caps: int

    @property
    def name(self):
        return CONTROL_NAMES[self.prop]

    @property
    def span(self):
        return max(1, self.max - self.min)

    @property
    def supports_auto(self):
        return bool(self.caps & FLAG_AUTO)

    def clamp(self, value):
        value = int(round(value))
        if self.step > 1:
            value = self.min + round((value - self.min) / self.step) * self.step
        return max(self.min, min(self.max, value))


class CameraControl:
    """Owns the COM objects on a dedicated STA thread.

    * set(): absolute positions, coalesced (only the latest value per control is sent).
    * move(): continuous motion in the direction that increases (+1) or decreases (-1)
      the absolute value, with an optional lease after which it stops by itself.
      Uses relative UVC controls when available, otherwise emulates them with
      absolute steps.
    """

    def __init__(self, device_index, device_name=""):
        self.device_index = device_index
        self.ranges = {}        # prop -> ControlRange
        self.rel_ranges = {}    # prop -> (min, max) of the relative control
        self.values = {}        # prop -> last known absolute value
        self.auto = {}
        self.error = None
        self.rel_sign = {PAN: 1, TILT: 1, ZOOM: 1}
        for key, signs in KNOWN_REL_SIGNS.items():
            if key in device_name.lower():
                self.rel_sign.update(signs)
        self._limits = {}
        for key, limits in KNOWN_LIMITS.items():
            if key in device_name.lower():
                self._limits.update(limits)
        self._abs_pending = {}  # prop -> (value, flags)
        self._moves = {}        # prop -> (sign, speed, deadline)
        self._active = {}       # prop -> sign currently executing
        self._segment = {}      # prop -> [value at start, set of signs] for sign learning
        self._next_step = {}
        self._readback = set()
        self._last_motion = 0.0
        self._busy_until = 0.0
        self._cv = threading.Condition()
        self._stop = False
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="uvc-control")
        self._thread.start()
        self._ready.wait(10)

    # ------------------------------------------------------------ public API
    @property
    def available(self):
        return bool(self.ranges)

    def supported(self, prop):
        return prop in self.ranges

    def get(self, prop):
        with self._cv:
            if prop in self._abs_pending:
                return self._abs_pending[prop][0]
        return self.values.get(prop)

    def set(self, prop, value, auto=False):
        rng = self.ranges.get(prop)
        if rng is None:
            return
        with self._cv:
            self._moves.pop(prop, None)
            self._abs_pending[prop] = (rng.clamp(value), FLAG_AUTO if auto else FLAG_MANUAL)
            self._cv.notify()

    def nudge(self, prop, delta):
        current = self.get(prop)
        if current is not None:
            self.set(prop, current + delta)

    def at_limit(self, prop, sign):
        rng, value = self.ranges.get(prop), self.values.get(prop)
        if rng is None or value is None or not sign:
            return False
        return value >= rng.max if sign > 0 else value <= rng.min

    def move(self, prop, sign, speed=0.6, lease=None):
        if prop not in self.ranges:
            return
        if self.at_limit(prop, sign) and not self._active.get(prop):
            sign = 0
        with self._cv:
            if sign == 0:
                self._moves.pop(prop, None)
            else:
                deadline = time.monotonic() + lease if lease else None
                self._moves[prop] = (1 if sign > 0 else -1, speed, deadline)
            self._cv.notify()

    def stop_all(self):
        with self._cv:
            self._moves.clear()
            self._cv.notify()

    def is_moving(self, prop=None):
        return bool(self._active) if prop is None else bool(self._active.get(prop))

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(4)

    # ------------------------------------------------------------ worker thread
    def _run(self):
        comtypes.CoInitialize()
        cam = None
        try:
            try:
                cam = _bind_filter(self.device_index).QueryInterface(IAMCameraControl)
            except Exception as exc:
                self.error = f"IAMCameraControl not available: {exc}"
                return
            self._probe(cam)
            self._ready.set()
            self._loop(cam)
        finally:
            self._ready.set()
            cam = None
            comtypes.CoUninitialize()

    def _probe(self, cam):
        for prop in CONTROL_NAMES:
            try:
                lo, hi, step, default, caps = cam.GetRange(prop)
            except Exception:
                continue
            if hi > lo:
                real_lo, real_hi = self._limits.get(prop, (None, None))
                lo = lo if real_lo is None else max(lo, real_lo)
                hi = hi if real_hi is None else min(hi, real_hi)
                default = max(lo, min(hi, default))
                self.ranges[prop] = ControlRange(prop, lo, hi, max(1, step), default, caps)
                self._read_back(cam, prop)
        for prop, rel in REL_PROP.items():
            if prop not in self.ranges:
                continue
            try:
                lo, hi = cam.GetRange(rel)[:2]
            except Exception:
                continue
            if hi > 0:
                self.rel_ranges[prop] = (lo, hi)

    def _loop(self, cam):
        while True:
            with self._cv:
                if not self._abs_pending and not self._stop:
                    if self._moves or self._active:
                        timeout = 0.02
                    elif self._readback:
                        timeout = 0.1
                    else:
                        timeout = None
                    self._cv.wait(timeout)
                stopping = self._stop
                abs_batch, self._abs_pending = ({}, self._abs_pending) if stopping else (self._abs_pending, {})
                moves = {} if stopping else dict(self._moves)

            now = time.monotonic()
            wanted = {}
            for prop, (sign, speed, deadline) in moves.items():
                if deadline is not None and now > deadline:
                    with self._cv:
                        if self._moves.get(prop) == (sign, speed, deadline):
                            del self._moves[prop]
                    continue
                if prop not in abs_batch:
                    wanted[prop] = (sign, speed)
            for prop in set(self._active) | set(wanted):
                sign, speed = wanted.get(prop, (0, 0.0))
                self._drive(cam, prop, sign, speed, now)
            if stopping:
                return

            for prop, (value, flags) in abs_batch.items():
                old = self.values.get(prop, value)
                try:
                    cam.Set(prop, value, flags)
                    self.values[prop] = value
                    self.auto[prop] = bool(flags & FLAG_AUTO)
                except Exception as exc:
                    self.error = f"Set {CONTROL_NAMES[prop]} failed: {exc}"
                self._segment.pop(prop, None)
                self._busy_until = max(self._busy_until,
                                       time.monotonic() + abs(value - old) / ABS_UNITS_PER_SEC + 0.3)
                self._readback.add(prop)

            now = time.monotonic()
            if self._readback and not self._active and now - self._last_motion > 0.5 and now > self._busy_until:
                with self._cv:
                    busy = bool(self._abs_pending or self._moves)
                if not busy:
                    for prop in list(self._readback):
                        self._finish_segment(cam, prop)
                    self._readback.clear()

    def _drive(self, cam, prop, sign, speed, now):
        current = self._active.get(prop, 0)
        if prop in self.rel_ranges:
            if sign != current:
                hi = self.rel_ranges[prop][1]
                magnitude = max(1, min(hi, round(hi * speed))) if sign else 0
                try:
                    cam.Set(REL_PROP[prop], self.rel_sign[prop] * sign * magnitude, FLAG_MANUAL)
                except Exception as exc:
                    self.error = f"Relative {CONTROL_NAMES[prop]} failed: {exc}"
                    sign = 0
                if sign:
                    seg = self._segment.get(prop)
                    if seg is None:
                        self._segment[prop] = [self.values.get(prop), {sign}]
                    else:
                        seg[1].add(sign)
                elif current:
                    self._readback.add(prop)
        elif sign and now >= self._next_step.get(prop, 0.0):
            rng = self.ranges[prop]
            step = max(rng.step, round(rng.span * 0.012 * (0.4 + speed)))
            base = self.values.get(prop)
            value = rng.clamp((rng.default if base is None else base) + sign * step)
            try:
                cam.Set(prop, value, FLAG_MANUAL)
                self.values[prop] = value
            except Exception as exc:
                self.error = f"Set {CONTROL_NAMES[prop]} failed: {exc}"
            self._next_step[prop] = now + EMULATED_STEP_INTERVAL
        if sign:
            self._active[prop] = sign
        else:
            self._active.pop(prop, None)
        if sign or current:
            self._last_motion = now

    def _finish_segment(self, cam, prop):
        seg = self._segment.pop(prop, None)
        self._read_back(cam, prop)
        if not seg or seg[0] is None or len(seg[1]) != 1 or prop not in self.values:
            return
        delta = self.values[prop] - seg[0]
        intended = next(iter(seg[1]))
        if abs(delta) >= 2 and (delta > 0) != (intended > 0):
            self.rel_sign[prop] *= -1

    def _read_back(self, cam, prop):
        try:
            value, flags = cam.Get(prop)
            self.values[prop] = value
            self.auto[prop] = bool(flags & FLAG_AUTO)
        except Exception:
            pass


def open_property_dialog(device_index, hwnd=0, caption="Camera properties"):
    """Show the driver's native property pages (blocking, run on its own thread)."""
    def worker():
        comtypes.CoInitialize()
        try:
            filt = _bind_filter(device_index)
            pages = filt.QueryInterface(ISpecifyPropertyPages).GetPages()
            objs = (POINTER(IUnknown) * 1)(filt)
            ctypes.oledll.oleaut32.OleCreatePropertyFrame(
                wintypes.HWND(hwnd), 30, 30, ctypes.c_wchar_p(caption), 1, objs,
                pages.cElems, pages.pElems, 0, 0, None)
            ctypes.windll.ole32.CoTaskMemFree(pages.pElems)
        except Exception:
            pass
        finally:
            comtypes.CoUninitialize()
    threading.Thread(target=worker, daemon=True, name="property-dialog").start()


if __name__ == "__main__":
    for idx, name in enumerate(list_video_devices()):
        print(idx, name)
        ctl = CameraControl(idx, name)
        if ctl.error:
            print("   ", ctl.error)
        for r in ctl.ranges.values():
            rel = ctl.rel_ranges.get(r.prop)
            print(f"    {r.name:9s} min={r.min} max={r.max} step={r.step} default={r.default} "
                  f"current={ctl.values.get(r.prop)} relative={'yes' if rel else 'no'}")
        ctl.close()
