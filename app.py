"""PTZ Pilot – a hands-free camera operator for UVC pan/tilt/zoom webcams."""
import ctypes
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

import autostart
import dshow
import outputs
import tracker
import usage

# OpenCV's default thread pool spreads each small detection over every core and its idle workers
# spin: measured ~200 % of a core in 25 worker threads. Single-threaded is ~6 ms per frame.
cv2.setNumThreads(1)

APP_NAME = "PTZ Pilot"
APPDATA = os.environ.get("APPDATA", os.path.expanduser("~"))
CONFIG_DIR = os.path.join(APPDATA, "PTZ Pilot")
CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")
LEGACY_CONFIG_FILES = (os.path.join(APPDATA, "Gimbalist", "settings.json"),      # earlier names
                       os.path.join(APPDATA, "MB12xControl", "settings.json"))
PREFERRED_DEVICES = ("mb cam12x", "yealink")
PTZ_PROPS = (dshow.PAN, dshow.TILT, dshow.ZOOM)
IMAGE_PROPS = (dshow.FOCUS, dshow.EXPOSURE, dshow.IRIS)
MOVE_SPEED = 0.6

# Plain choices shown in the Follow tab -> tracker settings
WHO = {"One person": "Closest to center", "Everyone in view": "Whole group"}
SHOT = {"Wide": 0.35, "Medium": 0.55, "Close": 0.8, "Face only": 0.65}
ZOOM_STYLES = {"Off": None, "Gentle": 0.1, "Normal": 0.4, "Quick": 0.8}
REACTION = {"Calm": 0.14, "Normal": 0.09, "Quick": 0.05}
WHEN_LOST = {"Stay": "Stay", "Zoom out": "Zoom out", "Go home": "Go home"}
FOLLOW_CHOICES = {"who": WHO, "shot": SHOT, "zoom": ZOOM_STYLES, "reaction": REACTION, "lost": WHEN_LOST}
FOLLOW_DEFAULTS = {"who": "One person", "shot": "Medium", "zoom": "Gentle", "reaction": "Normal", "lost": "Stay",
                   "boxes": True}
FEED_CLEAN, FEED_BOXES = outputs.FEEDS
START_POSITION = {dshow.PAN: 0, dshow.TILT: 0, dshow.ZOOM: 0}
PARK_POSITION = {dshow.PAN: 90, dshow.TILT: -90, dshow.ZOOM: 0}
PARK_DELAYS = {"30 s": 30, "1 min": 60, "5 min": 300, "15 min": 900}


def load_config():
    for path in (CONFIG_FILE, *LEGACY_CONFIG_FILES):   # settings from earlier names carry over
        try:
            with open(path, encoding="utf-8-sig") as f:  # tolerate a BOM from hand edits
                return json.load(f)
        except (OSError, ValueError):
            continue
    return {}


def save_config(cfg):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


class VideoSource:
    """Grabs frames on a background thread; retries while another app holds the camera."""

    def __init__(self, index):
        self.index = index
        self.frame = None
        self.ts = 0.0
        self.seq = 0
        self.fps = 0.0
        self.busy = False   # another app holds the camera picture
        self.status = "Starting the camera picture…"
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="video")
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(3)

    def _run(self):
        while not self._stop:
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                self.frame = None
                self.busy = True
                self.status = ("Another app (OBS, Teams, …) is using the camera picture. "
                               "Close it to see the picture here – moving the camera still works.")
                for _ in range(30):
                    if self._stop:
                        return
                    time.sleep(0.1)
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.busy = False
            self.status = "Camera picture is live"
            failures, count, t0 = 0, 0, time.monotonic()
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures > 30:
                        break
                    time.sleep(0.02)
                    continue
                failures = 0
                self.frame, self.ts = frame, time.monotonic()
                self.seq += 1
                count += 1
                if self.ts - t0 >= 1.0:
                    self.fps, count, t0 = count / (self.ts - t0), 0, self.ts
            cap.release()
            self.frame = None
            if not self._stop:
                self.status = "Lost the camera picture – reconnecting…"


class DetectionWorker:
    MAX_RATE = 20.0   # skip frames closer than this: every 2nd frame of a 30 fps camera (15 Hz)

    def __init__(self, video, detector):
        self.video = video
        self.detector = detector
        self.enabled = False
        self.result = (0.0, [], None)   # (frame time, faces as Person, tracked face box or None)
        self.seq = 0
        self.fps = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="detect")
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(3)

    def _run(self):
        last_seq, last_ts, count, t0 = -1, 0.0, 0, time.monotonic()
        while not self._stop:
            frame, ts, seq = self.video.frame, self.video.ts, self.video.seq
            if not self.enabled or frame is None or seq == last_seq or ts - last_ts < 1.0 / self.MAX_RATE:
                if frame is None:
                    self.result = (0.0, [], None)
                    self.detector.reset()
                time.sleep(0.005)
                continue
            last_seq, last_ts = seq, ts
            try:
                faces, target = self.detector.process(frame, ts)
            except (cv2.error, ValueError):
                self.detector.reset()
                faces, target = [], None
            # the tracked face moves every update; other faces only refresh when YuNet runs
            people = []
            if target is not None:
                tracked = tracker._face_person(*target, 1.0)
                tracked.kind = "tracked"
                people.append(tracked)
            people += [tracker._face_person(*f) for f in faces
                       if target is None or tracker._iou(f[:4], target) < 0.2]
            self.result = (ts, people, target)
            self.seq += 1
            count += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                self.fps, count, t0 = count / (now - t0), 0, now


class App:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        self.devices = []
        self.device_index = None
        self.control = None
        self.video = None
        self.detection = None
        self.detector = tracker.FaceTracker()
        saved = self.cfg.get("follow", {})
        self.follow = {k: saved[k] if k in saved and (k == "boxes" or saved[k] in FOLLOW_CHOICES[k]) else v
                       for k, v in FOLLOW_DEFAULTS.items()}
        self.framer = tracker.AutoFramer()
        self._apply_follow_choices()
        self.calibrating = False
        self._events = queue.Queue()     # callbacks from worker threads, run on the Tk thread
        self._manual = {}                # prop -> direction of a held button/key
        self._key_stop_jobs = {}
        self._photo = None
        self._preview_size = (960, 540)
        self._view_rect = (0, 0, 1, 1)
        self._last_det_seq = -1
        # plain copies of UI state for output threads (Tk variables are main-thread only)
        self.overlay = {"show_boxes": True, "tracking": False, "calibrating": False, "status": ""}
        self.hub = outputs.FrameHub(self._render_debug)
        self.vcam = outputs.VirtualCamOutput(self.hub)
        self.streams = outputs.MjpegServer(self.hub)
        self.usage = usage.WebcamUsageMonitor()
        self.parked = False
        self._idle_since = None
        self._last_manual = 0.0
        self._need_picture_until = 0.0

        root.title(APP_NAME)
        root.geometry(self.cfg.get("geometry", "1360x820"))
        root.minsize(960, 640)
        try:
            root.iconbitmap(tracker.resource_path("assets", "ptz-pilot.ico"))
        except tk.TclError:
            pass
        if getattr(sys, "frozen", False) and autostart.is_enabled():
            try:
                autostart.set_enabled(True)   # keep the startup entry pointing at this exe if it was moved
            except OSError:
                pass
        self._build_ui()
        self._bind_keys()
        self.refresh_devices(auto_connect=True)
        self._restore_outputs()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(33, self._render_loop)
        root.after(30, self._tracking_loop)
        root.after(250, self._sync_loop)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Hint.TLabel", foreground="#666")
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"))

        top = ttk.Frame(self.root, padding=(8, 6))
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Camera").pack(side=tk.LEFT)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(top, textvariable=self.device_var, state="readonly", width=30)
        self.device_combo.pack(side=tk.LEFT, padx=4)
        self.device_combo.bind("<<ComboboxSelected>>", lambda e: self.connect(self.device_combo.current()))
        ttk.Button(top, text="Search again", command=self.refresh_devices).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Camera settings…", command=self.open_driver_dialog).pack(side=tk.LEFT, padx=2)
        self.tracking_var = tk.BooleanVar(value=False)
        self.follow_btn = tk.Button(top, font=("Segoe UI", 11, "bold"), relief=tk.FLAT, padx=16, pady=3,
                                    cursor="hand2", command=self.toggle_follow)
        self.follow_btn.pack(side=tk.RIGHT, padx=6)
        self._refresh_follow_button()
        self.autostart_var = tk.BooleanVar(value=autostart.is_enabled())
        ttk.Checkbutton(top, text="Start with Windows", variable=self.autostart_var,
                        command=self._autostart_toggled).pack(side=tk.RIGHT, padx=8)
        self.preview_var = tk.BooleanVar(value=bool(self.cfg.get("preview", True)))
        ttk.Checkbutton(top, text="Preview", variable=self.preview_var,
                        command=self._preview_toggled).pack(side=tk.RIGHT, padx=8)

        self.status_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN,
                  padding=(8, 2)).pack(side=tk.BOTTOM, fill=tk.X)

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)
        side = ttk.Frame(body, width=390, padding=(4, 4, 8, 8))
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        self.preview = tk.Label(body, bg="#111", fg="#bbb", font=("Segoe UI", 12), cursor="crosshair",
                                text="No camera", wraplength=600)
        self.preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=(4, 8))
        self.preview.bind("<Configure>", lambda e: (setattr(self, "_preview_size", (e.width, e.height)),
                                                    self.preview.configure(wraplength=max(200, e.width - 40))))
        self.preview.bind("<Button-1>", self._preview_click)
        self.preview.bind("<MouseWheel>", self._wheel_zoom)

        self.tabs = ttk.Notebook(side)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.root.focus_set())  # keep arrow keys for the camera
        self.move_tab = ttk.Frame(self.tabs, padding=10)
        self.follow_tab = ttk.Frame(self.tabs, padding=10)
        self.share_tab = ttk.Frame(self.tabs, padding=10)
        self.picture_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.move_tab, text="  Move  ")
        self.tabs.add(self.follow_tab, text="  Follow  ")
        self.tabs.add(self.share_tab, text="  Share  ")
        self._build_move_tab()
        self._build_follow_tab()
        self._build_share_tab()

    def _refresh_follow_button(self):
        if self.tracking_var.get():
            self.follow_btn.configure(text="●  Following  (T)", bg="#2e7d32", fg="white",
                                      activebackground="#1b5e20", activeforeground="white")
        else:
            self.follow_btn.configure(text="○  Follow me  (T)", bg="#dcdcdc", fg="#222",
                                      activebackground="#c8c8c8", activeforeground="#222")

    def _build_move_tab(self):
        tab = self.move_tab
        s = self.framer.s
        ttk.Label(tab, text="Hold a button to move the camera", style="Title.TLabel").pack(anchor=tk.W)

        pad = ttk.Frame(tab)
        pad.pack(pady=(10, 6))
        buttons = [("▲", 0, 1, dshow.TILT, lambda: s.tilt_up), ("◀", 1, 0, dshow.PAN, lambda: -s.pan_right),
                   ("▶", 1, 2, dshow.PAN, lambda: s.pan_right), ("▼", 2, 1, dshow.TILT, lambda: -s.tilt_up)]
        for text, r, c, prop, direction in buttons:
            b = ttk.Button(pad, text=text, width=6)
            b.grid(row=r, column=c, padx=3, pady=3, ipady=12)
            self._hold_button(b, prop, direction)
        ttk.Button(pad, text="⌂", width=6, command=self.go_home).grid(row=1, column=1, padx=3, pady=3, ipady=12)

        zoom = ttk.Frame(tab)
        zoom.pack(pady=4)
        for text, sign in (("−   Zoom out", -1), ("+   Zoom in", 1)):
            b = ttk.Button(zoom, text=text, width=14)
            b.pack(side=tk.LEFT, padx=4, ipady=8)
            self._hold_button(b, dshow.ZOOM, lambda sign=sign: sign)

        self.position_label = ttk.Label(tab, text="", style="Hint.TLabel")
        self.position_label.pack(pady=(8, 0))
        ttk.Label(tab, text="Keyboard: arrow keys to turn, + and − to zoom.\n"
                            "On the picture: click to aim there, scroll to zoom.\n"
                            "⌂ returns to the start position.",
                  style="Hint.TLabel", justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 12))

        park = ttk.LabelFrame(tab, text="Parking", padding=8)
        park.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        p = self.cfg.get("parking", {})
        self.park_var = tk.BooleanVar(value=bool(p.get("enabled", True)))
        ttk.Checkbutton(park, text="Park the camera when nobody is using it", variable=self.park_var,
                        command=self._parking_changed).pack(anchor=tk.W)
        row = ttk.Frame(park)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text="after").pack(side=tk.LEFT, padx=(0, 6))
        self.park_delay_var = tk.StringVar(value=p.get("delay") if p.get("delay") in PARK_DELAYS else "1 min")
        for label in PARK_DELAYS:
            ttk.Radiobutton(row, text=label, value=label, variable=self.park_delay_var, style="Toolbutton",
                            command=self._parking_changed).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.park_status = ttk.Label(park, text="", style="Hint.TLabel", wraplength=340)
        self.park_status.pack(anchor=tk.W, pady=(6, 0))

        saved = ttk.LabelFrame(tab, text="Saved positions  (keys 1–9)", padding=8)
        saved.pack(fill=tk.BOTH, expand=True)
        self.preset_list = tk.Listbox(saved, height=5, activestyle="none", font=("Segoe UI", 10))
        self.preset_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preset_list.bind("<Double-Button-1>", lambda e: self.recall_preset())
        col = ttk.Frame(saved)
        col.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        ttk.Button(col, text="Save current…", command=self.save_preset).pack(fill=tk.X, pady=2)
        ttk.Button(col, text="Go there", command=self.recall_preset).pack(fill=tk.X, pady=2)
        ttk.Button(col, text="Delete", command=self.delete_preset).pack(fill=tk.X, pady=2)

    def _build_follow_tab(self):
        tab = self.follow_tab
        self.follow_vars = {}

        def choice(title, key, hint=None):
            ttk.Label(tab, text=title, style="Title.TLabel").pack(anchor=tk.W, pady=(12 if self.follow_vars else 0, 3))
            var = tk.StringVar(value=self.follow[key])
            row = ttk.Frame(tab)
            row.pack(fill=tk.X)
            for option in FOLLOW_CHOICES[key]:
                ttk.Radiobutton(row, text=option, value=option, variable=var, style="Toolbutton",
                                command=lambda k=key, v=var: self._follow_changed(k, v.get())).pack(
                    side=tk.LEFT, fill=tk.X, expand=True, padx=1, ipady=3)
            if hint:
                ttk.Label(tab, text=hint, style="Hint.TLabel", wraplength=350).pack(anchor=tk.W, pady=(2, 0))
            self.follow_vars[key] = var

        choice("Who to follow", "who", "Tip: click a person in the picture to follow exactly them.")
        choice("Shot size", "shot", "“Face only” zooms in hard on one face and follows it closely.")
        choice("Automatic zoom", "zoom")
        choice("How quickly the camera reacts", "reaction")
        choice("When nobody is visible", "lost")

        self.boxes_var = tk.BooleanVar(value=self.follow["boxes"])
        ttk.Checkbutton(tab, text="Show boxes around people in the picture", variable=self.boxes_var,
                        command=lambda: self._follow_changed("boxes", self.boxes_var.get())).pack(anchor=tk.W,
                                                                                               pady=(14, 0))
        ttk.Separator(tab).pack(fill=tk.X, pady=12)
        self.track_status = tk.StringVar(value="Off – press “Follow me” to start")
        ttk.Label(tab, textvariable=self.track_status, style="Status.TLabel", wraplength=350).pack(anchor=tk.W)
        ttk.Button(tab, text="Camera turns the wrong way? Check directions again",
                   command=self.calibrate).pack(anchor=tk.W, pady=(14, 0))

    def _build_share_tab(self):
        tab = self.share_tab
        o = self.cfg.get("output", {})
        self.streams_port = int(o.get("port", 8765))

        obs = ttk.LabelFrame(tab, text="Use this camera in OBS", padding=8)
        obs.pack(fill=tk.X)
        self.vcam_var = tk.BooleanVar(value=bool(o.get("vcam", False)))
        ttk.Checkbutton(obs, text="Send the picture to OBS Virtual Camera", variable=self.vcam_var,
                        command=self._vcam_toggled).pack(anchor=tk.W)
        self.vcam_boxes_var = tk.BooleanVar(value=o.get("vcam_feed") == FEED_BOXES)
        ttk.Checkbutton(obs, text="Include the tracking boxes", variable=self.vcam_boxes_var,
                        command=self._vcam_feed_changed).pack(anchor=tk.W, pady=(2, 0))
        self.vcam_status = ttk.Label(obs, text="Off", wraplength=340)
        self.vcam_status.pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(obs, text="In OBS: Sources → + → Video Capture Device → “OBS Virtual Camera”.\n"
                            "Don't press “Start Virtual Camera” in OBS while this is on.",
                  style="Hint.TLabel", wraplength=340, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))

        links = ttk.LabelFrame(tab, text="Browser links", padding=8)
        links.pack(fill=tk.X, pady=12)
        self.streams_var = tk.BooleanVar(value=bool(o.get("streams", False)))
        ttk.Checkbutton(links, text="Turn on links", variable=self.streams_var,
                        command=self._streams_toggled).pack(anchor=tk.W)
        self.url_vars = {}
        for label, feed in (("Picture", FEED_CLEAN), ("With boxes", FEED_BOXES)):
            ttk.Label(links, text=label).pack(anchor=tk.W, pady=(8, 1))
            row = ttk.Frame(links)
            row.pack(fill=tk.X)
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(row, text="Copy", width=6, command=lambda f=feed: self._copy_link(f)).pack(side=tk.LEFT, padx=2)
            ttk.Button(row, text="Open", width=6, command=lambda f=feed: self._open_link(f)).pack(side=tk.LEFT)
            self.url_vars[feed] = var
        self.streams_status = ttk.Label(links, text="Off")
        self.streams_status.pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(links, text="In OBS: Sources → + → Browser, paste a link, size 1280 × 720.\n"
                              "“With boxes” is handy to see what the tracking is doing.\n"
                              "Only this PC can open the links.",
                  style="Hint.TLabel", wraplength=340, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))

    def _build_picture_tab(self):
        """Focus/exposure sliders – only shown for cameras that offer them."""
        for child in self.picture_tab.winfo_children():
            child.destroy()
        props = [p for p in IMAGE_PROPS if self.control is not None and self.control.supported(p)]
        if str(self.picture_tab) in self.tabs.tabs():
            self.tabs.forget(self.picture_tab)
        if not props:
            return
        self.tabs.add(self.picture_tab, text="  Picture  ")
        for prop in props:
            rng = self.control.ranges[prop]
            ttk.Label(self.picture_tab, text=rng.name, style="Title.TLabel").pack(anchor=tk.W, pady=(8, 2))
            row = ttk.Frame(self.picture_tab)
            row.pack(fill=tk.X)
            current = self.control.get(prop)
            var = tk.DoubleVar(value=rng.default if current is None else current)
            scale = ttk.Scale(row, from_=rng.min, to=rng.max, variable=var,
                              command=lambda v, p=prop, var=var: self._picture_changed(p, var))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            # jump straight to the clicked spot instead of creeping one step per click
            scale.bind("<Button-1>", lambda e, s=scale, r=rng, var=var, p=prop: (
                var.set(r.min + (r.max - r.min) * min(1, max(0, e.x / max(1, s.winfo_width())))),
                self._picture_changed(p, var)))
            if rng.supports_auto:
                auto = tk.BooleanVar(value=self.control.auto.get(prop, False))
                ttk.Checkbutton(row, text="Automatic", variable=auto,
                                command=lambda p=prop, a=auto, v=var: self.control.set(p, v.get(), auto=a.get())
                                ).pack(side=tk.LEFT, padx=(8, 0))

    def _bind_keys(self):
        def guarded(fn):
            def handler(event):
                if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Listbox)):
                    return None
                fn(event)
                return "break"
            return handler

        s = self.framer.s
        r = self.root
        keys = [("Left", dshow.PAN, lambda: -s.pan_right), ("Right", dshow.PAN, lambda: s.pan_right),
                ("Up", dshow.TILT, lambda: s.tilt_up), ("Down", dshow.TILT, lambda: -s.tilt_up),
                ("plus", dshow.ZOOM, lambda: 1), ("KP_Add", dshow.ZOOM, lambda: 1),
                ("equal", dshow.ZOOM, lambda: 1), ("minus", dshow.ZOOM, lambda: -1),
                ("KP_Subtract", dshow.ZOOM, lambda: -1)]
        for keysym, prop, direction in keys:
            r.bind(f"<KeyPress-{keysym}>", guarded(lambda e, p=prop, d=direction: self._key_press(p, d())))
            r.bind(f"<KeyRelease-{keysym}>", guarded(lambda e, p=prop: self._key_release(p)))
        r.bind("<Home>", guarded(lambda e: self.go_home()))
        r.bind("<t>", guarded(lambda e: self.toggle_follow()))
        for n in range(1, 10):
            r.bind(str(n), guarded(lambda e, n=n: self.recall_preset(n - 1)))
        r.bind("<FocusOut>", lambda e: self._stop_manual() if e.widget is r else None)

    # ------------------------------------------------------------ devices
    def refresh_devices(self, auto_connect=False):
        try:
            self.devices = dshow.list_video_devices()
        except Exception as exc:
            self.devices = []
            self.status_var.set(f"Couldn't list cameras: {exc}")
        self.device_combo["values"] = self.devices
        if not self.devices:
            self.device_var.set("")
            self.status_var.set("No camera found – plug it in and press “Search again”")
            return
        if not auto_connect and self.device_index is not None:
            return
        wanted = self.cfg.get("device")
        index = self.devices.index(wanted) if wanted in self.devices else next(
            (i for i, n in enumerate(self.devices) if any(p in n.lower() for p in PREFERRED_DEVICES)), 0)
        self.connect(index)

    @property
    def device_name(self):
        return self.devices[self.device_index] if self.device_index is not None else ""

    def connect(self, index):
        if index < 0 or index >= len(self.devices):
            return
        self.tracking_var.set(False)
        self._refresh_follow_button()
        self.disconnect()
        self.device_index = index
        self.device_combo.current(index)
        self.cfg["device"] = self.device_name
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self.control = dshow.CameraControl(index, self.device_name)
        finally:
            self.root.config(cursor="")
        self._move_to(START_POSITION)   # every start begins from 0 / 0 / 0
        self.parked = False
        self._idle_since = None
        self._sync_video()
        if "obs virtual" in self.device_name.lower() and self.vcam.running:
            self.vcam_var.set(False)
            self.vcam.stop()
        self.framer.reset()
        cal = self.cfg.get("calibration", {}).get(self.device_name, {})
        self.framer.s.pan_right = cal.get("pan_right", 1)
        self.framer.s.tilt_up = cal.get("tilt_up", 1)
        self._build_picture_tab()
        self._load_presets()
        self._tracking_toggled()

    def disconnect(self):
        self._manual.clear()
        self.hub.video = None
        for worker in (self.detection, self.video, self.control):
            if worker is not None:
                (worker.close if hasattr(worker, "close") else worker.stop)()
        self.detection = self.video = self.control = None

    def open_driver_dialog(self):
        if self.device_index is None:
            return
        hwnd = int(self.root.wm_frame(), 16)
        dshow.open_property_dialog(self.device_index, hwnd, f"{self.device_name} – settings")

    # ------------------------------------------------------------ manual control
    def _manual_start(self, prop, direction):
        if self.control is None or not self.control.supported(prop) or self.calibrating:
            return
        if self._manual.get(prop) == direction:
            return
        self.framer.pause(3)
        self._mark_manual()
        self._manual[prop] = direction
        self.control.move(prop, direction, speed=MOVE_SPEED)

    def _manual_stop(self, prop):
        if prop in self._manual and self.control is not None:
            del self._manual[prop]
            self.control.move(prop, 0)
            self.framer.pause(2)

    def _stop_manual(self):
        for prop in list(self._manual):
            self._manual_stop(prop)

    def _key_press(self, prop, direction):
        job = self._key_stop_jobs.pop(prop, None)
        if job:
            self.root.after_cancel(job)
        self._manual_start(prop, direction)

    def _key_release(self, prop):
        # debounce: keyboard auto-repeat may emit release/press pairs
        job = self._key_stop_jobs.pop(prop, None)
        if job:
            self.root.after_cancel(job)
        self._key_stop_jobs[prop] = self.root.after(60, lambda: (self._key_stop_jobs.pop(prop, None),
                                                                 self._manual_stop(prop)))

    def _hold_button(self, button, prop, direction):
        button.bind("<ButtonPress-1>", lambda e: self._manual_start(prop, direction()))
        button.bind("<ButtonRelease-1>", lambda e: self._manual_stop(prop))
        button.bind("<Leave>", lambda e: self._manual_stop(prop))

    def _wheel_zoom(self, event):
        if self.control is None or not self.control.supported(dshow.ZOOM) or self.calibrating:
            return
        self.framer.pause(3)
        self._mark_manual()
        self.control.move(dshow.ZOOM, 1 if event.delta > 0 else -1, speed=MOVE_SPEED, lease=0.25)

    def _picture_changed(self, prop, var):
        if self.control is not None:
            self.control.set(prop, var.get())

    def go_home(self):
        if self.control is None:
            return
        self.framer.pause(4)
        self._mark_manual()
        for prop in PTZ_PROPS:
            if self.control.supported(prop):
                self.control.set(prop, self.control.ranges[prop].default)

    def _move_to(self, position):
        if self.control is None:
            return
        for prop, value in position.items():
            if self.control.supported(prop):
                self.control.set(prop, value)   # clamped to what the camera allows

    def _mark_manual(self):
        """The user is operating the camera: never park or un-park under their hands."""
        self._last_manual = time.monotonic()
        self.parked = False

    def _parking_changed(self):
        self._idle_since = None

    def _video_needed(self):
        """Only capture the camera picture while something uses it; never while parked."""
        if self.control is None or self.parked:
            return False
        return (self.preview_var.get() or self.tracking_var.get() or self.calibrating or self.vcam.running
                or time.monotonic() < self._need_picture_until
                or (self.streams.running and sum(self.streams.viewers.values()) > 0))

    def _sync_video(self):
        needed = self._video_needed()
        if needed and self.video is None and self.device_index is not None:
            self.video = VideoSource(self.device_index)
            self.hub.video = self.video
            self.detection = DetectionWorker(self.video, self.detector)
        elif not needed and self.video is not None:
            if self.control is not None:
                self.framer.release(self.control)
            self.hub.video = None
            video, detection = self.video, self.detection
            self.video = self.detection = None
            # stopping joins the worker threads (up to a few seconds): keep the UI responsive
            threading.Thread(target=lambda: (detection.stop(), video.stop()), daemon=True).start()

    def _preview_toggled(self):
        if self.preview_var.get() and self.parked:   # wanting to see the picture counts as using the camera
            self._mark_manual()
            self._move_to(START_POSITION)
        self._sync_video()

    def _autostart_toggled(self):
        try:
            autostart.set_enabled(self.autostart_var.get())
        except OSError as exc:
            self.autostart_var.set(autostart.is_enabled())
            messagebox.showwarning(APP_NAME, f"Couldn't change the Windows startup setting:\n{exc}",
                                   parent=self.root)

    def _camera_use(self):
        """Why the camera counts as in use right now (None if it's idle)."""
        if self.tracking_var.get() or self.calibrating:
            return "following"
        if self.video is not None and self.video.busy:
            return "another app has the camera picture"
        apps = self.usage.apps
        if apps:
            return "used by " + ", ".join(sorted(set(apps)))
        if self.streams.running and sum(self.streams.viewers.values()):
            return "someone is watching a browser link"
        return None

    def _update_parking(self):
        if self.control is None or not self.park_var.get():
            self._idle_since = None
            self.park_status.configure(text="Off")
            return
        now = time.monotonic()
        delay = PARK_DELAYS[self.park_delay_var.get()]
        use = self._camera_use()
        if use is not None:
            self._idle_since = None
            if self.parked:   # someone needs the camera again: face the room
                self.parked = False
                self.framer.pause(4)
                self._move_to(START_POSITION)
            self.park_status.configure(text=f"In use – {use}")
            return
        if self.parked:
            self.park_status.configure(text="Parked – returns to the start position as soon as the camera is used")
            return
        manual_quiet = now - self._last_manual
        if self._idle_since is None:
            self._idle_since = now
        left = delay - min(now - self._idle_since, manual_quiet)
        if left <= 0:
            self.parked = True
            self._move_to(PARK_POSITION)
            self.status_var.set("Camera parked – nobody is using it")
            return
        self.park_status.configure(text=f"Nobody is using the camera – parking in {int(left) // 60}:{int(left) % 60:02d}")

    def _preview_click(self, event):
        if self.control is None or self.video is None or self.video.frame is None:
            return
        x0, y0, w, h = self._view_rect
        px, py = (event.x - x0) / w, (event.y - y0) / h
        if not (0 <= px <= 1 and 0 <= py <= 1):
            return
        if self.tracking_var.get() and self.framer.s.target != "Whole group" and self.detector.pick(px, py):
            self.framer.release(self.control)
            self.framer.reset()
            self.track_status.set("Following the person you clicked")
            return
        self.framer.pause(4)
        self._mark_manual()
        s = self.framer.s
        fov = tracker.estimate_fov_units(self.control)
        if fov and self.control.supported(dshow.PAN):
            self.control.nudge(dshow.PAN, (px - 0.5) * fov * s.pan_right)
        if fov and self.control.supported(dshow.TILT):
            self.control.nudge(dshow.TILT, (0.5 - py) * fov * 9 / 16 * s.tilt_up)

    # ------------------------------------------------------------ saved positions
    def _device_presets(self):
        return self.cfg.setdefault("presets", {}).setdefault(self.device_name or "?", {})

    def _load_presets(self):
        self.preset_list.delete(0, tk.END)
        for i, name in enumerate(self._device_presets()):
            self.preset_list.insert(tk.END, f"{i + 1}.  {name}" if i < 9 else name)

    def _preset_name(self, list_index):
        names = list(self._device_presets())
        return names[list_index] if 0 <= list_index < len(names) else None

    def save_preset(self):
        if self.control is None or not self.control.available:
            return
        name = simpledialog.askstring(APP_NAME, "Name for this position:", parent=self.root)
        if not name:
            return
        self._device_presets()[name.strip()] = {
            dshow.CONTROL_NAMES[p]: self.control.get(p) for p in PTZ_PROPS if self.control.supported(p)}
        save_config(self.cfg)
        self._load_presets()

    def recall_preset(self, list_index=None):
        if list_index is None:
            sel = self.preset_list.curselection()
            if not sel:
                return
            list_index = sel[0]
        name = self._preset_name(list_index)
        if name is None or self.control is None:
            return
        self.framer.pause(5)
        self._mark_manual()
        by_name = {v: k for k, v in dshow.CONTROL_NAMES.items()}
        for key, value in self._device_presets()[name].items():
            if key in by_name and value is not None:
                self.control.set(by_name[key], value)
        self.status_var.set(f"Moving to “{name}”")

    def delete_preset(self):
        sel = self.preset_list.curselection()
        name = self._preset_name(sel[0]) if sel else None
        if name and messagebox.askyesno(APP_NAME, f"Delete the saved position “{name}”?", parent=self.root):
            del self._device_presets()[name]
            save_config(self.cfg)
            self._load_presets()

    # ------------------------------------------------------------ following & direction check
    def _follow_changed(self, key, value):
        self.follow[key] = value
        self._apply_follow_choices()

    def _apply_follow_choices(self):
        s, f = self.framer.s, self.follow
        s.target = WHO[f["who"]]
        s.framing = SHOT[f["shot"]]
        zoom = ZOOM_STYLES[f["zoom"]]
        s.auto_zoom = zoom is not None
        s.zoom_aggr = zoom if zoom is not None else s.zoom_aggr
        s.deadzone = REACTION[f["reaction"]]
        s.on_lost = WHEN_LOST[f["lost"]]
        s.head_height, s.max_zoom, s.lost_timeout, s.speed = 0.36, 0.75, 5.0, MOVE_SPEED
        s.face_only = f["shot"] == "Face only"
        if s.face_only:   # aggressive close-up on one face
            s.target = WHO["One person"]
            s.auto_zoom = True
            s.zoom_aggr = 1.0
            s.deadzone = max(s.deadzone, 0.08)   # a tight shot needs slack, or it keeps correcting
            s.head_height = 0.45
            s.max_zoom = 1.0

    def toggle_follow(self):
        self.tracking_var.set(not self.tracking_var.get())
        self._tracking_toggled()

    def _is_calibrated(self):
        return self.device_name in self.cfg.get("calibration", {})

    def _tracking_toggled(self):
        if self.control is not None:
            self.framer.release(self.control)
        self.framer.reset()
        if not self.tracking_var.get():
            self.track_status.set("Off – press “Follow me” to start")
        elif self.control is not None and not self._is_calibrated() and (
                self.control.supported(dshow.PAN) or self.control.supported(dshow.TILT)):
            self.tracking_var.set(False)
            self.calibrate(enable_tracking=True)
        self._refresh_follow_button()

    def calibrate(self, enable_tracking=False, attempt=0):
        if self.calibrating or self.control is None:
            return
        if (self.video is None or self.video.frame is None) and attempt < 20:
            # the picture may be switched off (preview off / parked): start it and try again shortly
            self._need_picture_until = time.monotonic() + 15
            if self.parked:
                self._mark_manual()
            self._sync_video()
            self.track_status.set("Starting the camera picture…")
            self.root.after(300, lambda: self.calibrate(enable_tracking, attempt + 1))
            return
        if self.video is None or self.video.frame is None:
            messagebox.showinfo(APP_NAME, "The app needs to see the camera picture for this.\n"
                                "Close other apps that use the camera (OBS, Teams, …) and try again.",
                                parent=self.root)
            return
        self.calibrating = True
        was_tracking = self.tracking_var.get()
        self.tracking_var.set(False)
        self._refresh_follow_button()
        self._stop_manual()
        self.framer.release(self.control)
        control, video, device = self.control, self.video, self.device_name
        self.track_status.set("Checking which way the camera turns – it will move briefly…")

        def worker():
            try:
                result, error = tracker.calibrate(control, video, lambda m: self._events.put(
                    lambda: self.track_status.set(m))), None
            except Exception as exc:
                result, error = None, str(exc)
            self._events.put(lambda: self._calibration_done(device, result, error,
                                                            enable_tracking or was_tracking))
        threading.Thread(target=worker, daemon=True, name="calibrate").start()

    def _calibration_done(self, device, result, error, enable_tracking):
        self.calibrating = False
        if device != self.device_name:
            return
        if error:
            self.track_status.set(f"Couldn't check the directions: {error}")
            return
        self.cfg.setdefault("calibration", {})[device] = result
        save_config(self.cfg)
        self.framer.s.pan_right = result.get("pan_right", 1)
        self.framer.s.tilt_up = result.get("tilt_up", 1)
        self.track_status.set("Directions checked")
        if enable_tracking:
            self.tracking_var.set(True)
            self._tracking_toggled()

    def _tracking_loop(self):
        try:
            while True:
                try:
                    self._events.get_nowait()()
                except queue.Empty:
                    break
            tracking = self.tracking_var.get() and not self.calibrating
            self.detector.mode = self.framer.s.target
            self.detector.hint = self.framer.head if self.framer.predicting else None
            self.overlay.update(show_boxes=self.follow["boxes"], tracking=self.tracking_var.get(),
                                calibrating=self.calibrating, status=self.track_status.get())
            if self.detection is not None:
                self.detection.enabled = tracking or self.follow["boxes"] or self.hub.debug_active()
                if tracking and self.detection.seq != self._last_det_seq:
                    self._last_det_seq = self.detection.seq
                    ts, people, target = self.detection.result
                    can_steer = self.control is not None and any(self.control.supported(p) for p in PTZ_PROPS)
                    if self.video.frame is None:
                        self.framer.release(self.control)
                        self.track_status.set("Waiting for the camera picture…")
                    elif not can_steer:
                        self.track_status.set("This camera can't be moved by the app")
                    else:
                        self.track_status.set(self.framer.update(target, [p.raw for p in people], ts,
                                                                 self.control))
        finally:
            self.root.after(30, self._tracking_loop)

    # ------------------------------------------------------------ outputs
    def _restore_outputs(self):
        self.vcam.feed = FEED_BOXES if self.vcam_boxes_var.get() else FEED_CLEAN
        if self.vcam_var.get():
            self._vcam_toggled()
        if self.streams_var.get():
            self._streams_toggled()
        self._update_links()

    def _vcam_toggled(self):
        if not self.vcam_var.get():
            self.vcam.stop()
            return
        if "obs virtual" in self.device_name.lower():
            messagebox.showinfo(APP_NAME, "“OBS Virtual Camera” is selected as the camera. "
                                "Choose your real camera at the top first.", parent=self.root)
            self.vcam_var.set(False)
            return
        self._vcam_feed_changed()
        self.vcam.start()

    def _vcam_feed_changed(self):
        self.vcam.feed = FEED_BOXES if self.vcam_boxes_var.get() else FEED_CLEAN

    def _streams_toggled(self):
        if self.streams_var.get():
            try:
                self.streams.start(self.streams_port)
            except OSError as exc:
                self.streams_var.set(False)
                self.streams_status.configure(text=f"Couldn't turn on links: {exc}")
        else:
            self.streams.stop()
        self._update_links()

    def _update_links(self):
        for feed, var in self.url_vars.items():
            var.set(self.streams.page_url(feed) if self.streams.running else "Turn on links first")

    def _ensure_links(self):
        if not self.streams.running:
            self.streams_var.set(True)
            self._streams_toggled()
        return self.streams.running

    def _copy_link(self, feed):
        if self._ensure_links():
            self.root.clipboard_clear()
            self.root.clipboard_append(self.streams.page_url(feed))
            self.status_var.set("Link copied")

    def _open_link(self, feed):
        if self._ensure_links():
            webbrowser.open(self.streams.page_url(feed))

    # ------------------------------------------------------------ loops
    def _sync_loop(self):
        try:
            if self.control is not None:
                parts = []
                names = {dshow.PAN: "Left/right", dshow.TILT: "Up/down"}
                for prop in (dshow.PAN, dshow.TILT):
                    if self.control.supported(prop) and self.control.get(prop) is not None:
                        limit = " (top)" if prop == dshow.TILT and self.control.at_limit(prop, 1) else ""
                        parts.append(f"{names[prop]} {self.control.get(prop)}{limit}")
                zoom = self.control.ranges.get(dshow.ZOOM)
                if zoom is not None and self.control.get(dshow.ZOOM) is not None:
                    parts.append(f"Zoom {round(100 * (self.control.get(dshow.ZOOM) - zoom.min) / zoom.span)}%")
                self.position_label.configure(
                    text="    ·    ".join(parts) if parts else "This camera can't be moved by the app")
            if self.vcam_var.get() and not self.vcam.running:
                self.vcam_var.set(False)   # failed to start or stopped
            self.vcam_status.configure(text=self.vcam.status)
            self.streams_status.configure(text=self.streams.status)
            self._update_parking()
            self._sync_video()
            if self.video is not None:
                status = self.video.status
            elif self.control is not None:
                status = "Camera picture is off – nothing needs it right now"
            else:
                status = "No camera"
            if self.parked:
                status = "Camera parked – nobody is using it"
            if self.control is not None and self.control.error:
                status += "   |   The camera didn't accept the last command"
            self.status_var.set(status)
        finally:
            self.root.after(250, self._sync_loop)

    def _render_loop(self):
        try:
            frame = self.video.frame if self.video is not None else None
            if self.parked or not self.preview_var.get() or frame is None:
                if self.parked:
                    text = "Camera parked – the picture is off until the camera is used again"
                elif not self.preview_var.get():
                    text = "Preview is off" + ("  (following keeps working)" if self.tracking_var.get() else "")
                else:
                    text = self.video.status if self.video is not None else "No camera"
                if self.preview.cget("text") != text or self._photo is not None:
                    self.preview.configure(image="", text=text)
                    self._photo = None
            elif self.video.seq != getattr(self, "_drawn_seq", -1):
                self._drawn_seq = self.video.seq
                self._draw_frame(frame)
        finally:
            self.root.after(50, self._render_loop)   # 20 fps preview keeps the UI thread cheap

    def _draw_frame(self, frame):
        pw, ph = self._preview_size
        fh, fw = frame.shape[:2]
        scale = min(pw / fw, ph / fh)
        w, h = max(1, int(fw * scale)), max(1, int(fh * scale))
        img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        self._view_rect = ((pw - w) / 2, (ph - h) / 2, w, h)
        self._render_overlay(img, self.overlay["show_boxes"], debug=False)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self._photo, text="")

    def _render_debug(self, img):
        """Called from output threads; uses only thread-safe snapshots of app state."""
        return self._render_overlay(img, True, debug=True)

    def _render_overlay(self, img, boxes, debug):
        h, w = img.shape[:2]
        k = max(1.0, w / 960)
        st = self.overlay
        font = cv2.FONT_HERSHEY_SIMPLEX

        def rect(box, color, thickness):
            x, y, bw, bh = box
            cv2.rectangle(img, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), color,
                          max(1, int(thickness * k)))

        def text(s, org, size, color, thickness):
            cv2.putText(img, s, org, font, size * k, (0, 0, 0), max(1, int(thickness * k)) + 2, cv2.LINE_AA)
            cv2.putText(img, s, org, font, size * k, color, max(1, int(thickness * k)), cv2.LINE_AA)

        detection = self.detection
        if boxes and detection is not None and detection.enabled:
            for p in detection.result[1]:
                tracked = p.kind == "tracked"
                color = (80, 220, 80) if tracked else (170, 170, 170)
                rect(p.raw, color, 2 if tracked else 1)
                if debug:
                    text(f"{p.kind} {p.score:.2f}", (int(p.raw[0] * w), max(int(14 * k), int(p.raw[1] * h) - 4)),
                         0.45, color, 1)
        if st["calibrating"]:
            text("CHECKING DIRECTIONS", (int(12 * k), int(30 * k)), 0.8, (0, 200, 255), 2)
        elif st["tracking"]:
            s = self.framer.s
            dz = s.deadzone
            cx, cy = 0.5, s.head_height
            if debug or boxes:
                cv2.rectangle(img, (int((cx - dz) * w), int((cy - dz * 1.25) * h)),
                              (int((cx + dz) * w), int((cy + dz * 1.25) * h)), (200, 200, 200), max(1, int(k)))
            box, head = self.framer.box, self.framer.head
            if box is not None and head is not None:
                rect(box, (0, 140, 255), 3)
                cv2.circle(img, (int(head[0] * w), int(head[1] * h)), int(5 * k), (0, 140, 255), -1)
                label = "FOLLOWING"
            else:
                label = "FOLLOWING" if debug else "FOLLOWING - click a person to choose"
            text(label, (int(12 * k), int(30 * k)), 0.8, (0, 140, 255), 2)
        if debug:
            lines = [st["status"] or "Following is off"]
            if detection is not None and detection.enabled:
                lines.append(f"detection {detection.fps:.0f} Hz  |  {len(detection.result[1])} detected")
            video = self.video
            if video is not None:
                lines.append(f"camera {video.fps:.0f} fps")
            control = self.control
            if control is not None:
                pos = "  ".join(f"{dshow.CONTROL_NAMES[p]} {control.values.get(p)}" for p in PTZ_PROPS
                                if control.supported(p))
                lines.append(pos + (f"  zoom: {self.framer.zoom_state}" if st["tracking"] else ""))
            for i, line in enumerate(reversed(lines)):
                text(line.replace("…", "...").replace("–", "-").replace("“", '"').replace("”", '"'),
                     (int(12 * k), h - int((14 + 24 * i) * k)), 0.55, (255, 255, 255), 1)
        return img

    # ------------------------------------------------------------ shutdown
    def on_close(self):
        self.cfg["follow"] = dict(self.follow)
        self.cfg["parking"] = {"enabled": self.park_var.get(), "delay": self.park_delay_var.get()}
        self.cfg["preview"] = self.preview_var.get()
        self.usage.stop()
        self.cfg["output"] = {"vcam": self.vcam_var.get(),
                              "vcam_feed": FEED_BOXES if self.vcam_boxes_var.get() else FEED_CLEAN,
                              "streams": self.streams_var.get(), "port": self.streams_port}
        for old in ("tracking", "manual_speed"):
            self.cfg.pop(old, None)
        self.vcam.stop()
        self.streams.stop()
        self.cfg["geometry"] = self.root.geometry()
        save_config(self.cfg)
        self.disconnect()
        self.root.destroy()


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    try:   # own taskbar identity, so Windows shows the PTZ Pilot icon instead of Python's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PTZPilot")
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    App(root)
    if autostart.STARTUP_ARG in sys.argv:   # launched at sign-in: stay out of the way
        root.iconify()
    root.mainloop()


if __name__ == "__main__":
    main()
