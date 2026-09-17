"""Video outputs for other apps (OBS etc.) while this app owns the camera.

* VirtualCamOutput: writes into the "OBS Virtual Camera" DirectShow device (pyvirtualcam).
* MjpegServer: local MJPEG streams + pages for an OBS Browser Source; clean and debug
  feeds can be used at the same time.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

FEEDS = ("Clean", "Debug overlay")
SLUGS = {"Clean": "clean", "Debug overlay": "debug"}
OUT_W, OUT_H, OUT_FPS = 1280, 720, 30


class FrameHub:
    """Hands out output-sized clean/debug frames, rendered at most once per camera frame."""

    def __init__(self, render_debug):
        self.video = None
        self._render_debug = render_debug
        self._cache = {}
        self._lock = threading.Lock()
        self._last_debug_request = 0.0
        self._placeholder = np.zeros((OUT_H, OUT_W, 3), np.uint8)
        cv2.putText(self._placeholder, "No camera video", (470, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (160, 160, 160), 2, cv2.LINE_AA)

    def debug_active(self):
        return time.monotonic() - self._last_debug_request < 1.0

    def frame(self, feed):
        """Return (sequence number, BGR image). Sequence is -1 while there is no video."""
        debug = feed == "Debug overlay"
        if debug:
            self._last_debug_request = time.monotonic()
        video = self.video
        src, seq = (video.frame, video.seq) if video is not None else (None, -1)
        if src is None:
            return -1, self._placeholder
        with self._lock:
            cached = self._cache.get(feed)
            if cached is not None and cached[0] == seq:
                return cached
            img = src if src.shape[:2] == (OUT_H, OUT_W) else cv2.resize(src, (OUT_W, OUT_H))
            if debug:
                img = self._render_debug(img.copy())
            self._cache[feed] = (seq, img)
            return seq, img


class VirtualCamOutput:
    def __init__(self, hub):
        self.hub = hub
        self.feed = FEEDS[0]
        self.status = "Off"
        self.running = False
        self._stop = False
        self._thread = None

    def start(self):
        if self.running:
            return
        self._stop = False
        self.running = True
        self.status = "Starting…"
        self._thread = threading.Thread(target=self._run, daemon=True, name="virtual-cam")
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(3)
        self._thread = None

    def _run(self):
        try:
            import pyvirtualcam
            cam = pyvirtualcam.Camera(OUT_W, OUT_H, OUT_FPS, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
        except Exception as exc:
            self.status = f"Couldn't start. Is OBS installed, and is its own virtual camera stopped? ({exc})"
            self.running = False
            return
        try:
            with cam:
                self.status = f"On. In OBS, choose “{cam.device}”"
                while not self._stop:
                    cam.send(self.hub.frame(self.feed)[1])
                    cam.sleep_until_next_frame()
        except Exception as exc:
            self.status = f"Stopped: {exc}"
        else:
            self.status = "Off"
        finally:
            self.running = False


PAGE = """<!doctype html><html><head><title>{title}</title></head>
<body style="margin:0;background:#000;overflow:hidden">{body}</body></html>"""
SINGLE = '<img src="/{slug}.mjpg" style="width:100vw;height:100vh;object-fit:contain;display:block">'
INDEX = ('<div style="display:flex;flex-wrap:wrap;gap:8px;padding:8px;font:14px sans-serif;color:#ccc">'
         '<div style="flex:1 1 480px">Clean<img src="/clean.mjpg" style="width:100%;display:block"></div>'
         '<div style="flex:1 1 480px">Debug overlay<img src="/debug.mjpg" style="width:100%;display:block"></div>'
         '</div>')


class MjpegServer:
    def __init__(self, hub):
        self.hub = hub
        self.port = None
        self.viewers = {feed: 0 for feed in FEEDS}
        self._server = None
        self._lock = threading.Lock()

    @property
    def running(self):
        return self._server is not None

    @property
    def status(self):
        if not self.running:
            return "Off"
        watching = sum(self.viewers.values())
        return f"On, {watching} watching" if watching else "On"

    def page_url(self, feed):
        return f"http://127.0.0.1:{self.port}/{SLUGS[feed]}"

    def start(self, port):
        self.stop()
        server = ThreadingHTTPServer(("127.0.0.1", port), self._handler_class())
        server.daemon_threads = True
        self._server, self.port = server, port
        threading.Thread(target=server.serve_forever, daemon=True, name="mjpeg-server").start()

    def stop(self):
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()

    def _handler_class(self):
        owner = self
        feeds = {slug: feed for feed, slug in SLUGS.items()}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = self.path.split("?")[0].rstrip("/")
                if path in ("", "/index.html"):
                    self._html("PTZ Pilot streams", INDEX)
                elif path.lstrip("/") in feeds:
                    slug = path.lstrip("/")
                    self._html(feeds[slug], SINGLE.format(slug=slug))
                elif path.endswith(".mjpg") and path[1:-5] in feeds:
                    self._stream(feeds[path[1:-5]])
                else:
                    self.send_error(404)

            def _html(self, title, body):
                data = PAGE.format(title=title, body=body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, feed):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.end_headers()
                with owner._lock:
                    owner.viewers[feed] += 1
                try:
                    last = None
                    while owner.running:
                        seq, img = owner.hub.frame(feed)
                        if seq == last:
                            time.sleep(0.5 if seq < 0 else 0.005)
                            continue
                        last = seq
                        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not ok:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                         % len(jpg))
                        self.wfile.write(jpg.tobytes())
                        self.wfile.write(b"\r\n")
                except OSError:
                    pass  # viewer disconnected
                finally:
                    with owner._lock:
                        owner.viewers[feed] -= 1

        return Handler
