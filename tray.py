"""Icon in the hidden icons area of the taskbar (system tray), so the app can keep running
without a window. Wraps pystray, which needs its own thread for the icon's message loop."""
import threading

import pystray
from PIL import Image


class TrayIcon:
    def __init__(self, name, icon_path, on_open, on_toggle_follow, on_quit, is_following):
        self._image = Image.open(icon_path)
        menu = pystray.Menu(
            pystray.MenuItem("Open " + name, lambda icon, item: on_open(), default=True),
            pystray.MenuItem("Follow me", lambda icon, item: on_toggle_follow(),
                             checked=lambda item: is_following()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda icon, item: on_quit()))
        self.icon = pystray.Icon(name, self._image, name, menu)
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if not self.running:
            self._thread = threading.Thread(target=self.icon.run, daemon=True, name="tray-icon")
            self._thread.start()

    def stop(self):
        try:
            self.icon.stop()
        except Exception:
            pass

    def refresh(self):
        """Redraw the menu, for example after Follow me was switched on or off elsewhere."""
        try:
            if self.running:
                self.icon.update_menu()
        except Exception:
            pass

    def message(self, text, title=None):
        try:
            if self.running:
                self.icon.notify(text, title or self.icon.name)
        except Exception:
            pass   # notifications may be switched off in Windows
