"""Which other apps currently have a webcam open (Windows camera privacy usage records)."""
import ctypes
import os
import sys
import threading
import time
import winreg
from ctypes import wintypes

WEBCAM_KEY = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"


def _running_image_paths():
    psapi, k32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = wintypes.HANDLE
    pids = (wintypes.DWORD * 8192)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
        return None
    paths = set()
    buf = ctypes.create_unicode_buffer(1024)
    for pid in pids[:needed.value // ctypes.sizeof(wintypes.DWORD)]:
        handle = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            continue
        size = wintypes.DWORD(len(buf))
        if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            paths.add(buf.value.lower())
        k32.CloseHandle(handle)
    return paths


def _in_use(key):
    """Windows writes LastUsedTimeStop = 0 while the app has the camera open."""
    try:
        start, _ = winreg.QueryValueEx(key, "LastUsedTimeStart")
        stop, _ = winreg.QueryValueEx(key, "LastUsedTimeStop")
    except OSError:
        return False
    return start > 0 and stop == 0


def apps_using_webcam(own_exe=None):
    """Names of apps (other than own_exe) that have a webcam open right now."""
    own = (own_exe or sys.executable).lower()
    running = _running_image_paths()
    names = []
    try:
        root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, WEBCAM_KEY)
    except OSError:
        return names
    with root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if sub == "NonPackaged":
                with winreg.OpenKey(root, sub) as np_root:
                    j = 0
                    while True:
                        try:
                            app = winreg.EnumKey(np_root, j)
                        except OSError:
                            break
                        j += 1
                        path = app.replace("#", "\\").lower()
                        if path == own:
                            continue
                        with winreg.OpenKey(np_root, app) as key:
                            # an app that crashed never writes the stop time: require it to still run
                            if _in_use(key) and (running is None or path in running):
                                names.append(os.path.basename(path))
            else:
                with winreg.OpenKey(root, sub) as key:
                    if _in_use(key):
                        names.append(sub.split("_")[0])   # packaged app, e.g. MSTeams_8wekyb3d8bbwe
    return names


class WebcamUsageMonitor:
    """Polls apps_using_webcam() in the background."""

    def __init__(self, interval=3.0):
        self.interval = interval
        self.apps = []
        self._stop = False
        threading.Thread(target=self._run, daemon=True, name="webcam-usage").start()

    def stop(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            try:
                self.apps = apps_using_webcam()
            except Exception:
                self.apps = []
            time.sleep(self.interval)


if __name__ == "__main__":
    print("apps using a webcam:", apps_using_webcam() or "none")
