"""Start the app when the user signs in to Windows (per-user Run key, no admin rights needed)."""
import os
import sys
import winreg

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "PTZ Pilot"
STARTUP_ARG = "--startup"   # launched by Windows: start minimized


def command():
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" {STARTUP_ARG}'
    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    return f'"{pythonw if os.path.exists(pythonw) else sys.executable}" "{script}" {STARTUP_ARG}'


def is_enabled(key_path=RUN_KEY):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            winreg.QueryValueEx(key, VALUE_NAME)
            return True
    except OSError:
        return False


def set_enabled(enabled, key_path=RUN_KEY):
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        if enabled:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command())
        else:
            try:
                winreg.DeleteValue(key, VALUE_NAME)
            except FileNotFoundError:
                pass
