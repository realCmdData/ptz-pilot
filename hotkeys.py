"""Configurable keyboard shortcuts, optionally working while other apps are in front.

Uses a low-level keyboard hook (WH_KEYBOARD_LL) because holding a key to move the camera needs
both key-down and key-up events; RegisterHotKey only reports presses.
"""
import ctypes
import os
import threading
from ctypes import wintypes

MODIFIER_KEYS = {0x10: "Shift", 0xA0: "Shift", 0xA1: "Shift", 0x11: "Ctrl", 0xA2: "Ctrl", 0xA3: "Ctrl",
                 0x12: "Alt", 0xA4: "Alt", 0xA5: "Alt", 0x5B: "Win", 0x5C: "Win"}
MODIFIER_ORDER = ("Ctrl", "Alt", "Shift", "Win")
ESCAPE = 0x1B

KEY_NAMES = {0x25: "Left", 0x26: "Up", 0x27: "Right", 0x28: "Down", 0x21: "PageUp", 0x22: "PageDown",
             0x24: "Home", 0x23: "End", 0x2D: "Insert", 0x2E: "Delete", 0x20: "Space", 0x0D: "Enter",
             0x09: "Tab", 0x08: "Backspace", 0x13: "Pause", 0x91: "ScrollLock", 0xBB: "Plus", 0xBD: "Minus",
             0xBC: "Comma", 0xBE: "Period", 0x6B: "NumPlus", 0x6D: "NumMinus", 0x6A: "NumMultiply",
             0x6F: "NumDivide"}
KEY_NAMES.update({0x30 + i: str(i) for i in range(10)})
KEY_NAMES.update({0x41 + i: chr(0x41 + i) for i in range(26)})
KEY_NAMES.update({0x60 + i: f"Num{i}" for i in range(10)})
KEY_NAMES.update({0x70 + i: f"F{i + 1}" for i in range(24)})
NAME_TO_VK = {name.lower(): vk for vk, name in KEY_NAMES.items()}


def parse(text):
    """'Ctrl+Alt+Left' -> (frozenset({'Ctrl', 'Alt'}), 0x25). None if empty or unknown."""
    parts = [p.strip() for p in (text or "").split("+") if p.strip()]
    if not parts:
        return None
    mods = {p.capitalize() for p in parts[:-1]}
    if not mods <= set(MODIFIER_ORDER) or parts[-1].lower() not in NAME_TO_VK:
        return None
    return frozenset(mods), NAME_TO_VK[parts[-1].lower()]


def format_binding(binding):
    if binding is None:
        return ""
    mods, vk = binding
    return "+".join([m for m in MODIFIER_ORDER if m in mods] + [KEY_NAMES.get(vk, f"Key{vk}")])


user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
user32.SetWindowsHookExW.restype = ctypes.c_void_p
user32.CallNextHookEx.argtypes = (ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.CallNextHookEx.restype = ctypes.c_ssize_t
user32.UnhookWindowsHookEx.argtypes = (ctypes.c_void_p,)
user32.GetMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
user32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
kernel32.GetModuleHandleW.restype = wintypes.HMODULE


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KeyboardHook:
    """Calls on_key(pressed, action, first_press) on the hook thread for bound keys.

    global_keys False: keys only act while this app is in front and are passed on as usual.
    global_keys True: keys act everywhere and are not passed on to the app in front."""

    def __init__(self, on_key):
        self.on_key = on_key
        self.bindings = {}          # action -> (frozenset of modifier names, vk)
        self.global_keys = False
        self._mods = set()          # pressed modifier vks
        self._held = {}             # vk -> action
        self._capture = None
        self._thread_id = None
        self._proc = HOOKPROC(self._handle)   # keep a reference: the hook calls into it
        self._pid = os.getpid()
        threading.Thread(target=self._run, daemon=True, name="keyboard-hook").start()

    def capture_next(self, callback):
        """The next key press (with its modifiers) goes to callback instead of acting. Esc gives None."""
        self._capture = callback

    def stop(self):
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)   # WM_QUIT

    def _run(self):
        self._thread_id = kernel32.GetCurrentThreadId()
        hook = user32.SetWindowsHookExW(13, self._proc, kernel32.GetModuleHandleW(None), 0)   # WH_KEYBOARD_LL
        if not hook:
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass
        user32.UnhookWindowsHookEx(hook)

    def _app_in_front(self):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
        return pid.value == self._pid

    def _handle(self, code, wparam, lparam):
        if code == 0:
            info = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            try:
                if self._event(info.vkCode, wparam in (0x0100, 0x0104)):   # WM_KEYDOWN, WM_SYSKEYDOWN
                    return 1
            except Exception:
                pass
        return user32.CallNextHookEx(None, code, wparam, lparam)

    def _event(self, vk, down):
        """Returns True to swallow the key."""
        if vk in MODIFIER_KEYS:
            (self._mods.add if down else self._mods.discard)(vk)
            return False
        if down and self._capture is not None:
            callback, self._capture = self._capture, None
            callback(None if vk == ESCAPE else (frozenset(MODIFIER_KEYS[m] for m in self._mods), vk))
            return True
        if not down:
            action = self._held.pop(vk, None)
            if action is None:
                return False
            self.on_key(False, action, True)
            return self.global_keys
        binding = (frozenset(MODIFIER_KEYS[m] for m in self._mods), vk)
        for action, bound in self.bindings.items():
            if bound == binding:
                if not (self.global_keys or self._app_in_front()):
                    return False
                first = vk not in self._held
                self._held[vk] = action
                self.on_key(True, action, first)
                return self.global_keys
        return False
