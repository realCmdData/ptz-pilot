<p align="center"><img src="assets/ptz-pilot.png" width="128" alt="PTZ Pilot icon"></p>

<h1 align="center">PTZ Pilot</h1>

<p align="center"><b>A hands-free camera operator for USB pan/tilt/zoom webcams on Windows.</b><br>
Move the camera, let it follow you, keep a face in a tight close-up, and hand the picture to OBS – in one small app.</p>

---

## Features

- **Pan, tilt and zoom** – hold-to-move buttons, arrow keys, click the picture to aim, scroll to zoom, saved positions (keys 1–9).
- **Follow me** – finds faces and keeps you framed: *Wide*, *Medium*, *Close* or *Face only* (an aggressive close-up that predicts where your face is going).
- **Automatic zoom** – *Off*, *Gentle*, *Normal* or *Quick*.
- **Parking** – when no app uses the camera, it turns away (pan 90, tilt −90, zoom 0) and the picture is switched off. It comes back as soon as the camera is needed.
- **Share with OBS** – send the picture (optionally with tracking boxes) to *OBS Virtual Camera*, or open local browser links for an OBS Browser Source.
- **Light on your PC** – about half a CPU core while following (measured on a Ryzen 9 9900X3D).
- **Start with Windows** (starts minimized) and a **preview toggle**.

## Supported cameras

Any camera that exposes pan/tilt/zoom through the standard UVC controls (DirectShow `IAMCameraControl`). Relative (continuous) movement is used when the camera offers it; otherwise it steps absolute positions.

Developed and tested with the **Yealink MB Cam12X Pro**. It includes a few measured quirks for that camera: it reports a tilt range up to +45 but never tilts above 0, its relative pan direction is inverted, and its motors run at a fixed speed.

PTZ Pilot is an independent project and is not affiliated with or endorsed by Yealink.

## Getting started

1. Build `PTZ-Pilot.exe` (see [Building from source](#building-from-source)). The exe runs without installation or Python.
2. Plug in the camera and start the app. It moves the camera to 0 / 0 / 0.
3. Press **Follow me**. The first time, it briefly moves the camera to learn which way it turns.

Only one app can use a camera's picture at a time. To use the camera in OBS, Teams or Zoom while PTZ Pilot follows you, turn on **Share → Send the picture to OBS Virtual Camera** and pick *OBS Virtual Camera* in the other app. Moving the camera works even while another app has the picture.

## How the tracking works

| Step | What | Why |
|---|---|---|
| Find faces | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (OpenCV `FaceDetectorYN`), about once per second while following | small, fast face detector |
| Follow the face | [VitTrack](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack) (OpenCV `TrackerVit`), 15 times per second | about 4–8 ms per update instead of detecting every frame |
| Smooth / predict | One Euro filter; in *Face only* a constant-velocity Kalman filter that separates the subject's motion from the camera's own moves | less jitter, no lag on steady motion, bridges short tracking gaps |
| Steer | start/stop moves with latency compensation; short timed bursts when zoomed in | fixed-speed PTZ motors overshoot tight shots otherwise |

Everything runs locally. The app makes no internet connections; the optional browser links only listen on `127.0.0.1`.

## Building from source

Requirements: Windows 10/11, Python 3.10 (64-bit).

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
```

This installs the dependencies and creates `dist\PTZ-Pilot.exe` with PyInstaller. To run from source instead: `python app.py`.

Settings are stored in `%APPDATA%\PTZ Pilot\settings.json`.

| File | Purpose |
|---|---|
| `app.py` | user interface, parking, video on demand |
| `dshow.py` | DirectShow/UVC camera control (COM via comtypes) |
| `tracker.py` | face detection + tracking, framing, prediction, direction check |
| `outputs.py` | OBS Virtual Camera output and local MJPEG links |
| `usage.py` | detects other apps using a webcam (Windows privacy usage records) |
| `autostart.py` | "Start with Windows" |

## License

PTZ Pilot is free software under the [GNU General Public License v3.0](LICENSE).

## Third-party components

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
