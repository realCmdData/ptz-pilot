# Third-party components

The Windows executable bundles the following components. Licenses as declared by each project (checked September 2026). Please see each project for the full license text.

## Models (in `models/`)

| Model | Source | License |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | [OpenCV Model Zoo: YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |
| `object_tracking_vittrack_2023sep.onnx` | [OpenCV Model Zoo: VitTrack](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack) | Apache 2.0 |

YuNet's authors ask to be cited when it is used:

> Wu, Wei and Peng, Hanyang and Yu, Shiqi. *YuNet: A Tiny Millisecond-level Face Detector.* Machine Intelligence Research 20(5), 656-665, 2023.

## Python packages

| Package | Version used | License (package metadata) |
|---|---|---|
| [opencv-python-headless](https://github.com/opencv/opencv-python) | 4.13.0.92 | Apache 2.0 |
| [NumPy](https://numpy.org) | 2.2.6 | BSD |
| [Pillow](https://python-pillow.org) | 12.2.0 | MIT-CMU |
| [comtypes](https://github.com/enthought/comtypes) | 1.4.16 | MIT |
| [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) | 0.15.0 | GPL v2 |
| [PyInstaller](https://pyinstaller.org) (build tool) | 6.22.3 | GPL v2 with bootloader exception |
| Python / Tkinter | 3.10 | PSF License |

**Note on pyvirtualcam and prebuilt executables:** PTZ Pilot is licensed under GPL v3. pyvirtualcam declares GPL v2 and does not state "or any later version", so it has to be treated as GPL v2 only, which the FSF considers incompatible with GPL v3 when both are combined in one distributed program. This repository contains only PTZ Pilot's own source code; pyvirtualcam is installed as a dependency at build time. Before distributing a built `PTZ-Pilot.exe`, either get clarification from pyvirtualcam's author that later GPL versions are allowed, or build without the virtual camera output.
