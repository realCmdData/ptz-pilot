# Builds dist\PTZ-Pilot.exe (single file, no Python needed on the target PC)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# The app's dependencies live in the Python 3.10 install; "python" on PATH may be a newer version.
$py = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m pip install --quiet opencv-python-headless pillow numpy comtypes pyvirtualcam pyinstaller

foreach ($f in "models\face_detection_yunet_2023mar.onnx", "models\object_tracking_vittrack_2023sep.onnx", "assets\ptz-pilot.ico") {
    if (-not (Test-Path $f)) { throw "Missing $f" }
}

& $py build_version_info.py "version_info.txt"   # not inside build\: PyInstaller --clean wipes that

& $py -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name "PTZ-Pilot" `
    --icon "assets\ptz-pilot.ico" `
    --version-file "version_info.txt" `
    --exclude-module matplotlib --exclude-module pandas --exclude-module scipy `
    --collect-all pyvirtualcam `
    --add-data "models\face_detection_yunet_2023mar.onnx;models" `
    --add-data "models\object_tracking_vittrack_2023sep.onnx;models" `
    --add-data "assets\ptz-pilot.ico;assets" `
    app.py

Write-Host "Built: $PSScriptRoot\dist\PTZ-Pilot.exe"
