"""Writes the Windows version resource for the exe, so PTZ-Pilot.exe shows a version in its
file properties. Called by build.ps1 with the output path."""
import os
import sys

from version import __version__

TEMPLATE = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({parts}),
    prodvers=({parts}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'realCmdData'),
        StringStruct('FileDescription', 'PTZ Pilot'),
        StringStruct('FileVersion', '{version}'),
        StringStruct('InternalName', 'PTZ-Pilot'),
        StringStruct('LegalCopyright', 'GNU General Public License v3.0'),
        StringStruct('OriginalFilename', 'PTZ-Pilot.exe'),
        StringStruct('ProductName', 'PTZ Pilot'),
        StringStruct('ProductVersion', '{version}')])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "build/version_info.txt"
    numbers = [int(p) for p in __version__.split(".")]
    while len(numbers) < 4:
        numbers.append(0)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.format(parts=", ".join(str(n) for n in numbers[:4]), version=__version__))
    print(f"version resource for {__version__} written to {out}")
