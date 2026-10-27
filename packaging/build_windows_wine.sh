#!/usr/bin/env bash
# Build the Windows installer from Linux using Wine (no Windows machine needed).
#   packaging/build_windows_wine.sh [output_dir]
# First run sets up a Wine prefix with Windows Python, the dependencies and Inno Setup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist-windows}"
VERSION="$(python3 -c "import re;print(re.search(r'__version__ = \"(.+?)\"', open('$ROOT/openkick/__init__.py').read())[1])")"
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine_openkick_build}" WINEDEBUG=-all WINEARCH=win64
CACHE="$WINEPREFIX/drive_c/openkick-build"
mkdir -p "$OUT" "$CACHE"
winpath() { winepath -w "$1"; }

if [ ! -x "$WINEPREFIX/drive_c/Python312/python.exe" ]; then
  wineboot -i
  curl -fL -o "$CACHE/python.exe" https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
  wine "$CACHE/python.exe" /quiet InstallAllUsers=0 PrependPath=0 Include_test=0 Include_doc=0 \
       Include_tcltk=0 Include_launcher=0 'TargetDir=C:\Python312'
  wine 'C:\Python312\python.exe' -m pip install --no-warn-script-location \
       pyinstaller PySide6-Essentials av numpy scipy mediapipe
fi
if [ ! -f "$WINEPREFIX/drive_c/InnoSetup/ISCC.exe" ]; then
  curl -fL -o "$CACHE/is.exe" https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe
  wine "$CACHE/is.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CURRENTUSER '/DIR=C:\InnoSetup'
fi
# Ship the MediaPipe model so the app works offline
MODEL="$ROOT/packaging/models/pose_landmarker_heavy.task"
if [ ! -f "$MODEL" ]; then
  mkdir -p "$(dirname "$MODEL")"
  curl -fL -o "$MODEL" https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
fi

rm -rf "$CACHE/dist"
wine 'C:\Python312\python.exe' -m PyInstaller "$(winpath "$ROOT/packaging/openkick.spec")" --noconfirm \
     --distpath "$(winpath "$CACHE/dist")" --workpath "$(winpath "$CACHE/work")"
wine 'C:\InnoSetup\ISCC.exe' /Q "/DDistDir=$(winpath "$CACHE/dist")" "/DAppVersion=$VERSION" \
     "/O$(winpath "$OUT")" "$(winpath "$ROOT/packaging/openkick.iss")"
echo "Installer: $OUT/OpenKick-$VERSION-Setup.exe"
