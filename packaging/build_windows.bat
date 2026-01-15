@echo off
REM Build standalone Windows EXE using PyInstaller
python -m pip install --upgrade pip setuptools wheel
pip install pyinstaller
if exist requirements.txt (
  pip install -r requirements.txt
)
pyinstaller --noconfirm --onefile --name kaogong.exe 11.py
echo Build complete. Output: dist\kaogong.exe
pause
