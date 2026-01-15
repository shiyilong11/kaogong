# Packaging `11.py` into a standalone executable

This document explains how to build standalone executables of the Streamlit app `11.py` using PyInstaller, and how to use the provided GitHub Actions workflow to produce release artifacts.

Important: bundling Streamlit apps may produce large artifacts and may not include some dynamic assets. The produced binary runs the Python app directly; Streamlit features that expect an interactive development environment should still work but test thoroughly.

Local build (Windows)
```powershell
# install Python 3.11 and pip
python -m pip install --upgrade pip
pip install pyinstaller
pip install -r requirements.txt

# from repo root
cd <repo-root>
pyinstaller --noconfirm --onefile --name kaogong.exe 11.py
# artifact -> dist\kaogong.exe
```

Local build (Linux / WSL)
```bash
python -m pip install --upgrade pip
pip install pyinstaller
pip install -r requirements.txt

pyinstaller --noconfirm --onefile --name kaogong 11.py
# artifact -> dist/kaogong
```

Run the binary
- Windows: `dist\kaogong.exe`
- Linux: `./dist/kaogong`

Notes
- If the app relies on data files or assets, include them via PyInstaller `--add-data` options or adjust the code to load from packaged resources.
- Packaging Streamlit apps produces large executables and some platforms (Streamlit sharing) may be easier for rapid iteration.

CI build
- Use the GitHub Actions workflow `/.github/workflows/packaging.yml` via the Actions tab → select workflow → Run workflow (workflow_dispatch). It produces two artifacts: `kaogong-windows.zip` and `kaogong-linux.tar.gz`.
