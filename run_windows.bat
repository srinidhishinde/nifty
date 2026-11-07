@echo off
setlocal
cd /d "%~dp0"

if not exist "requirements.txt" (
  echo ERROR: requirements.txt not found. Run this script from the repository root.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo ERROR: Python 3 is required. Install Python 3.10 or newer.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: Dependency installation failed.
  pause
  exit /b 1
)

echo Running UAT...
python scripts/run_uat.py
if errorlevel 1 (
  echo ERROR: UAT failed. Review the output above.
  pause
  exit /b 1
)

echo Starting Streamlit...
python -m streamlit run app/main.py
pause
