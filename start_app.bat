@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please set up the local environment using the README first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
