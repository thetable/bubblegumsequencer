@echo off
rem Double-click this in Explorer to play. The Windows twin of
rem Bubblegum.command, and the same bargain: no terminal to type into.
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv is missing. It is what builds the Python environment.
  echo.
  echo     powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
  echo.
  echo Then double-click this again.
  pause
  exit /b 1
)

rem The first run downloads Python and the libraries, about half a minute on
rem a new machine. Every run after that finds them already there.
uv sync --quiet
if errorlevel 1 (
  echo.
  echo Could not build the Python environment.
  pause
  exit /b 1
)

rem Anything wrong with the rig is reported by play.py itself, and it stops
rem rather than serving. Hold the window open so that it can be read.
uv run play.py
if errorlevel 1 pause
