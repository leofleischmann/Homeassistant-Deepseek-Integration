@echo off
setlocal
REM Testet die Integration gegen ein ECHTES Home Assistant in .venv-ha.
REM Die normale Suite laeuft gegen Stubs; das hier deckt ab, was nur der
REM echte Kern zeigt (Schema-Bibliothek, llm-Helper, ToolInput).
REM
REM Usage: hatest.bat                - neuestes Home Assistant
REM        hatest.bat ^<version^>      - beliebiges Release, z.B. 2026.9.0
REM        hatest.bat skip-setup     - nur Tests, venv unveraendert

if /I "%~1"=="-h"     goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="/?"     goto :usage
if /I "%~1"=="skip-setup" goto :run

python "%~dp0scripts\ha_testenv.py" %1
if errorlevel 1 exit /b 1

:run
if not exist "%~dp0.venv-ha\Scripts\python.exe" (
  echo .venv-ha fehlt - erst "hatest.bat" ohne skip-setup laufen lassen.
  exit /b 1
)

echo.
echo === Suite gegen echtes Home Assistant ===
"%~dp0.venv-ha\Scripts\python.exe" -m unittest discover -s "%~dp0tests" -v
exit /b %ERRORLEVEL%

:usage
echo Usage: hatest.bat [^<version^> ^| skip-setup]
echo   (ohne Argument)  neuestes Home Assistant installieren und testen
echo   ^<version^>        ein bestimmtes Release, z.B. 2026.9.0
echo   skip-setup       nur die Tests, .venv-ha unveraendert lassen
exit /b 0
