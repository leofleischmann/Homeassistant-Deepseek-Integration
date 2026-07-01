@echo off
setlocal
REM Version bump + CHANGELOG draft. Release notes: CHANGELOG.md section for this version.
REM Usage: bump.bat 1.3.1

if "%~1"=="" (
  echo Usage: bump.bat ^<version^>
  echo Example: bump.bat 1.3.1
  exit /b 1
)

python "%~dp0scripts\bump.py" %1
exit /b %ERRORLEVEL%
