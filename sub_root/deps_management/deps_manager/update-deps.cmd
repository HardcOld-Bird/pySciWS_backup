@echo off
rem update-deps.cmd - unified dependency entry. Thin wrapper.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update-deps.ps1" %*
exit /b %errorlevel%
