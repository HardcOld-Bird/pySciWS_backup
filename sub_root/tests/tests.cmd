@echo off
rem tests.cmd - one entry for the mechanical layer. Thin wrapper.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tests.ps1" %*
exit /b %errorlevel%
