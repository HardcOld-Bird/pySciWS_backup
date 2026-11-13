@echo off
rem init-deps.cmd - bootstrap the project-local toolchain. Thin wrapper.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0init-deps.ps1" %*
exit /b %errorlevel%
