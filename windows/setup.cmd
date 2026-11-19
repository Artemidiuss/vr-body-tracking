@echo off
chcp 65001 >nul
rem One-time setup on Windows (runs setup.ps1 without changing the system execution policy).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
pause
