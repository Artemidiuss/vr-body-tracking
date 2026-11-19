@echo off
chcp 65001 >nul
rem FBT on Windows: starts the hub and opens its window (Edge app mode).
rem The hub runs in this console: close it to quit FBT.
title FBT
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Спершу запусти windows\setup.cmd
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m fbtapp --browser
