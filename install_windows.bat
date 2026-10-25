@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem Copyright (C) 2026 Jonathan Colledge
rem Photo Suite - double-click to install (or update). Options are passed on, e.g. install_windows.bat -Models
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
echo.
pause
