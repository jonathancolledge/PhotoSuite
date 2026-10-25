@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem Copyright (C) 2026 Jonathan Colledge
rem Starts RAW Demosaic with the suite's Python environment, without a console window.
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0RAWDemosaic\desktop\rawdemosaic_desktop.py" %*
