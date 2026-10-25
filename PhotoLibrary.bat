@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem Copyright (C) 2026 Jonathan Colledge
rem Starts Photo Library with the suite's Python environment, without a console window.
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0PhotoLibrary\desktop\photolibrary_desktop.py" %*
