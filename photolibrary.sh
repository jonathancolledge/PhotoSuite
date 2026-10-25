#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
# Starts Photo Library with the suite's Python environment.
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
exec "$HERE/.venv-linux/bin/python" "$HERE/PhotoLibrary/desktop/photolibrary_desktop.py" "$@"
