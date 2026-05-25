# -*- coding: utf-8 -*-
"""
Точка входа backend: полный интерактивный пайплайн (блоки 1–6).

Если в PyCharm указан backend/main.py — запускается пайплайн, не CLI.

CLI (info / classify): python -m backend.cli
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.pipeline.runner import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
