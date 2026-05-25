# -*- coding: utf-8 -*-
"""Пути проекта (данные из CSV, не из БД)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_CSV = PROJECT_ROOT / "project_data" / "dataset_IMT.csv"
CLUSTERED_CSV = PROJECT_ROOT / "project_data" / "analysis_topics" / "articles_with_clusters.csv"
