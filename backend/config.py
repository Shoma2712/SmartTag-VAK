# -*- coding: utf-8 -*-
"""Пути проекта (данные из SQLite БД)."""
from __future__ import annotations

import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Путь к базе данных SQLite (основное хранилище данных)
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "project_data" / "smarttag_vak.db"

# УСТАРЕЛО: Пути к CSV (сохранены для обратной совместимости при миграции)
DEFAULT_DATASET_CSV = PROJECT_ROOT / "project_data" / "dataset_IMT.csv"
CLUSTERED_CSV = PROJECT_ROOT / "project_data" / "analysis_topics" / "articles_with_clusters.csv"

# Предупреждение при использовании CSV констант
def _warn_csv_deprecated():
    warnings.warn(
        "CSV file paths are deprecated. Use DEFAULT_DATABASE_PATH instead.",
        DeprecationWarning,
        stacklevel=3
    )
