# -*- coding: utf-8 -*-
"""Загрузка статей из CSV-датасета."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from backend.config import DEFAULT_DATASET_CSV


def load_dataset(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else DEFAULT_DATASET_CSV
    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден: {path}")
    return pd.read_csv(path)


def find_article(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    index: Optional[int] = None,
) -> pd.Series:
    if index is not None:
        return df.iloc[int(index)]
    if title:
        mask = df["title"].astype(str).str.strip() == title.strip()
        hits = df[mask]
        if len(hits) == 0:
            raise KeyError(f"Статья с title не найдена: {title[:80]}...")
        return hits.iloc[0]
    raise ValueError("Укажите title или index")


def row_to_fields(row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, str]:
    """Поля для UI-классификатора."""
    if isinstance(row, pd.Series):
        row = row.to_dict()
    return {
        "title": str(row.get("title", "") or ""),
        "annotation": str(row.get("annotation", "") or ""),
        "keywords": str(row.get("keywords", "") or ""),
        "main_text": str(row.get("main_text", "") or ""),
    }
