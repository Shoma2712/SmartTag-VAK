# -*- coding: utf-8 -*-
"""Загрузка статей из SQLite базы данных."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from backend.config import DEFAULT_DATABASE_PATH
from backend.database import get_connection, get_all_articles, get_article_by_title, get_article_by_id


def load_dataset(db_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load all articles from the database.
    
    Args:
        db_path: Path to SQLite database file (default: from config)
    
    Returns:
        pandas.DataFrame: DataFrame with all articles
    
    Raises:
        FileNotFoundError: if database file does not exist
    """
    path = Path(db_path) if db_path else DEFAULT_DATABASE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден: {path}")
    
    conn = get_connection(str(path))
    try:
        df = get_all_articles(conn)
        return df
    finally:
        conn.close()


def find_article(
    df: Optional[pd.DataFrame] = None,
    *,
    title: Optional[str] = None,
    index: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> pd.Series:
    """
    Find an article by title or index.
    
    Args:
        df: Optional DataFrame to search in (if None, loads from database)
        title: Article title to search for
        index: Article index (row number) to retrieve
        db_path: Path to SQLite database file (default: from config)
    
    Returns:
        pandas.Series: Article data
    
    Raises:
        KeyError: if article not found
        ValueError: if neither title nor index specified
    """
    if df is None:
        df = load_dataset(db_path)
    
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
