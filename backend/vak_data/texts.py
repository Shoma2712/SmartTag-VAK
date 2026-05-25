# -*- coding: utf-8 -*-
"""Сборка текста статьи для классификаторов."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import pandas as pd


def _normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def compress_vak_description(description: str, max_sentences: int = 4) -> str:
    """Сжатие официального описания: первые фрагменты перечня направлений (как в паспорте ВАК)."""
    parts = [p.strip() for p in description.strip().split(". ") if p.strip()]
    if max_sentences and len(parts) > max_sentences:
        parts = parts[:max_sentences]
    return ". ".join(parts)


def build_article_text(
    row: pd.Series,
    fields: Sequence[str] = ("title", "annotation", "keywords"),
    weights: Optional[Dict[str, int]] = None,
    max_chars: int = 8000,
) -> str:
    """Текст статьи для эмбеддинга: метаданные без полного текста."""
    chunks: List[str] = []
    for col in fields:
        val = str(row.get(col, "") or "").strip()
        if not val:
            continue
        repeat = (weights or {}).get(col, 1)
        if repeat != 1:
            val = (val + " ") * int(repeat)
        chunks.append(val)
    text = " ".join(chunks)
    return text[:max_chars] if len(text) > max_chars else text


