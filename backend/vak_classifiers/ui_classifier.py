# -*- coding: utf-8 -*-
"""
Бэкенд классификации ВАК для UI (отдельно от vak_embedding_classifier).

Ранжирование top-3: TF-IDF (полные описания ВАК) по тексту meta + фрагмент main_text.
match_score по умолчанию — min–max по 5 ВАК (0–1); сырой cosine: score_mode='cosine'.
Дополнительно: cosine эмбеддингов (короткие описания ВАК, только meta) для «Подробнее» и флагов.

Существующие модули не изменяются; описания ВАК импортируются из справочника.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

ScoreMode = Literal["cosine", "minmax"]

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.vak_data.descriptions import VAK_DESCRIPTIONS
from backend.vak_data.texts import build_article_text

VAK_NAMES: List[str] = list(VAK_DESCRIPTIONS.keys())


def vak_code(full_name: str) -> str:
    m = re.match(r"^(\d+\.\d+\.\d+)", full_name)
    return m.group(1) if m else full_name


@dataclass
class VakUiConfig:
    embed_model_name: str = "mlsa-iai-msu-lab/sci-rus-tiny"
    main_text_field: str = "main_text"
    main_text_max_chars: int = 3000
    meta_fields: tuple = ("title", "annotation", "keywords")
    field_weights: Optional[Dict[str, int]] = None
    vak_embed_max_sentences: int = 4
    tfidf_max_features: int = 40_000
    # cosine: сырой TF-IDF cosine (как semantic_similarity у embed)
    # minmax: прежняя шкала 0–1, top-1 всегда 1.0
    score_mode: ScoreMode = "minmax"
    ambiguous_gap: float = 0.02
    ambiguous_gap_minmax: float = 0.08
    embed_alternative_min_gap: float = 0.05
    top_k: int = 3


@dataclass
class SpecialtyScore:
    vak_key: str
    code: str
    title: str
    match_score: float
    semantic_similarity: float
    rank_tfidf: int
    rank_semantic: int
    match_score_normalized: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UiFlags:
    ambiguous: bool
    ambiguous_gap: float
    tfidf_top1_code: str
    semantic_top1_code: str
    discrepancy: bool
    embed_alternative_code: Optional[str] = None
    embed_alternative_title: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VakUiResult:
    top3: List[SpecialtyScore]
    all_specialties: List[SpecialtyScore]
    flags: UiFlags
    text_preview_tfidf: str = ""
    text_preview_semantic: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top3": [s.to_dict() for s in self.top3],
            "all_specialties": [s.to_dict() for s in self.all_specialties],
            "flags": self.flags.to_dict(),
            "text_preview_tfidf": self.text_preview_tfidf,
            "text_preview_semantic": self.text_preview_semantic,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, **kwargs)


def _minmax_row(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def _vak_short_docs(max_sentences: int) -> List[str]:
    out: List[str] = []
    for name in VAK_NAMES:
        parts = [p.strip() for p in VAK_DESCRIPTIONS[name].split(". ") if p.strip()]
        body = ". ".join(parts[:max_sentences])
        out.append(f"{name}. {body}")
    return out


def _vak_full_docs() -> List[str]:
    return [f"{name}. {VAK_DESCRIPTIONS[name]}" for name in VAK_NAMES]


def build_tfidf_input(
    row: Union[pd.Series, Dict[str, Any]],
    config: VakUiConfig,
) -> str:
    """Meta (с весами) + начало main_text для TF-IDF."""
    if config.field_weights is None:
        weights = {"title": 3, "annotation": 2, "keywords": 2}
    else:
        weights = config.field_weights

    if isinstance(row, dict):
        row = pd.Series(row)

    base = build_article_text(row, config.meta_fields, weights)
    main = str(row.get(config.main_text_field, "") or "").strip()
    if config.main_text_max_chars > 0 and main:
        base = f"{base} {main[: config.main_text_max_chars]}".strip()
    return base


def build_semantic_input(
    row: Union[pd.Series, Dict[str, Any]],
    config: VakUiConfig,
) -> str:
    """Только meta для эмбеддингов (main_text ухудшает отделение классов)."""
    if config.field_weights is None:
        weights = {"title": 3, "annotation": 2, "keywords": 2}
    else:
        weights = config.field_weights
    if isinstance(row, dict):
        row = pd.Series(row)
    return build_article_text(row, config.meta_fields, weights)


class VakUiClassifier:
    """Классификатор ВАК для UI: TF-IDF ranking + embed для подсказок."""

    def __init__(self, config: Optional[VakUiConfig] = None) -> None:
        self.config = config or VakUiConfig()
        if self.config.field_weights is None:
            self.config.field_weights = {"title": 3, "annotation": 2, "keywords": 2}
        self._embed_model = None
        self._vak_short = _vak_short_docs(self.config.vak_embed_max_sentences)
        self._vak_full = _vak_full_docs()
        self._vak_emb_matrix: Optional[np.ndarray] = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(self.config.embed_model_name)
            self._vak_emb_matrix = self._embed_model.encode(
                self._vak_short,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return self._embed_model

    def _semantic_raw(self, semantic_text: str) -> np.ndarray:
        model = self._get_embed_model()
        vec = model.encode(
            [semantic_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        assert self._vak_emb_matrix is not None
        return cosine_similarity(vec, self._vak_emb_matrix)[0]

    def _tfidf_raw(self, tfidf_text: str, corpus_tfidf_texts: Optional[List[str]] = None) -> np.ndarray:
        docs = corpus_tfidf_texts if corpus_tfidf_texts else [tfidf_text]
        vec = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        matrix = vec.fit_transform(docs + self._vak_full)
        n = len(docs)
        return cosine_similarity(matrix[:n], matrix[n:])[0]

    def _tfidf_ui_scores(self, tfidf_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(скоры для UI/ранжирования, min-max для справки в «Подробнее»)."""
        raw = np.asarray(tfidf_raw, dtype=float)
        return raw, _minmax_row(raw)

    def _ambiguous_threshold(self) -> float:
        cfg = self.config
        if cfg.score_mode == "minmax":
            return cfg.ambiguous_gap_minmax
        return cfg.ambiguous_gap

    def _build_flags(
        self,
        match_scores: np.ndarray,
        semantic_raw: np.ndarray,
    ) -> UiFlags:
        cfg = self.config
        order_t = np.argsort(match_scores)[::-1]
        order_s = np.argsort(semantic_raw)[::-1]
        gap = float(match_scores[order_t[0]] - match_scores[order_t[1]]) if len(order_t) > 1 else 1.0
        ambiguous = gap < self._ambiguous_threshold()

        code_t = vak_code(VAK_NAMES[order_t[0]])
        code_s = vak_code(VAK_NAMES[order_s[0]])
        discrepancy = code_t != code_s

        alt_code: Optional[str] = None
        alt_title: Optional[str] = None
        message: Optional[str] = None

        if discrepancy:
            sem_gap = float(
                semantic_raw[order_s[0]] - semantic_raw[order_s[1]]
            ) if len(order_s) > 1 else 0.0
            if sem_gap >= cfg.embed_alternative_min_gap:
                alt_key = VAK_NAMES[order_s[0]]
                alt_code = vak_code(alt_key)
                alt_title = alt_key.split(". ", 1)[-1] if ". " in alt_key else alt_key
                message = (
                    f"Смысловая модель также предлагает рассмотреть {alt_code} "
                    f"({alt_title})."
                )

        if ambiguous and not message:
            message = (
                "Результат неоднозначный: близкие оценки соответствия "
                "по формулировкам ВАК. Рекомендуется проверить вручную."
            )
        elif ambiguous and message:
            message += " Также близкие оценки по TF-IDF."

        return UiFlags(
            ambiguous=ambiguous,
            ambiguous_gap=round(gap, 4),
            tfidf_top1_code=code_t,
            semantic_top1_code=code_s,
            discrepancy=discrepancy,
            embed_alternative_code=alt_code if discrepancy else None,
            embed_alternative_title=alt_title if discrepancy else None,
            message=message,
        )

    def _pack_scores(
        self,
        tfidf_raw: np.ndarray,
        semantic_raw: np.ndarray,
    ) -> List[SpecialtyScore]:
        ui_scores, norm_scores = self._tfidf_ui_scores(tfidf_raw)
        if self.config.score_mode == "minmax":
            display_scores = norm_scores
        else:
            display_scores = ui_scores

        order_t = np.argsort(display_scores)[::-1]
        order_s = np.argsort(semantic_raw)[::-1]
        rank_t = {int(i): r + 1 for r, i in enumerate(order_t)}
        rank_s = {int(i): r + 1 for r, i in enumerate(order_s)}

        items: List[SpecialtyScore] = []
        for i, name in enumerate(VAK_NAMES):
            title = name.split(". ", 1)[-1] if ". " in name else name
            items.append(
                SpecialtyScore(
                    vak_key=name,
                    code=vak_code(name),
                    title=title,
                    match_score=round(float(display_scores[i]), 4),
                    semantic_similarity=round(float(semantic_raw[i]), 4),
                    rank_tfidf=rank_t[i],
                    rank_semantic=rank_s[i],
                    match_score_normalized=round(float(norm_scores[i]), 4),
                )
            )
        return items

    def classify_text(
        self,
        text_tfidf: str,
        text_semantic: Optional[str] = None,
    ) -> VakUiResult:
        """Классификация произвольного текста (для UI: один ввод пользователя)."""
        text_semantic = text_semantic if text_semantic is not None else text_tfidf
        tfidf_raw = self._tfidf_raw(text_tfidf)
        semantic_raw = self._semantic_raw(text_semantic)
        ui_scores, _ = self._tfidf_ui_scores(tfidf_raw)
        all_scores = self._pack_scores(tfidf_raw, semantic_raw)
        all_sorted = sorted(all_scores, key=lambda s: s.match_score, reverse=True)
        top3 = all_sorted[: self.config.top_k]
        flags = self._build_flags(ui_scores, semantic_raw)
        preview_len = 200
        return VakUiResult(
            top3=top3,
            all_specialties=all_sorted,
            flags=flags,
            text_preview_tfidf=text_tfidf[:preview_len],
            text_preview_semantic=text_semantic[:preview_len],
        )

    def classify_row(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        corpus_tfidf_texts: Optional[List[str]] = None,
    ) -> VakUiResult:
        tfidf_text = build_tfidf_input(row, self.config)
        semantic_text = build_semantic_input(row, self.config)
        if corpus_tfidf_texts is None:
            tfidf_raw = self._tfidf_raw(tfidf_text)
        else:
            idx = corpus_tfidf_texts.index(tfidf_text)
            vec = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=(1, 2),
                min_df=1,
            )
            matrix = vec.fit_transform(corpus_tfidf_texts + self._vak_full)
            n = len(corpus_tfidf_texts)
            tfidf_all = cosine_similarity(matrix[:n], matrix[n:])
            tfidf_raw = tfidf_all[idx]
        semantic_raw = self._semantic_raw(semantic_text)
        ui_scores, _ = self._tfidf_ui_scores(tfidf_raw)
        all_scores = self._pack_scores(tfidf_raw, semantic_raw)
        all_sorted = sorted(all_scores, key=lambda s: s.match_score, reverse=True)
        top3 = all_sorted[: self.config.top_k]
        flags = self._build_flags(ui_scores, semantic_raw)
        preview_len = 200
        return VakUiResult(
            top3=top3,
            all_specialties=all_sorted,
            flags=flags,
            text_preview_tfidf=tfidf_text[:preview_len],
            text_preview_semantic=semantic_text[:preview_len],
        )

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Пакетная классификация; добавляет колонки для UI и JSON-поле vak_ui."""
        cfg = self.config
        tfidf_texts = [build_tfidf_input(row, cfg) for _, row in df.iterrows()]
        semantic_texts = [build_semantic_input(row, cfg) for _, row in df.iterrows()]

        vec = TfidfVectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        matrix = vec.fit_transform(tfidf_texts + self._vak_full)
        n = len(tfidf_texts)
        tfidf_all = cosine_similarity(matrix[:n], matrix[n:])

        self._get_embed_model()
        assert self._embed_model is not None
        sem_vecs = self._embed_model.encode(
            semantic_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        assert self._vak_emb_matrix is not None
        semantic_all = cosine_similarity(sem_vecs, self._vak_emb_matrix)

        out = df.copy()
        top_codes: List[str] = []
        top_scores: List[float] = []
        flags_amb: List[bool] = []
        flags_disc: List[bool] = []
        flags_msg: List[str] = []
        ui_json: List[str] = []

        for i in range(len(df)):
            tfidf_raw = tfidf_all[i]
            semantic_raw = semantic_all[i]
            ui_scores, _ = self._tfidf_ui_scores(tfidf_raw)
            packed = self._pack_scores(tfidf_raw, semantic_raw)
            result = VakUiResult(
                top3=sorted(packed, key=lambda s: s.match_score, reverse=True)[: cfg.top_k],
                all_specialties=sorted(
                    packed, key=lambda s: s.match_score, reverse=True
                ),
                flags=self._build_flags(ui_scores, semantic_raw),
                text_preview_tfidf=tfidf_texts[i][:200],
                text_preview_semantic=semantic_texts[i][:200],
            )
            top_codes.append(result.top3[0].code if result.top3 else "")
            top_scores.append(result.top3[0].match_score if result.top3 else 0.0)
            flags_amb.append(result.flags.ambiguous)
            flags_disc.append(result.flags.discrepancy)
            flags_msg.append(result.flags.message or "")
            ui_json.append(result.to_json())

        out["vak_ui_top1_code"] = top_codes
        out["vak_ui_top1_match_score"] = top_scores
        out["vak_ui_ambiguous"] = flags_amb
        out["vak_ui_discrepancy"] = flags_disc
        out["vak_ui_message"] = flags_msg
        out["vak_ui_json"] = ui_json
        return out


def classify_for_ui(
    df: pd.DataFrame,
    main_text_max_chars: int = 3000,
) -> pd.DataFrame:
    """Удобная обёртка для ноутбука."""
    clf = VakUiClassifier(VakUiConfig(main_text_max_chars=main_text_max_chars))
    return clf.classify_dataframe(df)
