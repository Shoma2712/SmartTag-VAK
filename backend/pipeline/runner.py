# -*- coding: utf-8 -*-
"""Интерактивный полный пайплайн по блокам 1–6 (без blocks/modules)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from backend.config import DEFAULT_DATASET_CSV, PROJECT_ROOT
from backend.stages.clustering import run_clustering
from backend.stages.lda import get_document_topics, run_lda_modeling
from backend.stages.parser import parse_imt_folder
from backend.stages.scraper import run_scraper
from backend.stages.setup import initialize_project
from backend.vak_classifiers.ui_classifier import VakUiClassifier, VakUiConfig


def _ask_yes(prompt: str) -> bool:
    return input(prompt).strip().lower() == "y"


def _load_or_parse_dataset(paths: Dict[str, str], pdf_count: int) -> pd.DataFrame:
    output_csv = Path(paths["OUTPUT_IMT"])
    if output_csv.exists():
        print(f"Найден датасет: {output_csv}")
        print("Загрузка данных из CSV...")
        df = pd.read_csv(output_csv)
        print(f"Загружено {len(df)} статей")
        if _ask_yes("\nХотите перепарсить PDF файлы заново? (y/n): "):
            print(f"\nПарсинг {pdf_count} PDF файлов...")
            print("Это может занять много времени!")
            return parse_imt_folder(paths["PDF_IMT"], str(output_csv))
        return df

    print("Датасет не найден. Запуск парсинга PDF файлов...")
    print(f"Парсинг {pdf_count} PDF файлов...")
    print("Это может занять много времени!")
    return parse_imt_folder(paths["PDF_IMT"], str(output_csv))


def run_full_pipeline() -> None:
    """Интерактивный запуск: setup → scraper → parser → LDA → clustering → ВАК."""
    print("=" * 80)
    print("SmartTag VAK - Система обработки научных статей")
    print("=" * 80)
    print()

    print("БЛОК 1: Инициализация проекта")
    print("-" * 80)
    paths = initialize_project()
    data_root = PROJECT_ROOT / "project_data"

    print("\n" + "=" * 80)
    print("БЛОК 2: Сбор PDF файлов")
    print("-" * 80)

    pdf_dir = Path(paths["PDF_IMT"])
    pdf_count = len(list(pdf_dir.glob("*.pdf")))

    if pdf_count > 0:
        print(f"Найдено {pdf_count} PDF файлов в {pdf_dir}")
        if _ask_yes("\nХотите скачать дополнительные PDF со всех доступных страниц? (y/n): "):
            print("\nЗапуск скрейпера (все доступные страницы)...")
            print("Это может занять много времени!")
            stats = run_scraper(str(pdf_dir), max_pages=50)
            print(f"\n Скачано новых файлов: {stats['files_downloaded']}")
    else:
        print("PDF файлы не найдены.")
        if _ask_yes("Хотите скачать PDF файлы со всех доступных страниц? (y/n): "):
            print("\nЗапуск скрейпера (все доступные страницы)...")
            print("Это может занять много времени!")


    if len(list(pdf_dir.glob("*.pdf"))) == 0:
        print("\nPDF файлы не найдены. Завершение работы.")
        return

    pdf_count = len(list(pdf_dir.glob("*.pdf")))

    print("\n" + "=" * 80)
    print("БЛОК 3: Парсинг PDF файлов")
    print("-" * 80)

    df = _load_or_parse_dataset(paths, pdf_count)
    if df.empty:
        print("\nНе удалось извлечь статьи из PDF. Завершение работы.")
        return


    print("\n" + "=" * 80)
    print("БЛОК 4: Тематическое моделирование LDA")
    print("-" * 80)

    if _ask_yes("Запустить LDA моделирование? (y/n): "):
        start_raw = input("Минимальное количество тем (по умолчанию 4): ").strip()
        start_topics = int(start_raw) if start_raw.isdigit() else 4
        limit_raw = input("Максимальное количество тем (по умолчанию 20): ").strip()
        limit_topics = int(limit_raw) if limit_raw.isdigit() else 20
        output_dir = data_root / "lda_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nЗапуск LDA моделирования (темы: {start_topics}-{limit_topics})...")
        print(" Это может занять много времени!")
        try:
            df_lda = pd.read_csv(DEFAULT_DATASET_CSV)
            modeler = run_lda_modeling(
                df_lda,
                tokens_column="lda_tokens",
                start_topics=start_topics,
                limit_topics=limit_topics,
                output_dir=str(output_dir),
            )
            modeler.df["lda_topic"] = get_document_topics(modeler.lda_model, modeler.corpus)
            modeler.df.to_csv(
                output_dir / "dataset_with_lda_topics.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(f"\nLDA моделирование завершено!")
            print(f"Оптимальное количество тем: {modeler.best_num_topics}")
            print(f"Coherence (c_v): {modeler.best_coherence:.4f}")
            print(f"Результаты сохранены в: {output_dir}")
        except Exception as exc:
            print(f"\nОшибка при LDA моделировании: {exc}")
            print("Продолжаем без LDA...")
    else:
        print("Пропускаем LDA моделирование.")

    print("\n" + "=" * 80)
    print("БЛОК 5: Кластеризация на эмбеддингах")
    print("-" * 80)

    df_work = df
    if _ask_yes("Запустить кластеризацию? (y/n): "):
        output_dir = data_root / "clustering_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\nЗапуск автоматического подбора модели и представления...")
        print("Это может занять много времени!")
        try:
            _, df_clustered, best_k, best_model, best_text_view = run_clustering(
                df,
                min_k=5,
                max_k=50,
                output_dir=str(output_dir),
                random_state=42,
            )
            df_work = df_clustered
            print(f"\nКластеризация завершена!")
            print(f"Лучшая модель: {best_model}")
            print(f"Лучшее представление: {best_text_view}")
            print(f"Оптимальное количество кластеров: {best_k}")
            print(f"Результаты сохранены в: {output_dir}")
            for cluster_id, count in df_clustered["cluster"].value_counts().sort_index().items():
                print(f"  Кластер {cluster_id}: {count} статей")
        except Exception as exc:
            print(f"\nОшибка при кластеризации: {exc}")
            print("Продолжаем без кластеризации...")
    else:
        print("Пропускаем кластеризацию.")

    print("\n" + "=" * 80)
    print("БЛОК 6: Классификация по специальностям ВАК")
    print("-" * 80)

    if _ask_yes("Запустить классификацию по ВАК? (y/n): "):
        vak_dir = data_root / "vak_results"
        vak_dir.mkdir(parents=True, exist_ok=True)
        print("\nЗапуск классификации ВАК...")
        print("Первый запуск может занять время (загрузка модели).")
        try:
            clf = VakUiClassifier(VakUiConfig())
            vak_df = clf.classify_dataframe(df_work)

            import json

            def _extract_top1(json_str: str, kind: str) -> str:
                """
                kind == 'tfidf' → ищем запись с rank_tfidf == 1
                kind == 'embed' → ищем запись с rank_semantic == 1
                """
                try:
                    data = json.loads(json_str)
                except Exception:
                    return ""  # на всякий случай
                for spec in data.get("all_specialties", []):
                    if kind == "tfidf" and spec.get("rank_tfidf") == 1:
                        return spec.get("code", "")
                    if kind == "embed" and spec.get("rank_semantic") == 1:
                        return spec.get("code", "")
                return ""

            vak_tfidf_code_series = vak_df["vak_ui_json"].apply(
                lambda x: _extract_top1(x, "tfidf")
            )
            vak_embed_code_series = vak_df["vak_ui_json"].apply(
                lambda x: _extract_top1(x, "embed")
            )

            from backend.vak_data.descriptions import VAK_DESCRIPTIONS

            def _code_with_title(code: str) -> str:
                for full_name in VAK_DESCRIPTIONS:
                    if full_name.startswith(code + "."):
                        title = full_name.split(". ", 1)[-1]
                        return f"{code}({title})"
                return code

            vak_tfidf_pred_series = vak_tfidf_code_series.apply(_code_with_title)
            vak_embed_pred_series = vak_embed_code_series.apply(_code_with_title)

            mini_cols = {
                "title": vak_df["title"],
                "annotation": vak_df["annotation"],
                "keywords": vak_df["keywords"],
                "main_text": vak_df["main_text"],
                "vak_tfidf_pred": vak_tfidf_pred_series,
                "vak_embed_pred": vak_embed_pred_series,
            }

            vak_mini = pd.DataFrame(mini_cols)

            vak_path = vak_dir / "articles_with_vak.csv"
            vak_df.to_csv(vak_path, index=False, encoding="utf-8-sig")
            print(f"\nПолный датасет (для frontend) сохранён: {vak_path}")

            mini_path = vak_dir / "articles_with_vak_mini.csv"
            vak_mini.to_csv(mini_path, index=False, encoding="utf-8-sig")
            print(f"Минимальный датасет (для анализа) сохранён: {mini_path}")
            print(f"\nКлассификация ВАК завершена!")
            print(f"Результаты: {vak_path}")
        except Exception as exc:
            print(f"\nОшибка при классификации ВАК: {exc}")
    else:
        print("Пропускаем классификацию ВАК.")

    print("\n" + "=" * 80)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 80)
    print("\nРезультаты сохранены в:")
    print(f" Базовая директория: {data_root}")
    print(f" Парсированные данные: {paths['OUTPUT_IMT']}")
    if (data_root / "lda_results").exists():
        print(f" LDA результаты: {data_root / 'lda_results'}")
    if (data_root / "clustering_results").exists():
        print(f" Кластеризация: {data_root / 'clustering_results'}")
    if (data_root / "vak_results").exists():
        print(f" Классификация ВАК: {data_root / 'vak_results'}")
    print("\nВсе операции завершены успешно!")
