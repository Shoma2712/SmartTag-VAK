# -*- coding: utf-8 -*-
"""Интерактивный полный пайплайн по блокам 1–6 (с использованием SQLite БД)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from backend.config import DEFAULT_DATABASE_PATH, PROJECT_ROOT
from backend.database import (
    init_database,
    get_connection,
    get_all_articles,
    insert_lda_results_batch,
    insert_vak_results_batch,
)
from backend.stages.clustering import run_clustering
from backend.stages.lda import get_document_topics, run_lda_modeling
from backend.stages.parser import parse_imt_folder
from backend.stages.scraper import run_scraper
from backend.stages.setup import initialize_project
from backend.vak_classifiers.ui_classifier import VakUiClassifier, VakUiConfig


def _ask_yes(prompt: str) -> bool:
    return input(prompt).strip().lower() == "y"


def _load_or_parse_dataset(conn, paths: Dict[str, str], pdf_count: int) -> pd.DataFrame:
    """Load articles from database or parse PDFs if needed."""
    # Проверка if database has articles
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM articles")
    article_count = cursor.fetchone()[0]
    
    if article_count > 0:
        print(f"Найдено статей в базе данных: {article_count}")
        df = get_all_articles(conn)
        print(f"Загружено {len(df)} статей из БД")
        if _ask_yes("\nХотите перепарсить PDF файлы заново? (y/n): "):
            print(f"\nПарсинг {pdf_count} PDF файлов...")
            print("Это может занять много времени!")
            # Очистка existing articles
            cursor.execute("DELETE FROM articles")
            conn.commit()
            return parse_imt_folder(paths["PDF_IMT"], conn)
        return df

    print("Статей в базе данных не найдено. Запуск парсинга PDF файлов...")
    print(f"Парсинг {pdf_count} PDF файлов...")
    print("Это может занять много времени!")
    return parse_imt_folder(paths["PDF_IMT"], conn)


def run_full_pipeline() -> None:
    """Интерактивный запуск: setup → scraper → parser → LDA → clustering → ВАК."""
    print("=" * 80)
    print("SmartTag VAK - Система обработки научных статей")
    print("=" * 80)
    print()

    print("БЛОК 0: Инициализация базы данных")
    print("-" * 80)
    db_path = DEFAULT_DATABASE_PATH
    print(f"База данных: {db_path}")
    
    # Инициализация database if it doesn't exist
    if not db_path.exists():
        print("Создание схемы базы данных...")
        conn = init_database(str(db_path))
        print("База данных инициализирована")
    else:
        print("База данных уже существует")
        conn = get_connection(str(db_path))
    
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
        conn.close()
        return

    pdf_count = len(list(pdf_dir.glob("*.pdf")))

    print("\n" + "=" * 80)
    print("БЛОК 3: Парсинг PDF файлов")
    print("-" * 80)

    df = _load_or_parse_dataset(conn, paths, pdf_count)
    if df.empty:
        print("\nНе удалось извлечь статьи из PDF. Завершение работы.")
        conn.close()
        return
    
    # Проверка articles were written to database
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM articles")
    article_count = cursor.fetchone()[0]
    print(f"Проверка: в базе данных {article_count} статей")


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
        print("Это может занять много времени!")
        try:
            # Загрузка articles from database
            df_lda = get_all_articles(conn)
            modeler = run_lda_modeling(
                df_lda,
                tokens_column="lda_tokens",
                start_topics=start_topics,
                limit_topics=limit_topics,
                output_dir=str(output_dir),
            )
            modeler.df["lda_topic"] = get_document_topics(modeler.lda_model, modeler.corpus)
            
            # Сохранение LDA results to database через метод модели
            # Очистка старых результатов перед записью новых (UNIQUE на article_id)
            conn.cursor().execute("DELETE FROM lda_results")
            conn.commit()
            count = modeler.save_results_to_database(conn)
            
            # Проверка LDA results were written to database
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM lda_results")
            lda_count = cursor.fetchone()[0]
            print(f"Проверка: в базе данных {lda_count} LDA результатов")
            
            print(f"\nLDA моделирование завершено!")
            print(f"Оптимальное количество тем: {modeler.best_num_topics}")
            print(f"Coherence (c_v): {modeler.best_coherence:.4f}")
            print(f"Визуализации сохранены в: {output_dir}")
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
            # Передача соединения с базой данных вместо DataFrame
            _, df_clustered, best_k, best_model, best_text_view = run_clustering(
                conn,
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
            print(f"Визуализации сохранены в: {output_dir}")
            
            # Проверка clustering results were written to database
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM clustering_results")
            clustering_count = cursor.fetchone()[0]
            print(f"Проверка: в базе данных {clustering_count} результатов кластеризации")
            
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
        print("\nЗапуск классификации ВАК...")
        print("Первый запуск может занять время (загрузка модели).")
        try:
            clf = VakUiClassifier(VakUiConfig())
            vak_df = clf.classify_dataframe(df_work)

            import json

            def _extract_top1(json_str: str, kind: str) -> tuple:
                """
                Извлекает код и cosine similarity для топ-1 специальности.
                kind == 'tfidf' → ищем запись с rank_tfidf == 1
                kind == 'embed' → ищем запись с rank_semantic == 1
                
                Возвращает:
                    tuple: (code, similarity_score) или ("", 0.0) если не найдено
                """
                try:
                    data = json.loads(json_str)
                except Exception:
                    return ("", 0.0)
                
                for spec in data.get("all_specialties", []):
                    if kind == "tfidf" and spec.get("rank_tfidf") == 1:
                        code = spec.get("code", "")
                        score = spec.get("tfidf_similarity", 0.0)
                        return (code, score)
                    if kind == "embed" and spec.get("rank_semantic") == 1:
                        code = spec.get("code", "")
                        score = spec.get("match_score", 0.0)
                        return (code, score)
                return ("", 0.0)

            vak_tfidf_code_series = vak_df["vak_ui_json"].apply(
                lambda x: _extract_top1(x, "tfidf")[0]
            )
            vak_embed_code_series = vak_df["vak_ui_json"].apply(
                lambda x: _extract_top1(x, "embed")[0]
            )
            
            # Сохранение VAK results to database с флагами и cosine similarity
            vak_results = []
            for _, row in vak_df.iterrows():
                # Получение article_id by title
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM articles WHERE title = ?", (row['title'],))
                result = cursor.fetchone()
                if result:
                    # Извлекаем флаги и similarity scores из JSON
                    try:
                        vak_ui_data = json.loads(row.get('vak_ui_json', '{}'))
                        flags = vak_ui_data.get('flags', {})
                        is_ambiguous = 1 if flags.get('ambiguous', False) else 0
                        has_discrepancy = 1 if flags.get('discrepancy', False) else 0
                    except Exception:
                        # Если не удалось распарсить, устанавливаем 0
                        is_ambiguous = 0
                        has_discrepancy = 0
                    
                    # Извлекаем коды и cosine similarity
                    tfidf_code, tfidf_cosine = _extract_top1(row.get('vak_ui_json', ''), 'tfidf')
                    embed_code, embed_cosine = _extract_top1(row.get('vak_ui_json', ''), 'embed')
                    
                    vak_results.append({
                        'article_id': result[0],
                        'vak_embed_code': embed_code,
                        'embed_cosine': embed_cosine,
                        'vak_tfidf_code': tfidf_code,
                        'tfidf_cosine': tfidf_cosine,
                        'is_ambiguous': is_ambiguous,
                        'has_discrepancy': has_discrepancy
                    })
            
            if vak_results:
                # Очистка старых результатов перед записью новых (UNIQUE на article_id)
                conn.cursor().execute("DELETE FROM vak_results")
                conn.commit()
                count = insert_vak_results_batch(conn, vak_results)
                print(f"\nРезультаты ВАК сохранены в БД: {count} записей")
                
                # Статистика по флагам
                ambiguous_count = sum(1 for r in vak_results if r['is_ambiguous'] == 1)
                discrepancy_count = sum(1 for r in vak_results if r['has_discrepancy'] == 1)
                print(f"Неоднозначных классификаций: {ambiguous_count} ({ambiguous_count/count*100:.1f}%)")
                print(f"С расхождением между методами: {discrepancy_count} ({discrepancy_count/count*100:.1f}%)")
            
            # Проверка VAK results were written to database
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vak_results")
            vak_count = cursor.fetchone()[0]
            print(f"Проверка: в базе данных {vak_count} VAK результатов")
            
            print(f"\nКлассификация ВАК завершена")
        except Exception as exc:
            print(f"\nОшибка при классификации ВАК: {exc}")
    else:
        print("Пропускаем классификацию ВАК.")

    # Создание таблиц распределения кластеров и VAK
    print("\n" + "=" * 80)
    print("БЛОК 7: Аналитика - распределение кластеров и VAK")
    print("=" * 80)
    
    if _ask_yes("Создать таблицы распределения кластеров и VAK? (y/n): "):
        try:
            from backend.stages.analytics import create_cluster_vak_distributions
            
            # Переоткрываем соединение для создания таблиц
            conn = get_connection(str(db_path))
            
            if create_cluster_vak_distributions(conn):
                print("\nТаблицы cluster_vak_tfidf_distribution и cluster_vak_embed_distribution созданы")
            else:
                print("\nНе удалось создать таблицы распределения")
            
            conn.close()
        except Exception as e:
            print(f"\nОшибка при создании таблиц распределения: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Пропускаем создание таблиц распределения.")

    # Закрытие database connection
    print("\n" + "=" * 80)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 80)
    print("\nВсе данные сохранены в базе данных:")
    print(f"  База данных: {db_path}")
    print(f"  Визуализации LDA: {data_root / 'lda_results'}")
    print(f"  Визуализации кластеризации: {data_root / 'clustering_results'}")
    print("\nВсе операции завершены успешно!")
