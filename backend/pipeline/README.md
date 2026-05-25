# Полный Пайплайн (`backend.pipeline`)

## Описание

Модуль запускает **весь цикл обработки** в консоли с интерактивными вопросами на каждом этапе. Использует только `backend.stages.*` и `backend.vak_classifiers` (без импорта из `blocks/`):

1. Инициализация проекта
2. Сбор PDF
3. Парсинг PDF
4. LDA
5. Кластеризация
6. Классификация ВАК

На каждом этапе пользователь выбирает `y/n` — перезапустить блок или пропустить.

## Запуск

```bash
python -m backend.main pipeline
```

или

```bash
python main.py
```

## Файл `runner.py`

### `run_full_pipeline() -> None`

**Вход:** интерактивные ответы пользователя в консоли.

**Выход:** печать статусов в stdout и сохранение CSV/артефактов в `project_data`.

**Что делает:** по очереди вызывает блоки setup/scraper/parser/LDA/clustering/VAK, проверяет наличие уже готовых артефактов, запрашивает подтверждение на перезапуск.

---

### `_ask_yes_no(prompt: str, default: bool = False) -> bool`

**Вход:** текст вопроса и значение по умолчанию.

**Выход:** `True`/`False`.

**Что делает:** единый безопасный ввод `y/n` (поддерживает `да/нет`).

---

### `_load_or_parse_dataset(paths, pdf_count) -> pd.DataFrame`

**Вход:** пути из setup-блока и число PDF.

**Выход:** DataFrame статей.

**Что делает:** если `dataset_IMT.csv` уже есть — предлагает перепарсить; иначе запускает парсер.

---

### `_run_lda_if_requested(df, base_dir) -> Optional[Path]`

**Вход:** DataFrame статей, базовая папка `project_data`.

**Выход:** путь к CSV с LDA-темами или `None`.

**Что делает:** по запросу запускает `prepare_dataset_for_analysis` + `train_lda_model`, сохраняет `dataset_with_lda_topics.csv`.

---

### `_run_clustering_if_requested(df, base_dir) -> Optional[Path]`

**Вход:** DataFrame статей.

**Выход:** путь к `articles_with_clusters.csv` или `None`.

**Что делает:** по запросу запускает `cluster_articles` с автоподбором `k`, сохраняет метрики и CSV.

---

### `_run_vak_if_requested(df, base_dir) -> Optional[Path]`

**Вход:** DataFrame (с кластерами или без).

**Выход:** путь к `articles_with_vak.csv` или `None`.

**Что делает:** запускает `VakUiClassifier.classify_dataframe`, сохраняет top-1 и JSON по всем 5 специальностям ВАК.
