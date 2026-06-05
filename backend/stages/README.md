# Этапы пайплайна (`backend.stages`)

Самодостаточные модули SmartTag VAK для обработки научных статей и классификации по специальностям ВАК. Все результаты сохраняются в **базу данных SQLite** (`project_data/smarttag_vak.db`).

| Этап | Пакет | README | Таблица БД |
|------|--------|--------|-----------|
| 1. Инициализация | `backend.stages.setup` | [setup/README.md](setup/README.md) | — |
| 2. Скрейпер PDF | `backend.stages.scraper` | [scraper/README.md](scraper/README.md) | — |
| 3. Парсер PDF | `backend.stages.parser` | [parser/README.md](parser/README.md) | `articles` |
| 4. LDA | `backend.stages.lda` | [lda/README.md](lda/README.md) | `lda_results` |
| 5. Кластеризация | `backend.stages.clustering` | [clustering/README.md](clustering/README.md) | `clustering_results` |
| 6. Классификация ВАК | `backend.vak_classifiers` | [../vak_classifiers/README.md](../vak_classifiers/README.md) | `vak_classifications` |
| 7. Аналитика | `backend.stages.analytics` | — | `cluster_vak_distribution_*` |

**Полный интерактивный запуск:** `python main.py` или `python -m backend.main pipeline`.

**Данные:** 
- База данных: `project_data/smarttag_vak.db` (SQLite)
- PDF файлы: `project_data/pdfs/`
- Визуализации: `project_data/lda_results/`, `project_data/clustering_results/`

**Архитектура:** Модульная система с централизованным хранением в БД, внешние ключи обеспечивают целостность данных.


