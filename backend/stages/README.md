# Этапы пайплайна (`backend.stages`)

Самодостаточные модули SmartTag VAK. **Не импортируют** папку `blocks/` и `modules/` — код живёт здесь, логика согласована с `main.ipynb`.

| Этап | Пакет | README |
|------|--------|--------|
| 1. Инициализация | `backend.stages.setup` | [setup/README.md](setup/README.md) |
| 2. Скрейпер PDF | `backend.stages.scraper` | [scraper/README.md](scraper/README.md) |
| 3. Парсер PDF | `backend.stages.parser` | [parser/README.md](parser/README.md) |
| 4. LDA | `backend.stages.lda` | [lda/README.md](lda/README.md) |
| 5. Кластеризация | `backend.stages.clustering` | [clustering/README.md](clustering/README.md) |
| 6. ВАК | `backend.vak_classifiers` | [../vak_classifiers/README.md](../vak_classifiers/README.md) |

Полный интерактивный запуск: `python main.py` или `python -m backend.main pipeline`.

Данные: `project_data/` в **корне репозитория** (не в `blocks/project_data`).
