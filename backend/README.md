# Backend SmartTag VAK

## Описание

Серверная логика проекта: справочник ВАК, работа с базой данных SQLite, классификаторы, тематическое моделирование и кластеризация.

## Структура

| Папка / файл | Назначение |
|--------------|------------|
| `config.py` | Пути к базе данных и корню проекта |
| `database.py` | Схема базы данных SQLite, создание таблиц, индексов |
| `main.py` | Полный пайплайн (`python backend/main.py` или PyCharm) |
| `vak_data/` | Описания ВАК, сборка текста для классификации |
| `vak_classifiers/` | UI-классификатор с двумя методами (TF-IDF + embeddings) |
| `stages/` | Самодостаточные этапы: setup, scraper, parser, LDA, clustering, analytics |
| `pipeline/` | Полный интерактивный пайплайн по всем блокам с сохранением в БД |

Подробнее: [stages/README.md](stages/README.md), [vak_data/README.md](vak_data/README.md), [vak_classifiers/README.md](vak_classifiers/README.md), [pipeline/README.md](pipeline/README.md).

## База данных

Система использует SQLite базу данных (`project_data/smarttag_vak.db`) для централизованного хранения:
- Статей с метаданными
- Результатов тематического моделирования (LDA)
- Результатов кластеризации
- Результатов классификации по специальностям ВАК
- Аналитики распределения специальностей по кластерам

## Запуск

Из корня проекта (активирован `.venv`):

```bash
python main.py
python backend/main.py
python -m backend
```


