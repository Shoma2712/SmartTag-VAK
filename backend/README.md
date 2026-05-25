# Backend SmartTag VAK

## Описание

Серверная логика проекта: справочник ВАК, загрузка статей из CSV (`project_data/dataset_IMT.csv`), классификаторы.

## Структура

| Папка / файл | Назначение |
|--------------|------------|
| `config.py` | Пути к CSV и корню проекта |
| `main.py` | Полный пайплайн (`python backend/main.py` или PyCharm) |
| `vak_data/` | Описания ВАК, сборка текста, загрузка CSV |
| `vak_classifiers/` | UI-классификатор (prod) и гибридный (эксперимент) |
| `stages/` | Самодостаточные этапы: setup, scraper, parser, LDA, clustering |
| `pipeline/` | Полный интерактивный пайплайн по всем блокам |

Подробнее: [stages/README.md](stages/README.md), [vak_data/README.md](vak_data/README.md), [vak_classifiers/README.md](vak_classifiers/README.md), [pipeline/README.md](pipeline/README.md).

## Запуск

Из корня проекта (активирован `.venv`):

```bash
python main.py
python backend/main.py
python -m backend
```


