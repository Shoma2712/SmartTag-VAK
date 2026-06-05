# Модуль аналитики (Analytics Stage)

Этот модуль отвечает за создание дополнительных аналитических таблиц и вычисление статистики на основе результатов обработки статей. Все данные хранятся в базе данных SQLite.

## Функциональность

### Таблицы распределения кластеров и VAK специальностей

Модуль создает две таблицы для анализа связи между кластерами статей и специальностями ВАК:

1. **cluster_vak_tfidf_distribution** - распределение для TF-IDF метода классификации
2. **cluster_vak_embed_distribution** - распределение для Embedding метода классификации

Каждая таблица содержит:
- `cluster_name` - название кластера (тематическая рубрика)
- `vak_code` - код специальности ВАК
- `article_count` - количество статей с данной комбинацией кластера и VAK кода
- `percentage` - процент статей с данным VAK кодом в кластере

**Структура таблиц:**
```sql
CREATE TABLE cluster_vak_tfidf_distribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_name TEXT NOT NULL,
    vak_code TEXT NOT NULL,
    article_count INTEGER NOT NULL,
    percentage REAL NOT NULL
)

CREATE TABLE cluster_vak_embed_distribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_name TEXT NOT NULL,
    vak_code TEXT NOT NULL,
    article_count INTEGER NOT NULL,
    percentage REAL NOT NULL
)
```

## Использование

### В пайплайне

```python
from backend.pipeline import run_pipeline

# Аналитика выполняется автоматически на этапе 7
run_pipeline(db_path="project_data/smarttag_vak.db")
```

Или напрямую:

```python
from backend.stages.analytics import create_cluster_vak_distributions
from backend.database import get_connection

conn = get_connection("project_data/smarttag_vak.db")
success = create_cluster_vak_distributions(conn)
if success:
    print("✅ Таблицы распределения созданы")
```

### Примеры SQL запросов

#### 1. Все VAK специальности в кластере (TF-IDF метод)
```sql
SELECT * FROM cluster_vak_tfidf_distribution 
WHERE cluster_name = 'Математическое моделирование' 
ORDER BY article_count DESC;
```

#### 2. Все кластеры для VAK специальности (Embedding метод)
```sql
SELECT * FROM cluster_vak_embed_distribution 
WHERE vak_code LIKE '%1.2.2%' 
ORDER BY article_count DESC;
```

#### 3. Сравнение методов для кластера
```sql
SELECT 'TF-IDF' as method, vak_code, article_count 
FROM cluster_vak_tfidf_distribution 
WHERE cluster_name = 'Программные системы и комплексы'
UNION ALL
SELECT 'Embedding' as method, vak_code, article_count 
FROM cluster_vak_embed_distribution 
WHERE cluster_name = 'Программные системы и комплексы'
ORDER BY method, article_count DESC;
```

#### 4. Топ-3 VAK специальности по кластерам
```sql
SELECT cluster_name, vak_code, article_count, percentage
FROM cluster_vak_tfidf_distribution
WHERE cluster_name IN (
    SELECT DISTINCT cluster_name 
    FROM cluster_vak_tfidf_distribution
)
ORDER BY cluster_name, article_count DESC;
```

## Требования

- Должны быть заполнены таблицы в базе данных `project_data/smarttag_vak.db`:
  - `articles` (статьи с ID)
  - `clustering_results` (результаты кластеризации с article_id)
  - `vak_results` (результаты классификации ВАК с article_id)
- Обе таблицы должны содержать `article_id` для связи со статьями

## Структура модуля

```
analytics/
├── __init__.py                      # Экспорт функций модуля
├── cluster_vak_distribution.py      # Создание таблиц распределения
└── README.md                        # Документация
```

## Логика работы

1. **Создание таблиц** - создаются две таблицы с индексами для быстрого поиска
2. **Очистка данных** - удаляются старые данные перед заполнением
3. **Вычисление распределений** - для каждого метода (TF-IDF и Embedding):
   - Группировка статей по кластерам и VAK кодам
   - Подсчет количества статей в каждой группе
   - Вычисление процентов относительно общего количества статей в кластере
4. **Вывод статистики** - показ топ-3 VAK специальностей для каждого кластера

## Примечания

- Таблицы создаются автоматически при запуске полного пайплайна
- Данные пересчитываются при каждом запуске (старые данные удаляются)
- Используется чистый SQL без pandas для максимальной производительности
- Все данные хранятся в базе данных SQLite `project_data/smarttag_vak.db`
- Индексы создаются автоматически для быстрого поиска по `cluster_name` и `vak_code`
