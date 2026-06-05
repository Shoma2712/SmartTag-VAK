# Блок 4: Тематическое моделирование LDA

## Описание

Этот блок отвечает за тематическое моделирование научных статей с помощью алгоритма LDA (Latent Dirichlet Allocation). Блок автоматически определяет оптимальное количество тем, обучает модель и визуализирует результаты.

## Основные функции

### `ensure_tokens(x) -> list`

Преобразует строковое представление списка в список.

**Параметры:**
- `x`: Значение (может быть list или str)

**Возвращает:**
- Список токенов

**Что делает:**
- Если значение уже является списком — возвращает его
- Если значение — строка, пытается преобразовать её через `ast.literal_eval()`
- Если преобразование не удалось — возвращает пустой список

---

### `build_ngrams(docs, min_count_bigram=5, threshold_bigram=20, min_count_trigram=3, threshold_trigram=30) -> tuple`

Строит биграммы и триграммы из документов.

**Параметры:**
- `docs` (list): Список документов (каждый документ — список токенов)
- `min_count_bigram` (int): Минимальная частота для биграмм (по умолчанию 5)
- `threshold_bigram` (int): Порог для биграмм (по умолчанию 20)
- `min_count_trigram` (int): Минимальная частота для триграмм (по умолчанию 3)
- `threshold_trigram` (int): Порог для триграмм (по умолчанию 30)

**Возвращает:**
- Кортеж `(документы_с_ngramмами, bigram_модель, trigram_модель)`

**Что делает:**
- Обучает модель Phrases для биграмм
- Обучает модель Phrases для триграмм (на основе биграмм)
- Применяет модели к документам, создавая составные токены (например, «машинное_обучение»)

---

### `compute_coherence_grid(dictionary, corpus, texts, start=4, limit=20, step=1) -> tuple`

Вычисляет coherence для разного количества тем.

**Параметры:**
- `dictionary`: Словарь gensim
- `corpus`: Корпус документов
- `texts`: Тексты для вычисления coherence
- `start` (int): Начальное количество тем
- `limit` (int): Максимальное количество тем
- `step` (int): Шаг

**Возвращает:**
- Кортеж `(список_количеств_тем, список_моделей, список_coherence)`

**Что делает:**
- Для каждого k от start до limit с шагом step:
  1. Обучает модель LdaMulticore (4 воркера, 15 проходов, 300 итераций)
  2. Вычисляет Coherence (c_v метрика)
  3. Сохраняет модель и значение coherence
- Выводит прогресс обучения

**Пример вывода:**
```
Обучение LDA: 4 тем
  coherence(c_v)=0.4523
Обучение LDA: 5 тем
  coherence(c_v)=0.4876
...
```

---

## Класс `LDATopicModeler`

Класс для тематического моделирования с помощью LDA.

### `LDATopicModeler(df, tokens_column='lda_tokens', max_tokens_per_doc=800)`

Инициализация модели LDA.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `tokens_column` (str): Название колонки с токенами (по умолчанию 'lda_tokens')
- `max_tokens_per_doc` (int): Максимальное количество токенов на документ (по умолчанию 800)

**Что делает при инициализации:**
- Копирует DataFrame
- Преобразует токены через `ensure_tokens()`
- Фильтрует документы (оставляет те, где >20 токенов)
- Ограничивает количество токенов на документ

---

### `build_dictionary_and_corpus(no_below=5, no_above=0.6, keep_n=20000) -> tuple`

Строит словарь и корпус для LDA.

**Параметры:**
- `no_below` (int): Минимальная частота слова в документах (по умолчанию 5)
- `no_above` (float): Максимальная доля документов со словом (по умолчанию 0.6)
- `keep_n` (int): Максимальный размер словаря (по умолчанию 20000)

**Возвращает:**
- Кортеж `(словарь, корпус, тексты_с_ngramмами)`

**Что делает:**
1. Строит биграммы и триграммы через `build_ngrams()`
2. Создает словарь gensim (id2word)
3. Фильтрует словарь (удаляет слишком редкие и слишком частые слова)
4. Создает корпус (bag-of-words представление)

**Пример вывода:**
```
Подготовка данных...
Статей для LDA: 312
Построение биграмм и триграмм...
Построение словаря и корпуса...
Размер словаря: 4521
```

---

### `find_optimal_topics(start=4, limit=20, step=1) -> tuple`

Находит оптимальное количество тем по coherence.

**Возвращает:**
- Кортеж `(topic_range, model_list, coherence_values)`

**Что делает:**
1. Вычисляет coherence для диапазона тем через `compute_coherence_grid()`
2. Находит лучшую модель (с максимальным coherence)
3. Сохраняет информацию о лучшей модели
4. Выводит результаты

**Пример вывода:**
```
Поиск оптимального числа тем...
Обучение LDA: 4 тем
  coherence(c_v)=0.4523
...
Обучение LDA: 12 тем
  coherence(c_v)=0.5634  ← МАКСИМУМ

ЛУЧШАЯ МОДЕЛЬ
- Количество тем: 12
- Coherence (c_v): 0.5634
```

---

### `plot_coherence(save_path=None)`

Строит график coherence от количества тем.

**Параметры:**
- `save_path` (str, optional): Путь для сохранения графика

**Что делает:**
- Строит график зависимости coherence от числа тем
- Сохраняет в файл (если указан `save_path`) или показывает на экране

---

### `print_topics(num_words=10)`

Выводит ключевые слова по темам.

**Параметры:**
- `num_words` (int): Количество слов на тему (по умолчанию 10)

**Пример вывода:**
```
Ключевые слова по темам:
Тема 0: 0.023*"алгоритм" + 0.018*"данные" + 0.015*"метод" + ...
Тема 1: 0.031*"модель" + 0.020*"сеть" + 0.018*"нейрон" + ...
...
```

---

### `plot_wordclouds(save_path=None)`

Строит облака слов для каждой темы.

**Параметры:**
- `save_path` (str, optional): Путь для сохранения графика

**Что делает:**
- Для каждой темы строит облако слов (30 самых значимых слов)
- Располагает графики в сетке 2 колонки
- Сохраняет в файл или показывает на экране

---

### `get_model_info() -> dict`

Возвращает информацию о модели.

**Возвращает:**
- Словарь с полями: `best_num_topics`, `best_coherence`, `vocabulary_size`, `num_documents`

---

### Функция `run_lda_modeling(df, tokens_column='lda_tokens', start_topics=4, limit_topics=20, conn=None, output_dir=None)`

Упрощенная функция для запуска LDA моделирования с сохранением в базу данных.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `tokens_column` (str): Название колонки с токенами
- `start_topics` (int): Начальное количество тем
- `limit_topics` (int): Максимальное количество тем
- `conn`: Соединение с базой данных SQLite (опционально)
- `output_dir` (str, optional): Директория для сохранения графиков

**Возвращает:**
- Обученная модель `LDATopicModeler`

**Что делает:**
1. Инициализирует `LDATopicModeler`
2. Строит словарь и корпус
3. Находит оптимальное количество тем
4. Выводит темы
5. Строит графики (coherence и облака слов)
6. Сохраняет графики (если указан `output_dir`)
7. Сохраняет результаты в таблицу `lda_results` (если передан `conn`)

**Пример использования:**
```python
from backend.stages.lda import run_lda_modeling
from backend.database import get_connection

conn = get_connection("project_data/smarttag_vak.db")
modeler = run_lda_modeling(
    df,
    tokens_column='lda_tokens',
    start_topics=4,
    limit_topics=20,
    conn=conn,
    output_dir="project_data/lda_results"
)

print(modeler.get_model_info())
```

---

## Структура выходных данных

### База данных SQLite

Результаты LDA сохраняются в таблицу **`lda_results`** базы данных `project_data/smarttag_vak.db`:

**Структура таблицы `lda_results`:**
```sql
CREATE TABLE lda_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    lda_tokens TEXT,              -- Токены для LDA
    lda_topic_keywords TEXT,      -- Ключевые слова темы
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
)
```

### Графики

```
project_data/
└── lda_results/
    ├── lda_coherence.png        # График coherence
    └── lda_wordclouds.png      # Облака слов по темам
```

---

## Запуск

Блок можно запустить через пайплайн:

```python
from backend.pipeline import run_pipeline

# LDA выполняется автоматически на этапе 4
run_pipeline(db_path="project_data/smarttag_vak.db")
```

Или импортировать в другой код:

```python
from backend.stages.lda import run_lda_modeling
from backend.database import get_connection, get_all_articles

conn = get_connection("project_data/smarttag_vak.db")
df = get_all_articles(conn)

modeler = run_lda_modeling(
    df,
    tokens_column='lda_tokens',
    start_topics=4,
    limit_topics=20,
    conn=conn,
    output_dir="project_data/lda_results"
)
```

---

## Зависимости

- `numpy`: Работа с массивами
- `pandas`: Работа с табличными данными
- `matplotlib`: Построение графиков
- `wordcloud`: Облака слов
- `gensim`: Тематическое моделирование (LDA)
- `ast`: Преобразование строк в списки

---

## Примечания

- Используется многопоточная версия LDA (`LdaMulticore`) с 4 воркерами
- Coherence вычисляется по метрике c_v (наиболее качественная)
- Автоматически отбрасываются документы с менее чем 20 токенами
- Графики сохраняются в формате PNG с разрешением 150 DPI
- Модель обучается с asymmetric alpha и auto eta для лучшего качества
