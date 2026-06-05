# Блок 5: Кластеризация на эмбеддингах

## Описание

Этот блок отвечает за кластеризацию научных статей на основе их семантических эмбеддингов. Блок автоматически подбирает лучшую модель эмбеддингов, оптимальное количество кластеров и визуализирует результаты.

## Основные функции

### `normalize_text(s: str) -> str`

Нормализация текста.

**Параметры:**
- `s` (str): Исходный текст

**Возвращает:**
- Нормализованный текст (схлопывание пробелов, замена `\xa0` на пробел)

---

### `build_text_views(row: pd.Series) -> dict`

Строит различные представления текста для эмбеддингов.

**Параметры:**
- `row` (pd.Series): Строка DataFrame

**Возвращает:**
- Словарь с представлениями:
  - `title_ann_kw`: заголовок + аннотация + ключевые слова (ограничение 9000 символов)
  - `full_text`: полный текст (ограничение 9000 символов)

**Что делает:**
- Объединяет заголовок, аннотацию и ключевые слова
- Обрезает полный текст до 9000 символов
- Нормализует все представления

---

### `load_embedding_model(model_name: str, allow_download: bool = True)`

Загружает модель для создания эмбеддингов.

**Параметры:**
- `model_name` (str): Название модели (например, "cointegrated/rubert-tiny2")
- `allow_download` (bool): Разрешить загрузку модели

**Возвращает:**
- Кортеж `(модель, описание_источника)`

**Что делает:**
- Пытается загрузить модель через `SentenceTransformer`
- Поддерживает работу в офлайн-режиме (`local_files_only`)
- Возвращает источник модели (локальная или скачанная)

---

### `encode_texts(model, texts: list, batch_size: int = 32) -> np.ndarray`

Создает эмбеддинги для текстов.

**Параметры:**
- `model`: Модель sentence-transformers
- `texts` (list): Список текстов
- `batch_size` (int): Размер батча (по умолчанию 32)

**Возвращает:**
- Матрица эмбеддингов (numpy array, float32)

**Что делает:**
- Кодирует тексты с нормализацией эмбеддингов
- Показывает прогресс-бар
- Нормализует результат через `sklearn.preprocessing.normalize()`

---

### `evaluate_k_range(x: np.ndarray, k_values: list, sample_size: int = 500, random_state: int = 42) -> pd.DataFrame`

Оценивает качество кластеризации для разных значений k.

**Параметры:**
- `x` (np.ndarray): Матрица эмбеддингов
- `k_values` (list): Список значений k для проверки
- `sample_size` (int): Размер выборки для silhouette (по умолчанию 500)
- `random_state` (int): Random state (по умолчанию 42)

**Возвращает:**
- DataFrame со столбцами `k` и `silhouette`

**Что делает:**
- Для каждого k выполняет KMeans (n_init=20)
- Вычисляет Silhouette Score (метрика cosine)
- Использует подвыборку для ускорения (sample_size)

---

## Класс `EmbeddingClusterer`

Класс для кластеризации текстов на основе эмбеддингов.

### `EmbeddingClusterer(df: pd.DataFrame, random_state: int = 42)`

Инициализация кластеризатора.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `random_state` (int): Random state для воспроизводимости

**Что делает при инициализации:**
- Копирует DataFrame
- Сохраняет random_state
- Инициализирует кэш эмбеддингов

---

### `prepare_data(text_views: dict = None)`

Подготовка данных для кластеризации.

**Параметры:**
- `text_views` (dict): Словарь с описаниями представлений текста

**Что делает:**
1. Строит представления текста через `build_text_views()`
2. Добавляет колонки с представлениями в DataFrame
3. Удаляет статьи с неопределенной темой («Тема не указана в содержании», «Тема не определена», пустые)
4. Фильтрует пустые тексты
5. Выводит размер датасета и количество уникальных тем

**Пример вывода:**
```
Построение представлений текста...
Удалено статей с неопределённой темой: 12
Удалённые темы: {'Тема не определена': 12}
Размер датасета: 300 | уникальные темы: 15
```

---

### `encode_with_model(model_name: str, text_view: str, batch_size: int = 32) -> np.ndarray`

Создает эмбеддинги для текстов с помощью модели.

**Параметры:**
- `model_name` (str): Название модели
- `text_view` (str): Представление текста
- `batch_size` (int): Размер батча

**Возвращает:**
- Матрица эмбеддингов

**Что делает:**
- Проверяет кэш (если эмбеддинги уже посчитаны — возвращает из кэша)
- Загружает модель через `load_embedding_model()`
- Кодирует тексты через `encode_texts()`
- Сохраняет в кэш

---

### `find_optimal_k(x: np.ndarray, min_k: int = 6, max_k: int = 35, sample_size: int = 500) -> pd.DataFrame`

Находит оптимальное количество кластеров.

**Параметры:**
- `x` (np.ndarray): Матрица эмбеддингов
- `min_k` (int): Минимальное количество кластеров
- `max_k` (int): Максимальное количество кластеров
- `sample_size` (int): Размер выборки для silhouette

**Возвращает:**
- DataFrame с метриками для каждого k

**Что делает:**
- Определяет диапазон k (с учетом размера данных)
- Вычисляет Silhouette Score для каждого k через `evaluate_k_range()`
- Выводит прогресс

---

### `cluster(x: np.ndarray, k: int) -> np.ndarray`

Выполняет кластеризацию.

**Параметры:**
- `x` (np.ndarray): Матрица эмбеддингов
- `k` (int): Количество кластеров

**Возвращает:**
- Метки кластеров

**Что делает:**
- Выполняет KMeans (n_init=20, фиксированный random_state)
- Выводит информацию о процессе

---

### `compute_tsne(x: np.ndarray, pca_dim: int = 50) -> np.ndarray`

Вычисляет t-SNE для визуализации.

**Параметры:**
- `x` (np.ndarray): Матрица эмбеддингов
- `pca_dim` (int): Размерность PCA перед t-SNE (по умолчанию 50)

**Возвращает:**
- 2D координаты t-SNE

**Что делает:**
1. Снижает размерность через PCA (до pca_dim)
2. Применяет t-SNE (perplexity адаптивный, max_iter=1500)
3. Использует метрику cosine и инициализацию PCA

---

### `save_results(output_dir: Path, labels: np.ndarray, z_tsne: np.ndarray, best_k: int, model_name: str, text_view: str, conn=None)`

Сохраняет результаты кластеризации в базу данных и файлы.

**Параметры:**
- `output_dir` (Path): Директория для сохранения файлов
- `labels` (np.ndarray): Метки кластеров
- `z_tsne` (np.ndarray): 2D координаты t-SNE
- `best_k` (int): Количество кластеров
- `model_name` (str): Название модели эмбеддингов
- `text_view` (str): Представление текста
- `conn`: Соединение с базой данных SQLite (опционально)

**Что делает:**
1. Добавляет метки кластеров и параметры в DataFrame
2. Сохраняет результаты в таблицу `clustering_results` (если передан `conn`)
3. Сохраняет CSV с результатами (`articles_with_clusters.csv`)
4. Строит матрицу «кластер-тема» (`cluster_topic_matrix.csv`)

---

### Функция `run_clustering(df: pd.DataFrame, min_k: int = 6, max_k: int = 35, output_dir: str = None, conn=None, random_state: int = 42) -> tuple`

Автоматический подбор лучшей модели и представления текста для кластеризации с сохранением в базу данных.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `min_k` (int): Минимальное количество кластеров
- `max_k` (int): Максимальное количество кластеров
- `output_dir` (str): Директория для сохранения результатов
- `conn`: Соединение с базой данных SQLite (опционально)
- `random_state` (int): Random state

**Возвращает:**
- Кортеж `(clusterer, df_with_clusters, best_k, best_model, best_text_view)`

**Что делает:**
1. Перебирает модели:
   - `cointegrated/rubert-tiny2`
   - `mlsa-iai-msu-lab/sci-rus-tiny`
2. Перебирает представления текста:
   - `title_ann_kw`
   - `full_text`
3. Для каждой комбинации:
   - Создает эмбеддинги
   - Находит оптимальное k (по Silhouette Score)
4. Выбирает лучшую комбинацию
5. Выполняет кластеризацию с лучшими параметрами
6. Вычисляет t-SNE
7. Сохраняет результаты в базу данных (если передан `conn`)
8. Сохраняет графики

**Пример использования:**
```python
from backend.stages.clustering import run_clustering
from backend.database import get_connection, get_all_articles

conn = get_connection("project_data/smarttag_vak.db")
df = get_all_articles(conn)

clusterer, df_clustered, best_k, best_model, best_text_view = run_clustering(
    df,
    min_k=6,
    max_k=35,
    output_dir="project_data/clustering_results",
    conn=conn
)

print(f"Кластеризация завершена:")
print(f"  Модель: {best_model}")
print(f"  Представление: {best_text_view}")
print(f"  Количество кластеров: {best_k}")
```

---

## Структура выходных данных

### База данных SQLite

Результаты кластеризации сохраняются в таблицу **`clustering_results`** базы данных `project_data/smarttag_vak.db`:

**Структура таблицы `clustering_results`:**
```sql
CREATE TABLE clustering_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    cluster_name TEXT,            -- Название кластера
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
)
```

### Файлы

```
project_data/
└── clustering_results/
    ├── articles_with_clusters.csv    # Статьи с кластерами (резервная копия)
    ├── cluster_topic_matrix.csv      # Матрица кластер-тема
    ├── tsne_clusters.png             # t-SNE визуализация
    └── silhouette_curve.png         # График подбора k
```

---

## Запуск

Блок можно запустить через пайплайн:

```python
from backend.pipeline import run_pipeline

# Кластеризация выполняется автоматически на этапе 5
run_pipeline(db_path="project_data/smarttag_vak.db")
```

Или импортировать в другой код:

```python
from backend.stages.clustering import run_clustering
from backend.database import get_connection, get_all_articles

conn = get_connection("project_data/smarttag_vak.db")
df = get_all_articles(conn)

clusterer, df_clustered, best_k, best_model, best_text_view = run_clustering(
    df,
    min_k=6,
    max_k=35,
    output_dir="project_data/clustering_results",
    conn=conn
)

print(f"Кластеризация завершена:")
print(f"  Модель: {best_model}")
print(f"  Представление: {best_text_view}")
print(f"  Количество кластеров: {best_k}")
```

---

## Зависимости

- `time`: Замер времени выполнения
- `pathlib`: Работа с путями
- `matplotlib`: Построение графиков
- `numpy`: Работа с массивами
- `pandas`: Работа с табличными данными
- `sklearn`: Кластеризация (KMeans), метрики (silhouette), декомпозиция (PCA)
- `sklearn.manifold`: t-SNE для визуализации
- `sentence_transformers`: Создание эмбеддингов

---

## Примечания

- Блок автоматически подбирает лучшую модель по Silhouette Score
- **Типичные значения Silhouette Score:** 0.15-0.45 (зависит от данных и количества кластеров)
- KMeans обучается с `n_init=20` для стабильных результатов
- t-SNE использует предварительную PCA для ускорения
- Эмбеддинги кэшируются: повторный запуск с той же моделью не пересчитывает их
- Silhouette Score вычисляется на подвыборке (500 объектов) для ускорения
- perplexity для t-SNE адаптивный (от 10 до 30)
- Результаты сохраняются в базу данных SQLite для дальнейшего анализа
