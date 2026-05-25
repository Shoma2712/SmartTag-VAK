# Модуль `vak_classifiers`

## Описание

Два варианта классификации по пяти специальностям ВАК журнала:

1. **`ui_classifier.py`** — основной для UI и CLI: ранжирование top-3 по TF-IDF (полные описания ВАК), отображаемый балл min–max, эмбеддинги только для «Подробнее» и флагов.

---

## `ui_classifier.py` — продакшен для интерфейса

### Типы и конфигурация

#### `VakUiConfig` (dataclass)

| Поле | По умолчанию | Смысл |
|------|----------------|-------|
| `embed_model_name` | `mlsa-iai-msu-lab/sci-rus-tiny` | Модель sentence-transformers |
| `main_text_max_chars` | 3000 | Сколько символов `main_text` добавить к TF-IDF |
| `field_weights` | title×3, annotation×2, keywords×2 | Веса полей |
| `score_mode` | `"minmax"` | `"minmax"` — UI-проценты; `"cosine"` — сырой TF-IDF cosine |
| `ambiguous_gap` / `ambiguous_gap_minmax` | 0.02 / 0.08 | Порог «близких» 1–2 места |
| `top_k` | 3 | Размер top-N |

#### `SpecialtyScore`, `UiFlags`, `VakUiResult`

Структуры ответа: код ВАК, название, `match_score`, `semantic_similarity`, ранги, флаги `ambiguous` / `discrepancy`, `message`. Методы `to_dict()`, `to_json()`.

---

### `vak_code(full_name: str) -> str`

Извлекает код вида `2.3.1` из полного ключа специальности.

---

### `build_tfidf_input(row, config) -> str`

**Вход:** словарь или `Series` с полями статьи, `VakUiConfig`.

**Выход:** строка = взвешенные meta + начало `main_text` (до `main_text_max_chars`).

**Назначение:** корпус для сравнения с полными описаниями ВАК через TF-IDF.

---

### `build_semantic_input(row, config) -> str`

**Вход / выход:** как выше, но **без** `main_text`.

**Назначение:** текст для эмбеддингов (полный текст ухудшает разделение классов).

---

### Класс `VakUiClassifier`

#### `__init__(config=None)`

Загружает конфиг, готовит короткие и полные тексты ВАК; модель эмбеддингов подгружается лениво при первом вызове.

#### `classify_row(row, corpus_tfidf_texts=None) -> VakUiResult`

**Вход:** поля статьи; опционально общий корпус TF-IDF для пакетного режима.

**Выход:** `VakUiResult` с `top3`, `all_specialties`, `flags`.

**Логика:** TF-IDF cosine по 5 ВАК → min–max (или cosine) для отображения; параллельно semantic cosine по коротким описаниям ВАК; флаги неоднозначности и расхождения top-1.

#### `classify_text(text_tfidf, text_semantic=None) -> VakUiResult`

Классификация одной произвольной строки (для простых сценариев).

#### `classify_dataframe(df) -> pd.DataFrame`

**Вход:** DataFrame со статьями.

**Выход:** копия с колонками `vak_ui_top1_code`, `vak_ui_top1_match_score`, `vak_ui_ambiguous`, `vak_ui_discrepancy`, `vak_ui_message`, `vak_ui_json`.

---

### `classify_for_ui(df, main_text_max_chars=3000) -> pd.DataFrame`

Обёртка для ноутбука: создаёт `VakUiClassifier` и вызывает `classify_dataframe`.

---

