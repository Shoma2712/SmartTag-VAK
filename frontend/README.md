# Модуль `frontend` — Tkinter UI

## Описание

Локальное окно «Подбор специальности ВАК»: четыре поля ввода с теми же весами, что в бэкенде, кнопка классификации, таблица top-3, опциональная таблица всех пяти специальностей с `semantic_similarity`, блок подсказок (неоднозначность, расхождение TF-IDF и embed).

Зависимости: `tkinter`, `pandas`, `backend.vak_classifiers.ui_classifier`.

## Константы

- `MIN_TEXT_LEN = 30` — минимальная суммарная длина всех полей перед запуском.

---

## Класс `VakClassifierApp(tk.Tk)`

### `__init__()`

Создаёт окно 960×820, инициализирует UI, запускает фоновую предзагрузку модели эмбеддингов (`after(200, _preload_model)`).

**Состояние:**

- `_classifier` — ленивый `VakUiClassifier`;
- `_busy` — блокировка повторного нажатия;
- `_last_result` — последний `VakUiResult` для режима «Подробнее».

---

### `_build_ui() -> None`

**Что делает:** размещает подсказку, форму из четырёх `ScrolledText`, кнопки «Подобрать», «Вставить», «Очистить», чекбокс «Подробнее», статус, фрейм флагов, `Treeview` для top-3 и детальной таблицы.

**Выход:** только изменение виджетов.

---

### `_add_field(parent, label, height, first=False) -> ScrolledText`

**Параметры:** родительский фрейм, подпись, высота в строках.

**Возвращает:** виджет текста с привязкой буфера обмена и фокуса.

---

### `_collect_article_fields() -> Dict[str, str]`

**Возвращает:** `title`, `annotation`, `keywords`, `main_text` из полей формы.

---

### `_on_classify() -> None`

**Вход:** текст из полей (проверка длины и наличия title/annotation).

**Что делает:** в отдельном потоке вызывает `VakUiClassifier.classify_row(pd.Series(fields))`, по завершении — `_show_result` в главном потоке Tk.

**Выход:** обновление таблиц и подсказок; при ошибке — `messagebox`.

---

### `_show_result(result: VakUiResult) -> None`

Заполняет top-3 (`match_score` как проценты 0–100%), текст флагов (`ambiguous`, `discrepancy`, `message`), обновляет детальную таблицу если включена.

---

### `_refresh_details() -> None`

Показывает/скрывает фрейм «Подробнее»; выводит все 5 специальностей с `semantic_similarity` и рангами TF-IDF / embed.

---

### `_preload_model()` / `_load_classifier()` / `_get_classifier()`

Фоновая и синхронная загрузка `VakUiClassifier(VakUiConfig())` с блокировкой `threading.Lock`; статусная строка «Модель загружается…» / «Готово».

---

### `_on_clear()`, `_clear_results()`, `_set_busy(busy, message)`

Очистка полей и таблиц; отключение кнопки на время классификации.

---

### Буфер обмена

| Метод | Назначение |
|-------|------------|
| `_paste_into_focused()` | Вставка в активное поле |
| `_setup_text_clipboard(widget)` | Ctrl+V, контекстное меню |
| `_paste_clipboard`, `_copy`, `_cut`, `_select_all` | Статические обработчики |

---

## `main() -> None`

**Что делает:** `VakClassifierApp().mainloop()`; при отсутствии Tkinter — сообщение в stderr и `sys.exit(1)`.

**Запуск:** `python -m frontend.tkinter_app`.
