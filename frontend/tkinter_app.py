# -*- coding: utf-8 -*-
"""
Локальный интерфейс подбора специальности ВАК (Tkinter).

Ранжирование: Sentence Embeddings (семантическая близость).
Дополнительно: TF-IDF для режима «Подробнее» и флагов расхождения.
"""
from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from backend.config import DEFAULT_DATABASE_PATH
from backend.database import (
    get_connection, 
    get_articles_with_vak, 
    get_articles_with_clustering,
    insert_saved_classification
)
from backend.vak_classifiers.ui_classifier import VakUiClassifier, VakUiConfig, VakUiResult

MIN_TEXT_LEN = 30
TextWidget = Union[scrolledtext.ScrolledText, tk.Entry, ttk.Entry]


class VakClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Подбор специальности ВАК")
        self.geometry("960x820")
        self.minsize(800, 700)

        self._classifier: Optional[VakUiClassifier] = None
        self._classifier_lock = threading.Lock()
        self._busy = False
        self._last_result: Optional[VakUiResult] = None
        self._focus_widget: Optional[TextWidget] = None
        self._input_widgets: List[TextWidget] = []
        
        # Database connection management
        self._db_conn = None
        self._db_path = DEFAULT_DATABASE_PATH
        self._init_database_connection()

        self._build_ui()
        self.after(200, self._preload_model)
        
        # Register cleanup handler
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _init_database_connection(self) -> None:
        """Initialize database connection and check if database exists."""
        try:
            if not Path(self._db_path).exists():
                # Database doesn't exist yet - this is okay for manual input mode
                # User can still manually enter article data for classification
                self._db_conn = None
                return
            
            # Connect to database
            self._db_conn = get_connection(str(self._db_path))
        except Exception as exc:
            # Log error but don't prevent app from starting
            # Manual input mode will still work
            print(f"Warning: Could not connect to database: {exc}", file=sys.stderr)
            self._db_conn = None
    
    def _on_closing(self) -> None:
        """Clean up resources and close the application."""
        # Close database connection if open
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception as exc:
                print(f"Error closing database connection: {exc}", file=sys.stderr)
        
        # Destroy the window
        self.destroy()
    
    def _load_articles_from_database(self) -> Optional[pd.DataFrame]:
        """
        Load articles with VAK classification results from database.
        
        Returns:
            DataFrame with articles and VAK results, or None if database not available
        """
        if self._db_conn is None:
            if not Path(self._db_path).exists():
                messagebox.showerror(
                    "База данных не найдена",
                    f"Файл базы данных не найден: {self._db_path}\n\n"
                    "Запустите pipeline для создания базы данных или используйте "
                    "ручной ввод для классификации статей."
                )
                return None
            
            # Try to reconnect
            try:
                self._db_conn = get_connection(str(self._db_path))
            except Exception as exc:
                messagebox.showerror(
                    "Ошибка подключения к БД",
                    f"Не удалось подключиться к базе данных:\n{exc}"
                )
                return None
        
        try:
            # Load articles with VAK classification results
            df = get_articles_with_vak(self._db_conn)
            return df
        except Exception as exc:
            messagebox.showerror(
                "Ошибка загрузки данных",
                f"Не удалось загрузить данные из базы:\n{exc}"
            )
            return None
    
    def _load_articles_with_clustering_from_database(self) -> Optional[pd.DataFrame]:
        """
        Load articles with clustering results from database.
        
        Returns:
            DataFrame with articles and clustering results, or None if database not available
        """
        if self._db_conn is None:
            if not Path(self._db_path).exists():
                messagebox.showerror(
                    "База данных не найдена",
                    f"Файл базы данных не найден: {self._db_path}\n\n"
                    "Запустите pipeline для создания базы данных."
                )
                return None
            
            # Try to reconnect
            try:
                self._db_conn = get_connection(str(self._db_path))
            except Exception as exc:
                messagebox.showerror(
                    "Ошибка подключения к БД",
                    f"Не удалось подключиться к базе данных:\n{exc}"
                )
                return None
        
        try:
            # Load articles with clustering results
            df = get_articles_with_clustering(self._db_conn)
            return df
        except Exception as exc:
            messagebox.showerror(
                "Ошибка загрузки данных",
                f"Не удалось загрузить данные из базы:\n{exc}"
            )
            return None

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        header = ttk.Label(
            self,
            text=(
                "Введите имеющиеся данные. "
                "Вставка: Shift+Ins в активном поле или кнопка «Вставить»."
            ),
            font=("Segoe UI", 9),
            wraplength=900,
        )
        header.pack(anchor="w", **pad)

        form = ttk.LabelFrame(self, text="Данные статьи")
        form.pack(fill=tk.BOTH, expand=True, **pad)

        self.field_title = self._add_field(form, "Название", height=2, first=True)
        self.field_annotation = self._add_field(
            form, "Аннотация", height=5
        )
        self.field_keywords = self._add_field(
            form, "Ключевые слова", height=2
        )
        self.field_main = self._add_field(
            form,
            "Фрагмент текста статьи",
            height=8,
        )

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, **pad)

        self.btn_run = ttk.Button(
            btn_row, text="Подобрать специальность", command=self._on_classify
        )
        self.btn_run.pack(side=tk.LEFT)

        self.btn_paste = ttk.Button(
            btn_row, text="Вставить", command=self._paste_into_focused
        )
        self.btn_paste.pack(side=tk.LEFT, padx=(8, 0))

        self.btn_clear = ttk.Button(btn_row, text="Очистить", command=self._on_clear)
        self.btn_clear.pack(side=tk.LEFT, padx=(8, 0))
        
        self.btn_save = ttk.Button(
            btn_row, 
            text="💾 Сохранить результаты", 
            command=self._on_save_results,
            state=tk.DISABLED
        )
        self.btn_save.pack(side=tk.LEFT, padx=(8, 0))

        self.show_details = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            btn_row,
            text="Подробнее (все 5 специальностей + TF-IDF)",
            variable=self.show_details,
            command=self._refresh_details,
        ).pack(side=tk.LEFT, padx=(16, 0))

        self.status_var = tk.StringVar(value="Модель загружается при старте…")
        ttk.Label(self, textvariable=self.status_var, foreground="#444").pack(
            anchor="w", **pad
        )

        self.flag_frame = ttk.LabelFrame(self, text="Подсказки")
        self.flag_frame.pack(fill=tk.X, **pad)
        self.flag_label = ttk.Label(
            self.flag_frame,
            text="-",
            wraplength=860,
            justify=tk.LEFT,
        )
        self.flag_label.pack(anchor="w", padx=8, pady=8)

        top_frame = ttk.LabelFrame(
            self, text="Top-3 (основной результат - семантическое сходство)"
        )
        top_frame.pack(fill=tk.BOTH, expand=True, **pad)

        cols = ("rank", "code", "title", "match", "hint")
        self.top_tree = ttk.Treeview(
            top_frame, columns=cols, show="headings", height=4
        )
        self.top_tree.heading("rank", text="№")
        self.top_tree.heading("code", text="Код ВАК")
        self.top_tree.heading("title", text="Специальность")
        self.top_tree.heading("match", text="Семант. сходство")
        self.top_tree.heading("hint", text="Примечание")
        self.top_tree.column("rank", width=40, anchor=tk.CENTER)
        self.top_tree.column("code", width=70, anchor=tk.CENTER)
        self.top_tree.column("title", width=420)
        self.top_tree.column("match", width=100, anchor=tk.CENTER)
        self.top_tree.column("hint", width=220)
        self.top_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.details_frame = ttk.LabelFrame(
            self,
            text="Подробнее: все специальности (TF-IDF — статистическое сходство, для справки)",
        )
        self.details_frame.pack(fill=tk.BOTH, expand=True, **pad)

        dcols = ("code", "title", "match", "tfidf", "rank_s", "rank_t")
        self.details_tree = ttk.Treeview(
            self.details_frame, columns=dcols, show="headings", height=6
        )
        for cid, title, w in [
            ("code", "Код", 70),
            ("title", "Специальность", 380),
            ("match", "Семант. сходство", 110),
            ("tfidf", "TF-IDF sim.", 95),
            ("rank_s", "Ранг Embedding", 90),
            ("rank_t", "Ранг TF-IDF", 90),
        ]:
            self.details_tree.heading(cid, text=title)
            self.details_tree.column(cid, width=w, anchor=tk.CENTER if cid != "title" else tk.W)
        self.details_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.details_frame.pack_forget()

        foot = ttk.Label(
            self,
            text=(
                "Семантическое сходство — сырой cosine similarity в процентаъ. "
                "TF-IDF similarity — статистическое сходство терминов (режим «Подробнее»)."
            ),
            font=("Segoe UI", 8),
            foreground="#666",
            wraplength=880,
        )
        foot.pack(anchor="w", **pad)

    def _add_field(
        self,
        parent: ttk.LabelFrame,
        label: str,
        height: int,
        first: bool = False,
    ) -> scrolledtext.ScrolledText:
        row_pad = {"padx": 8, "pady": (8 if first else 4, 0)}
        ttk.Label(parent, text=label, font=("Segoe UI", 9)).pack(anchor="w", **row_pad)
        widget = scrolledtext.ScrolledText(
            parent, height=height, wrap=tk.WORD, font=("Segoe UI", 10)
        )
        widget.pack(fill=tk.BOTH if height >= 5 else tk.X, expand=height >= 5, padx=8, pady=(0, 4))
        self._setup_text_clipboard(widget)
        self._input_widgets.append(widget)
        return widget

    def _paste_into_focused(self) -> None:
        target = self._focus_widget or self.field_title
        self._paste_clipboard(target)

    def _register_focus(self, widget: TextWidget) -> None:
        widget.bind("<FocusIn>", lambda e: setattr(self, "_focus_widget", widget))

    def _setup_text_clipboard(self, widget: scrolledtext.ScrolledText) -> None:
        self._register_focus(widget)
        """Shift+Insert и контекстное меню для вставки из буфера."""
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Вставить", command=lambda: self._paste_clipboard(widget))
        menu.add_command(label="Вырезать", command=lambda: self._cut(widget))
        menu.add_command(label="Копировать", command=lambda: self._copy(widget))
        menu.add_separator()
        menu.add_command(label="Выделить всё", command=lambda: self._select_all(widget))

        def show_menu(event: tk.Event) -> str:
            menu.tk_popup(event.x_root, event.y_root)
            return "break"

        widget.bind("<Button-3>", show_menu)

        for seq in ("<Control-v>", "<Control-V>", "<Shift-Insert>"):
            widget.bind(
                seq,
                lambda e, w=widget: self._paste_clipboard(w) or "break",
            )

        for seq, handler in (
            ("<Control-c>", self._copy),
            ("<Control-C>", self._copy),
            ("<Control-x>", self._cut),
            ("<Control-X>", self._cut),
            ("<Control-a>", self._select_all),
            ("<Control-A>", self._select_all),
        ):
            widget.bind(seq, lambda e, h=handler, w=widget: h(w))

    @staticmethod
    def _paste_clipboard(widget: Optional[TextWidget] = None) -> None:
        if widget is None:
            return
        if isinstance(widget, (tk.Entry, ttk.Entry)):
            try:
                text = widget.clipboard_get()
            except tk.TclError:
                return
            try:
                if widget.selection_present():
                    widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                pass
            widget.insert(tk.INSERT, text)
            return
        try:
            widget.event_generate("<<Paste>>")
            return
        except tk.TclError:
            pass
        try:
            text = widget.clipboard_get()
        except tk.TclError:
            return
        if widget.tag_ranges(tk.SEL):
            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
        widget.insert(tk.INSERT, text)

    @staticmethod
    def _copy(widget: scrolledtext.ScrolledText) -> str:
        try:
            widget.event_generate("<<Copy>>")
        except tk.TclError:
            pass
        return "break"

    @staticmethod
    def _cut(widget: scrolledtext.ScrolledText) -> str:
        try:
            widget.event_generate("<<Cut>>")
        except tk.TclError:
            pass
        return "break"

    @staticmethod
    def _select_all(widget: scrolledtext.ScrolledText) -> str:
        widget.tag_add(tk.SEL, "1.0", tk.END)
        widget.mark_set(tk.INSERT, tk.END)
        widget.see(tk.END)
        return "break"

    def _preload_model(self) -> None:
        thread = threading.Thread(target=self._load_classifier, daemon=True)
        thread.start()

    def _load_classifier(self) -> None:
        try:
            with self._classifier_lock:
                if self._classifier is None:
                    # Используем score_mode='cosine' для сырого cosine similarity embeddings
                    config = VakUiConfig(score_mode='cosine')
                    self._classifier = VakUiClassifier(config)
            self.after(0, lambda: self.status_var.set("Готово. Введите текст и нажмите «Подобрать»."))
        except Exception as exc:
            self.after(
                0,
                lambda: self.status_var.set(f"Ошибка загрузки модели: {exc}"),
            )

    def _get_classifier(self) -> VakUiClassifier:
        with self._classifier_lock:
            if self._classifier is None:
                # Используем score_mode='cosine' для сырого cosine similarity embeddings
                config = VakUiConfig(score_mode='cosine')
                self._classifier = VakUiClassifier(config)
            return self._classifier

    def _collect_article_fields(self) -> Dict[str, str]:
        return {
            "title": self.field_title.get("1.0", tk.END).strip(),
            "annotation": self.field_annotation.get("1.0", tk.END).strip(),
            "keywords": self.field_keywords.get("1.0", tk.END).strip(),
            "main_text": self.field_main.get("1.0", tk.END).strip(),
        }

    def _on_clear(self) -> None:
        for widget in self._input_widgets:
            widget.delete("1.0", tk.END)
        self._clear_results()
        self.btn_save.config(state=tk.DISABLED)

    def _clear_results(self) -> None:
        for tree in (self.top_tree, self.details_tree):
            for item in tree.get_children():
                tree.delete(item)
        self.flag_label.config(text="—")

    def _set_busy(self, busy: bool, message: str) -> None:
        self._busy = busy
        self.btn_run.config(state=tk.DISABLED if busy else tk.NORMAL)
        self.status_var.set(message)

    def _on_classify(self) -> None:
        if self._busy:
            return
        fields = self._collect_article_fields()
        total_len = sum(len(v) for v in fields.values())
        if total_len < MIN_TEXT_LEN:
            messagebox.showwarning(
                "Мало текста",
                f"Заполните поля суммарно хотя бы на {MIN_TEXT_LEN} символов "
                f"(сейчас {total_len}).",
            )
            return
        if not fields["title"] and not fields["annotation"]:
            messagebox.showwarning(
                "Нет основных полей",
                "Укажите хотя бы название или аннотацию.",
            )
            return

        self._set_busy(True, "Классификация… (первый раз может занять до минуты)")
        thread = threading.Thread(
            target=self._classify_worker, args=(fields,), daemon=True
        )
        thread.start()

    def _classify_worker(self, fields: Dict[str, str]) -> None:
        try:
            clf = self._get_classifier()
            row = pd.Series(fields)
            result = clf.classify_row(row)
            
            self.after(0, lambda: self._show_result(result))
        except Exception as exc:
            self.after(
                0,
                lambda: messagebox.showerror("Ошибка", str(exc)),
            )
        finally:
            self.after(0, lambda: self._set_busy(False, "Готово."))

    def _show_result(self, result: VakUiResult) -> None:
        self._last_result = result
        self._clear_results()
        
        # Включаем кнопку "Сохранить результаты"
        self.btn_save.config(state=tk.NORMAL)

        flags = result.flags
        parts = []
        if flags.message:
            parts.append(flags.message)
        parts.append(
            f"Разрыв 1–2 места (Embeddings): {flags.ambiguous_gap:.3f}"
            + (" - неоднозначно" if flags.ambiguous else "")
        )
        if flags.discrepancy:
            parts.append(
                f"Расхождение: Embeddings → {flags.semantic_top1_code}, "
                f"TF-IDF → {flags.tfidf_top1_code}"
            )
        self.flag_label.config(text="\n".join(parts))

        # Отладка: проверяем, что top3 не пустой
        print(f"DEBUG: result.top3 length = {len(result.top3)}")
        for i, item in enumerate(result.top3, start=1):
            print(f"DEBUG: Item {i}: code={item.code}, match_score={item.match_score}")
            hint = ""
            if i == 1 and flags.ambiguous:
                hint = "близкие 2-3 места"
            if i == 1 and flags.discrepancy and flags.tfidf_alternative_code:
                hint = (hint + "; " if hint else "") + "см. подсказку выше"
            self.top_tree.insert(
                "",
                tk.END,
                values=(
                    i,
                    item.code,
                    item.title[:120],
                    f"{item.match_score * 100:.1f}%",
                    hint,
                ),
            )

        self._refresh_details()

    def _refresh_details(self) -> None:
        show_details = self.show_details.get()
        
        if show_details:
            self.details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        else:
            self.details_frame.pack_forget()
            return

        for item in self.details_tree.get_children():
            self.details_tree.delete(item)

        if not self._last_result:
            return

        for item in self._last_result.all_specialties:
            self.details_tree.insert(
                "",
                tk.END,
                values=(
                    item.code,
                    item.title[:100],
                    f"{item.match_score * 100:.1f}%",  # Embeddings (сырой) - основной
                    f"{item.tfidf_similarity:.3f}",    # TF-IDF (сырой) - дополнительный
                    item.rank_semantic,
                    item.rank_tfidf,
                ),
            )
    
    def _on_save_results(self) -> None:
        """Сохранение результатов классификации в базу данных."""
        if not self._last_result:
            messagebox.showwarning(
                "Нет результатов",
                "Сначала выполните классификацию, затем сохраните результаты."
            )
            return
        
        if self._db_conn is None:
            messagebox.showerror(
                "База данных недоступна",
                "Не удалось подключиться к базе данных.\n"
                "Результаты не могут быть сохранены."
            )
            return
        
        try:
            # Собираем данные из полей ввода
            fields = self._collect_article_fields()
            
            # Получаем результаты классификации
            result = self._last_result
            
            # Находим топ-1 по Embedding (основной метод)
            top1_embed = result.top3[0] if result.top3 else None
            
            # Находим топ-1 по TF-IDF (дополнительный метод)
            top1_tfidf = None
            for spec in result.all_specialties:
                if spec.rank_tfidf == 1:
                    top1_tfidf = spec
                    break
            
            # С новой логикой:
            # - match_score содержит сырой cosine similarity для Embeddings (основной)
            # - tfidf_similarity содержит сырой cosine similarity для TF-IDF (дополнительный)
            
            embed_score = top1_embed.match_score if top1_embed else None  # Embeddings (основной)
            tfidf_score = top1_tfidf.tfidf_similarity if top1_tfidf else None  # TF-IDF (дополнительный)
            
            # Формируем JSON с топ-3
            import json
            top3_data = [
                {
                    'rank': i + 1,
                    'code': item.code,
                    'title': item.title,
                    'match_score': item.match_score,  # Embeddings (основной)
                    'tfidf_similarity': item.tfidf_similarity  # TF-IDF (дополнительный)
                }
                for i, item in enumerate(result.top3)
            ]
            
            # Подготовка данных для сохранения
            classification_data = {
                'title': fields['title'] or '(без названия)',
                'annotation': fields['annotation'],
                'keywords': fields['keywords'],
                'main_text': fields['main_text'],
                'vak_tfidf_code': top1_tfidf.code if top1_tfidf else None,
                'vak_tfidf_title': top1_tfidf.title if top1_tfidf else None,
                'vak_tfidf_score': tfidf_score,  # Сырой cosine similarity (дополнительный)
                'vak_embed_code': top1_embed.code if top1_embed else None,
                'vak_embed_title': top1_embed.title if top1_embed else None,
                'vak_embed_score': embed_score,  # Сырой cosine similarity (основной)
                'top3_json': json.dumps(top3_data, ensure_ascii=False),
                'is_ambiguous': 1 if result.flags.ambiguous else 0,
                'has_discrepancy': 1 if result.flags.discrepancy else 0,
                'notes': None
            }
            
            # Сохранение в базу данных
            saved_id = insert_saved_classification(self._db_conn, classification_data)
            
            # Уведомление пользователя
            messagebox.showinfo(
                "Результаты сохранены",
                f"Результаты классификации успешно сохранены!\n\n"
                f"ID записи: {saved_id}\n"
                f"Embeddings (основной): {classification_data['vak_embed_code']}\n"
                f"TF-IDF (дополнительный): {classification_data['vak_tfidf_code']}\n\n"
                f"Вы можете просмотреть сохранённые результаты в базе данных."
            )
            
        except Exception as exc:
            messagebox.showerror(
                "Ошибка сохранения",
                f"Не удалось сохранить результаты:\n{exc}"
            )
            print(f"Error saving classification: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()


def main() -> None:
    try:
        app = VakClassifierApp()
        app.mainloop()
    except tk.TclError as exc:
        print("Tkinter недоступен:", exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
