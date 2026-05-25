# -*- coding: utf-8 -*-
"""
Локальный интерфейс подбора специальности ВАК (Tkinter).

Запуск из корня проекта:
    .venv\\Scripts\\python.exe vak_tkinter_app.py

Ранжирование: TF-IDF (полные описания ВАК).
Дополнительно: эмбеддинги для режима «Подробнее» и флагов расхождения.
"""
from __future__ import annotations

import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional, Union

import pandas as pd

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

        self._build_ui()
        self.after(200, self._preload_model)

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

        self.show_details = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            btn_row,
            text="Подробнее (все 5 специальностей + semantic similarity)",
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
            self, text="Top-3 (основной результат - степень соответствия)"
        )
        top_frame.pack(fill=tk.BOTH, expand=True, **pad)

        cols = ("rank", "code", "title", "match", "hint")
        self.top_tree = ttk.Treeview(
            top_frame, columns=cols, show="headings", height=4
        )
        self.top_tree.heading("rank", text="№")
        self.top_tree.heading("code", text="Код ВАК")
        self.top_tree.heading("title", text="Специальность")
        self.top_tree.heading("match", text="Соответствие")
        self.top_tree.heading("hint", text="Примечание")
        self.top_tree.column("rank", width=40, anchor=tk.CENTER)
        self.top_tree.column("code", width=70, anchor=tk.CENTER)
        self.top_tree.column("title", width=420)
        self.top_tree.column("match", width=100, anchor=tk.CENTER)
        self.top_tree.column("hint", width=220)
        self.top_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.details_frame = ttk.LabelFrame(
            self,
            text="Подробнее: все специальности (semantic similarity — смысловая близость, для справки)",
        )
        self.details_frame.pack(fill=tk.BOTH, expand=True, **pad)

        dcols = ("code", "title", "match", "semantic", "rank_t", "rank_s")
        self.details_tree = ttk.Treeview(
            self.details_frame, columns=dcols, show="headings", height=6
        )
        for cid, title, w in [
            ("code", "Код", 70),
            ("title", "Специальность", 380),
            ("match", "Соответствие", 95),
            ("semantic", "Semantic sim.", 95),
            ("rank_t", "Ранг TF-IDF", 80),
            ("rank_s", "Ранг embed", 80),
        ]:
            self.details_tree.heading(cid, text=title)
            self.details_tree.column(cid, width=w, anchor=tk.CENTER if cid != "title" else tk.W)
        self.details_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.details_frame.pack_forget()

        foot = ttk.Label(
            self,
            text=(
                "Соответствие — min–max по 5 специальностям ВАК (TF-IDF), не вероятность. "
                "Semantic similarity — сырой cosine эмбеддингов (режим «Подробнее»)."
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
                    self._classifier = VakUiClassifier(VakUiConfig())
            self.after(0, lambda: self.status_var.set("Готово. Введите текст и нажмите «Подобрать»."))
        except Exception as exc:
            self.after(
                0,
                lambda: self.status_var.set(f"Ошибка загрузки модели: {exc}"),
            )

    def _get_classifier(self) -> VakUiClassifier:
        with self._classifier_lock:
            if self._classifier is None:
                self._classifier = VakUiClassifier(VakUiConfig())
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

        flags = result.flags
        parts = []
        if flags.message:
            parts.append(flags.message)
        parts.append(
            f"Разрыв 1–2 места (TF-IDF): {flags.ambiguous_gap:.3f}"
            + (" - неоднозначно" if flags.ambiguous else "")
        )
        if flags.discrepancy:
            parts.append(
                f"Расхождение: TF-IDF → {flags.tfidf_top1_code}, "
                f"embed → {flags.semantic_top1_code}"
            )
        self.flag_label.config(text="\n".join(parts))

        for i, item in enumerate(result.top3, start=1):
            hint = ""
            if i == 1 and flags.ambiguous:
                hint = "близкие 2-3 места"
            if i == 1 and flags.discrepancy and flags.embed_alternative_code:
                hint = (hint + "; " if hint else "") + "см. подсказку выше"
            self.top_tree.insert(
                "",
                tk.END,
                values=(
                    i,
                    item.code,
                    item.title[:120],
                    f"{item.match_score * 100:.0f}%",
                    hint,
                ),
            )

        self._refresh_details()

    def _refresh_details(self) -> None:
        if self.show_details.get():
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
                    f"{item.match_score * 100:.1f}%",
                    f"{item.semantic_similarity:.3f}",
                    item.rank_tfidf,
                    item.rank_semantic,
                ),
            )


def main() -> None:
    try:
        app = VakClassifierApp()
        app.mainloop()
    except tk.TclError as exc:
        print("Tkinter недоступен:", exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
