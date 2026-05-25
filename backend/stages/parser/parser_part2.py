# БЛОК 3 - ЧАСТЬ 2: Извлечение тематики из содержания
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
import pdfplumber


# Константы
TOPIC_NO_CONTENTS = "Тема не указана в содержании"
TOPIC_UNDEFINED = "Тема не определена"

RUS_CONTENTS = "Содержание"
RUS_UDK = "УДК"
RUS_JOURNAL_FULL = "информационные и математические технологии в науке и управлении"
ENG_JOURNAL_FULL = "information and mathematical technologies in science and management"

RU_UP = "А-ЯЁ"
RU_LOW = "а-яё"

PURE_PAGE_RE = re.compile(r"^\d{1,3}$")
TAIL_PAGE_RE = re.compile(r"(?:\s|\.)(\d{1,3})\s*$")
UDK_PAGE_RE = re.compile(rf"(?im)^(?:{re.escape(RUS_UDK)}|UDC)\s*[\d\.\+\:\-]")

AUTHOR_START_RE = re.compile(
    rf"^(?:[{RU_UP}][{RU_LOW}\-]+(?:\s+[{RU_UP}]\.\s?[{RU_UP}]\.?)+|"
    rf"[{RU_UP}][{RU_LOW}\-]+\s+[{RU_UP}][{RU_LOW}\-]+\s+[{RU_UP}][{RU_LOW}\-]+|"
    rf"[A-Z][a-z\-]+(?:\s+[A-Z]\.\s?[A-Z]\.?)+)"
)

RUS_PREP_END_RE = re.compile(
    r"(?:^|\s)(?:"
    r"в|на|и|или|а|но|для|по|об|при|с|к|о|из|за|у|до|со|под|над|от|без|через|про|"
    r"между|не|что|как|это|тоже|их|его|её"
    r")\s*$",
    re.IGNORECASE,
)

TOPIC_KEYWORDS = [
    "модел", "технолог", "метод", "систем", "интеллект", "энергет", "эконом",
    "программ", "информац", "кибер", "вычисл", "управл", "анализ", "аналит",
    "цифров", "визуал", "поддерж", "онтолог", "безопас", "эколог",
]

EDITORIAL_RE = re.compile(r"(?i)(предисловие|editor's\s+preface|редакц|содержание|content|памяти|in memory)")
AUTHOR_IN_TOPIC_RE = re.compile(
    r"(?:[А-ЯЁ][а-яё\-]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.?)|"
    r"(?:[A-Z][a-z\-]+\s+[A-Z]\.\s?[A-Z]\.?)"
)
START_INITIALS_RE = re.compile(r"^(?:[А-ЯЁA-Z]\.){1,2}")
LOWER_START_RE = re.compile(r"^[а-яёa-z]")


class TOCExtractor:
    """
    Класс для извлечения тематических разделов из содержания PDF.
    """

    @staticmethod
    def normalize_line(line: str) -> str:
        """
        Нормализация строки из содержания.

        Args:
            line (str): Исходная строка

        Returns:
            str: Нормализованная строка
        """
        return re.sub(r"\s+", " ", str(line or "").replace(" ", " ")).strip().strip("-—.;:")

    @staticmethod
    def tail_page(line: str):
        """
        Извлекает номер страницы из конца строки.

        Args:
            line (str): Строка с номером страницы

        Returns:
            int: Номер страницы или None
        """
        m = TAIL_PAGE_RE.search(line)
        return int(m.group(1)) if m else None

    @staticmethod
    def strip_page(line: str) -> str:
        """
        Удаляет номер страницы из строки.

        Args:
            line (str): Строка с номером страницы

        Returns:
            str: Строка без номера страницы
        """
        return TAIL_PAGE_RE.sub("", line).strip(" .-;:")

    @staticmethod
    def is_author_like(line: str) -> bool:
        """
        Проверяет, является ли строка строкой с авторами.

        Args:
            line (str): Строка для проверки

        Returns:
            bool: True если строка содержит авторов
        """
        if not line:
            return False
        if AUTHOR_START_RE.search(line):
            return True
        if line.count(",") >= 2 and re.search(r"[А-ЯЁA-Z]\.\s?", line):
            return True
        return False

    @staticmethod
    def is_heading_start(line: str) -> bool:
        """
        Проверяет, является ли строка началом тематического заголовка.

        Args:
            line (str): Строка для проверки

        Returns:
            bool: True если строка - начало заголовка темы
        """
        line = TOCExtractor.normalize_line(line)
        if not line or PURE_PAGE_RE.fullmatch(line) or TOCExtractor.tail_page(line) is not None:
            return False
        if TOCExtractor.is_author_like(line):
            return False
        if any(ch.isdigit() for ch in line):
            return False
        if not re.match(rf"^[{RU_UP}]", line):
            return False

        low = line.lower()
        if EDITORIAL_RE.search(low):
            return False

        wc = len(line.split())
        if wc < 2 or wc > 12:
            return False
        if len(line) > 120:
            return False
        if line.endswith(".") or ":" in line:
            return False

        if not any(k in low for k in TOPIC_KEYWORDS):
            return False

        return True

    @staticmethod
    def is_heading_cont(line: str, active: bool) -> bool:
        """
        Проверяет, является ли строка продолжением тематического заголовка.

        Args:
            line (str): Строка для проверки
            active (bool): Активен ли режим сбора заголовка

        Returns:
            bool: True если строка - продолжение заголовка
        """
        if not active:
            return False

        line = TOCExtractor.normalize_line(line)
        if not line or PURE_PAGE_RE.fullmatch(line) or TOCExtractor.tail_page(line) is not None:
            return False
        if TOCExtractor.is_author_like(line):
            return False
        if any(ch.isdigit() for ch in line):
            return False
        if not re.match(rf"^[{RU_LOW}]", line):
            return False

        return 1 <= len(line.split()) <= 8

    @staticmethod
    def is_bad_topic(topic: str) -> bool:
        """
        Проверяет, является ли тема некорректной.

        Args:
            topic (str): Название темы

        Returns:
            bool: True если тема некорректна
        """
        t = TOCExtractor.normalize_line(topic)
        if not t:
            return True
        low = t.lower()

        if len(t) > 120 or len(t.split()) > 14:
            return True
        if any(ch.isdigit() for ch in t):
            return True
        if EDITORIAL_RE.search(low):
            return True
        if AUTHOR_IN_TOPIC_RE.search(t):
            return True
        if START_INITIALS_RE.search(t):
            return True
        if LOWER_START_RE.search(t):
            return True
        if t.endswith("."):
            return True
        if not any(k in low for k in TOPIC_KEYWORDS):
            return True

        return False

    @staticmethod
    def extract_toc_lines(pdf_path: Path, max_pages: int = 30):
        """
        Извлекает строки из раздела "Содержание" PDF.

        Args:
            pdf_path (Path): Путь к PDF файлу
            max_pages (int): Максимальное количество страниц для поиска

        Returns:
            list: Список строк из содержания
        """
        lines = []
        collecting = False

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:max_pages]:
                text = page.extract_text() or page.extract_text(layout=True) or ""
                page_lines = [TOCExtractor.normalize_line(x) for x in text.splitlines() if TOCExtractor.normalize_line(x)]
                if not page_lines:
                    continue

                # Проверяем, не начались ли статьи (УДК)
                if collecting and UDK_PAGE_RE.search("\n".join(page_lines[:35])):
                    break

                # Ищем начало содержания
                if not collecting:
                    start_idx = None
                    for i, line in enumerate(page_lines):
                        if line.lower() == RUS_CONTENTS.lower():
                            start_idx = i + 1
                            break
                    if start_idx is None:
                        continue
                    collecting = True
                    page_lines = page_lines[start_idx:]

                for line in page_lines:
                    low = line.lower()
                    if low == "content":
                        return lines
                    if RUS_JOURNAL_FULL in low or ENG_JOURNAL_FULL in low:
                        continue
                    lines.append(line)

        return lines

    @staticmethod
    def parse_toc_entries(pdf_path: Path):
        """
        Парсит записи из содержания PDF в структурированный формат.

        Args:
            pdf_path (Path): Путь к PDF файлу

        Returns:
            list: Список записей с полями topic, toc_title, toc_page
        """
        lines = TOCExtractor.extract_toc_lines(pdf_path)
        entries = []

        current_topic = ""
        topic_parts = []
        topic_active = False
        entry_parts = []

        def flush_entry(page_num: int):
            nonlocal entry_parts, entries, current_topic
            if page_num is None:
                return

            title = TOCExtractor.normalize_line(" ".join(entry_parts))
            title = TOCExtractor.strip_page(title)

            if title:
                entries.append({
                    "topic": current_topic,
                    "toc_title": title,
                    "toc_page": int(page_num),
                })

            entry_parts = []

        for line in lines:
            line_is_heading_start = TOCExtractor.is_heading_start(line)
            line_is_heading_cont = TOCExtractor.is_heading_cont(line, topic_active)

            if not entry_parts:
                if line_is_heading_start:
                    topic_parts = [line]
                    current_topic = TOCExtractor.normalize_line(" ".join(topic_parts))
                    topic_active = True
                    continue
                if line_is_heading_cont or (topic_active and RUS_PREP_END_RE.search(" ".join(topic_parts))):
                    topic_parts.append(line)
                    current_topic = TOCExtractor.normalize_line(" ".join(topic_parts))
                    continue

            page_pure = int(line) if PURE_PAGE_RE.fullmatch(line) else None
            if page_pure is not None:
                if entry_parts:
                    flush_entry(page_pure)
                topic_active = False
                continue

            page_tail = TOCExtractor.tail_page(line)
            if page_tail is not None:
                body = TOCExtractor.strip_page(line)
                if body:
                    entry_parts.append(body)
                if entry_parts:
                    flush_entry(page_tail)
                topic_active = False
                continue

            if line_is_heading_start and entry_parts:
                entry_parts = []
                topic_parts = [line]
                current_topic = TOCExtractor.normalize_line(" ".join(topic_parts))
                topic_active = True
                continue

            entry_parts.append(line)
            topic_active = False

        # Нормализуем записи
        for entry in entries:
            entry["topic"] = TOCExtractor.normalize_line(entry.get("topic", ""))
            entry["toc_title"] = TOCExtractor.normalize_line(entry.get("toc_title", ""))

        return entries

    @staticmethod
    def sanitize_toc_topics(toc_entries):
        """
        Очищает и валидирует темы из содержания.

        Args:
            toc_entries (list): Записи из содержания

        Returns:
            tuple: (очищенные записи, флаг no_theme_mode)
        """
        entries = [dict(e) for e in toc_entries]

        # Удаляем темы, которые слишком похожи на заголовки статей
        for e in entries:
            topic = TOCExtractor.normalize_line(e.get("topic", ""))
            toc_title = TOCExtractor.normalize_line(e.get("toc_title", ""))

            if topic and toc_title and SequenceMatcher(None, topic.lower(), toc_title.lower()).ratio() >= 0.82:
                topic = ""

            if topic and TOCExtractor.is_bad_topic(topic):
                topic = ""

            e["topic"] = topic
            e["toc_title"] = toc_title

        # Подсчитываем частоту тем
        counts = Counter(e["topic"] for e in entries if e["topic"])

        # Удаляем уникальные длинные темы
        for e in entries:
            topic = e.get("topic", "")
            if not topic:
                continue
            if counts.get(topic, 0) == 1 and (len(topic) > 80 or len(topic.split()) > 10):
                e["topic"] = ""

        # Определяем режим "без тем"
        counts = Counter(e["topic"] for e in entries if e["topic"])
        uniq_n = len(counts)
        max_cnt = max(counts.values()) if counts else 0

        no_theme_mode = False
        if uniq_n == 0:
            no_theme_mode = True
        elif uniq_n >= max(6, int(len(entries) * 0.60)):
            no_theme_mode = True
        elif uniq_n >= 5 and max_cnt <= 2:
            no_theme_mode = True
        elif uniq_n >= 3 and max_cnt <= 1 and len(entries) >= 10:
            no_theme_mode = True
        elif uniq_n <= 2 and max_cnt <= 2 and len(entries) >= 10:
            no_theme_mode = True
        elif uniq_n == 1 and max_cnt <= max(3, int(len(entries) * 0.35)):
            no_theme_mode = True

        if no_theme_mode:
            for e in entries:
                e["topic"] = ""

        return entries, no_theme_mode
