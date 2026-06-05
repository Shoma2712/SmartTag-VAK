# БЛОК 3 - ЧАСТЬ 3: Сопоставление статей с темами и главный парсер
import re
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd
import pdfplumber
from tqdm import tqdm

from .parser_part1 import TextProcessor, UDCParser, build_text_for_embedding, build_clean_text
from .parser_part2 import TOCExtractor, TOPIC_NO_CONTENTS, TOPIC_UNDEFINED


class TopicMatcher:
    """
    Класс для сопоставления статей с темами из содержания.
    """

    @staticmethod
    def tokenize(text: str):
        """
        Токенизация текста для сопоставления заголовков.

        Args:
            text (str): Исходный текст

        Returns:
            list: Список токенов (слова длиной ≥3 символа)
        """
        text = str(text or "").lower().replace("ё", "е")
        text = re.sub(r"[^0-9a-zа-я\s-]", " ", text)
        text = text.replace("-", " ")
        return [token for token in text.split() if len(token) >= 3]

    @staticmethod
    def title_score(a: str, b: str) -> float:
        """
        Вычисляет схожесть двух заголовков.

        Args:
            a (str): Первый заголовок
            b (str): Второй заголовок

        Returns:
            float: Оценка схожести от 0 до 1
        """
        ta = set(TopicMatcher.tokenize(a))
        tb = set(TopicMatcher.tokenize(b))
        if not ta or not tb:
            return 0.0

        overlap = len(ta & tb) / max(1, len(ta))
        seq = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
        return 0.7 * overlap + 0.3 * seq

    @staticmethod
    def build_topic_points(toc_entries):
        """
        Строит список точек смены тем (страница -> тема).

        Args:
            toc_entries (list): Записи из содержания

        Returns:
            list: Список кортежей (номер_страницы, тема)
        """
        points = []
        last_topic = None

        for e in sorted(toc_entries, key=lambda x: int(x.get("toc_page") or 0)):
            topic = TOCExtractor.normalize_line(e.get("topic", ""))
            page = e.get("toc_page")
            if not topic or not page:
                continue
            page = int(page)
            if topic != last_topic:
                points.append((page, topic))
                last_topic = topic

        return points

    @staticmethod
    def topic_by_page(pdf_page, points):
        """
        Определяет тему по номеру страницы.

        Args:
            pdf_page (int): Номер страницы
            points (list): Список точек смены тем

        Returns:
            str: Название темы
        """
        if not points or pdf_page is None:
            return ""

        topic = points[0][1]
        for page, t in points:
            if int(pdf_page) >= page:
                topic = t
            else:
                break
        return topic

    @staticmethod
    def assign_topics_from_toc(rows, toc_entries, no_theme_mode):
        """
        Назначает темы статьям на основе содержания.

        Args:
            rows (list): Список статей с полями title, pdf_page
            toc_entries (list): Записи из содержания
            no_theme_mode (bool): Режим без тем

        Returns:
            list: Статьи с добавленными полями topic, topic_match_score
        """
        topic_points = TopicMatcher.build_topic_points(toc_entries)
        pointer = 0

        for row in rows:
            title = str(row.get("title", "") or "")
            pdf_page = row.get("pdf_page")

            best_idx = -1
            best_score = 0.0

            # Локальный поиск (ближайшие 22 записи)
            for idx in range(pointer, min(len(toc_entries), pointer + 22)):
                entry = toc_entries[idx]
                score = TopicMatcher.title_score(title, entry.get("toc_title", ""))

                # Бонус за близость страниц
                toc_page = entry.get("toc_page")
                if pdf_page and toc_page:
                    dist = abs(int(pdf_page) - int(toc_page))
                    score += max(0.0, 0.08 - min(dist, 20) * 0.004)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # Глобальный поиск, если локальный не дал результата
            if best_idx == -1 or best_score < 0.34:
                for idx, entry in enumerate(toc_entries):
                    score = TopicMatcher.title_score(title, entry.get("toc_title", ""))

                    toc_page = entry.get("toc_page")
                    if pdf_page and toc_page:
                        dist = abs(int(pdf_page) - int(toc_page))
                        score += max(0.0, 0.05 - min(dist, 25) * 0.002)

                    if score > best_score:
                        best_score = score
                        best_idx = idx

            topic = ""
            matched_toc_title = ""

            if best_idx != -1 and best_score >= 0.25:
                matched = toc_entries[best_idx]
                topic = TOCExtractor.normalize_line(matched.get("topic", ""))
                matched_toc_title = matched.get("toc_title", "")
                pointer = max(pointer, best_idx)

            # Определение темы по странице
            if not topic and not no_theme_mode:
                topic = TopicMatcher.topic_by_page(pdf_page, topic_points)

            if not topic and no_theme_mode:
                topic = TOPIC_NO_CONTENTS

            if not topic:
                topic = TOPIC_UNDEFINED

            row["topic"] = topic
            row["topic_match_score"] = round(float(best_score), 4)
            row["topic_match_toc_title"] = matched_toc_title

        return rows


class IMTArticleParser:
    """
    Главный класс для парсинга статей из PDF файлов IMT.
    """

    def __init__(self):
        """
        Инициализация парсера.
        """
        self.text_processor = TextProcessor()
        self.udc_parser = UDCParser()
        self.toc_extractor = TOCExtractor()
        self.topic_matcher = TopicMatcher()

    @staticmethod
    def extract_pages_with_numbers(pdf_path: Path):
        """
        Извлекает текст из PDF постранично с номерами страниц.

        Args:
            pdf_path (Path): Путь к PDF файлу

        Returns:
            list: Список кортежей (номер_страницы, текст)
        """
        pages = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    pdf_page_num = idx + 1
                    w, h = page.width, page.height
                    # Обрезаем верх и низ для удаления колонтитулов
                    cropped = page.crop((0, 45, w, max(0, h - 45)))
                    text = cropped.extract_text()
                    if not text:
                        text = cropped.extract_text(layout=True)
                    if not text:
                        text = page.extract_text()
                    if text:
                        pages.append((pdf_page_num, text))
        except Exception as exc:
            print(f"[WARN] PDF read error {pdf_path.name}: {exc}")
        return pages

    @staticmethod
    def is_author_line(line: str) -> bool:
        """
        Проверяет, является ли строка строкой с авторами.

        Args:
            line (str): Строка для проверки

        Returns:
            bool: True если строка содержит ФИО автора
        """
        return bool(re.match(r"[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+", line))

    def get_title(self, chunk: str) -> str:
        """
        Извлекает заголовок статьи из текстового блока.

        Args:
            chunk (str): Текстовый блок статьи

        Returns:
            str: Заголовок статьи
        """
        lines = chunk.split("\n")
        after_udc, collecting = False, False
        buf = []

        for line in lines:
            line = line.strip()
            if "УДК" in line:
                after_udc = True
                continue
            if not after_udc:
                continue
            if self.is_author_line(line) or "аннотация" in line.lower():
                break
            if self.text_processor.is_garbage_line(line):
                continue
            if not collecting and len(line.split()) >= 2:
                collecting = True
            if collecting:
                buf.append(line)
            if len(buf) >= 4:
                break

        return " ".join(buf).strip()

    @staticmethod
    def get_authors(chunk: str) -> str:
        """
        Извлекает авторов статьи.

        Args:
            chunk (str): Текстовый блок статьи

        Returns:
            str: Строка с авторами через запятую
        """
        pat = re.compile(r"([А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+)")
        for line in chunk.split("\n"):
            line = line.strip()
            if not line or "аннотация" in line.lower():
                break
            if "@" in line:
                continue
            ms = pat.findall(line)
            if ms:
                return ", ".join(ms)
        return ""

    def extract_main_text(self, chunk: str) -> str:
        """
        Извлекает основной текст статьи.

        Args:
            chunk (str): Текстовый блок статьи

        Returns:
            str: Основной текст статьи
        """
        # Ищем начало (Введение)
        m = re.search(r"(?:\n|^)\s*(?:1\.?)?\s*(?:Введение|Introduction)(?:\.|:)?\s+", chunk, re.IGNORECASE)
        if m:
            raw = chunk[m.end():]
        else:
            # Альтернативный поиск после ключевых слов
            fb = re.search(r"(Ключевые слова|Keywords|Цитирование|Citation).*?(\n\s*\n|\n[А-Я])", chunk, re.DOTALL | re.IGNORECASE)
            if not fb:
                return ""
            raw = chunk[fb.end():]

        # Обрезаем список литературы
        ref = re.search(r"(?:\n|^)\s*(?:СПИСОК ЛИТЕРАТУРЫ|Список источников|REFERENCES|Библиографический список)", raw, re.IGNORECASE)
        raw = raw[:ref.start()] if ref else self.text_processor.cut_references_tail(raw)

        return self.text_processor.clean_main_text(raw)

    @staticmethod
    def extract_annotation_and_keywords(chunk: str):
        """
        Извлекает аннотацию и ключевые слова.

        Args:
            chunk (str): Текстовый блок статьи

        Returns:
            tuple: (аннотация, ключевые_слова)
        """
        annotation = ""
        keywords = ""

        # Аннотация
        ann_patterns = [
            r"Аннотация\s*[:.]?\s*(.*?)(?=\bКлючевые\s+слова\b|\bKeywords\b|\bВведение\b|\bIntroduction\b|\bЦитирование\b|\bCitation\b)",
            r"Аннотация\s*[:.]?\s*(.*?)(?=\n\s*\n)",
        ]
        for pattern in ann_patterns:
            m = re.search(pattern, chunk, flags=re.IGNORECASE | re.DOTALL)
            if m:
                annotation = TextProcessor.normalize_text(m.group(1))
                if annotation:
                    break

        # Ключевые слова
        kw_patterns = [
            r"Ключевые\s+слова\s*[:.]?\s*(.*?)(?=\bЦитирование\b|\bCitation\b|\bВведение\b|\bIntroduction\b|\bСПИСОК\b|\bREFERENCES\b)",
            r"Ключевые\s+слова\s*[:.]?\s*(.*?)(?=\n\s*\n)",
        ]
        for pattern in kw_patterns:
            m = re.search(pattern, chunk, flags=re.IGNORECASE | re.DOTALL)
            if m:
                keywords = TextProcessor.normalize_text(m.group(1))
                if keywords:
                    break

        return annotation, keywords

    def parse_articles_from_pdf(self, pages_with_nums, filename: str):
        """
        Парсит статьи из страниц PDF.

        Args:
            pages_with_nums (list): Страницы с номерами
            filename (str): Имя PDF файла

        Returns:
            list: Список статей
        """
        rows = []

        # Объединяем все страницы с маркерами
        full_text_parts = []
        for pdf_page_num, text in pages_with_nums:
            full_text_parts.append(f"__PAGE_{pdf_page_num}__\n{text}")
        full_text = "\n".join(full_text_parts)

        # Разбиваем по УДК
        chunks = re.split(r"\n(?=УДК\s+[\d\.\/\+\:]+)", full_text)

        for chunk in chunks:
            if len(chunk) < 1000:
                continue

            # Извлекаем номер страницы
            page_match = re.search(r"__PAGE_(\d+)__", chunk)
            pdf_page_num = int(page_match.group(1)) if page_match else None

            # УДК
            udc_match = re.search(r"УДК\s+([\d\.\/\+\:]+)", chunk)
            udc_raw = udc_match.group(1) if udc_match else "N/A"
            udc = self.udc_parser.parse_udc(udc_raw)

            # Заголовок и авторы
            title = TextProcessor.normalize_text(self.get_title(chunk))
            authors = self.get_authors(chunk)

            # Аннотация и ключевые слова
            annotation, keywords = self.extract_annotation_and_keywords(chunk)

            # Основной текст
            main_text = self.extract_main_text(chunk)
            if len(main_text.split()) < 120:
                continue

            # Проверка обязательных полей
            if not title or not annotation or not keywords:
                continue

            # Токенизация для LDA
            lda_tokens = self.text_processor.tokenize_for_lda(main_text)

            # Полный текст для эмбеддингов
            full_text_emb = build_text_for_embedding(title, annotation, keywords, main_text)

            # Очищенный текст
            clean_text = build_clean_text(main_text)

            rows.append({
                "source": filename,
                "udc": udc,
                "title": title,
                "authors": authors,
                "annotation": annotation,
                "keywords": keywords,
                "main_text": main_text,
                "lda_tokens": lda_tokens,
                "clean_text": clean_text,
                "full_text": full_text_emb
            })

        return rows

    def parse_folder(self, pdf_folder: Path, db_conn=None):
        """
        Парсит все PDF файлы в папке и сохраняет результат в базу данных.

        Args:
            pdf_folder (Path): Путь к папке с PDF файлами
            db_conn: SQLite connection object (если None, возвращает DataFrame без сохранения)

        Returns:
            pd.DataFrame: DataFrame со всеми статьями
        """
        pdf_files = sorted(p for p in pdf_folder.iterdir() if p.suffix.lower() == ".pdf")
        if not pdf_files:
            print(f"[WARN] В папке нет PDF: {pdf_folder}")
            return pd.DataFrame()

        all_rows = []

        for pdf_path in tqdm(pdf_files, desc="Parsing IMT PDFs"):
            # Извлекаем содержание
            toc_raw = self.toc_extractor.parse_toc_entries(pdf_path)
            toc_entries, no_theme_mode = self.toc_extractor.sanitize_toc_topics(toc_raw)

            # Извлекаем страницы
            pages_with_nums = self.extract_pages_with_numbers(pdf_path)
            if not pages_with_nums:
                continue

            # Парсим статьи
            rows = self.parse_articles_from_pdf(pages_with_nums, pdf_path.name)

            # Назначаем темы
            rows = self.topic_matcher.assign_topics_from_toc(rows, toc_entries, no_theme_mode)

            all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        if not df.empty:
            # Удаляем служебные колонки
            drop_columns = ["pdf_page", "journal", "topic_match_toc_title"]
            df = df.drop(columns=[c for c in drop_columns if c in df.columns])

            # Сохраняем в базу данных если передано соединение
            if db_conn is not None:
                from backend.database import insert_articles_batch
                
                # Подготавливаем данные для вставки (только поля таблицы articles)
                articles_list = []
                for _, row in df.iterrows():
                    articles_list.append({
                        'source': row.get('source'),  # Имя PDF файла
                        'title': row.get('title'),
                        'annotation': row.get('annotation'),
                        'keywords': row.get('keywords'),
                        'main_text': row.get('main_text'),
                        'udc': row.get('udc'),
                        'authors': row.get('authors'),
                        'lda_tokens': row.get('lda_tokens'),
                        'topic': row.get('topic')
                    })
                
                try:
                    count = insert_articles_batch(db_conn, articles_list)
                    print(f"\n✓ Парсинг завершен: {len(df)} статей")
                    print(f"✓ Сохранено в базу данных: {count} записей")
                    print(f"✓ Уникальных УДК: {df['udc'].nunique()}")
                    print(f"✓ Уникальных тем: {df['topic'].nunique()}")
                except Exception as e:
                    print(f"❌ Ошибка при сохранении в БД: {e}")
                    raise

        return df


def parse_imt_folder(pdf_folder: str, db_conn=None):
    """
    Упрощенная функция для парсинга папки с PDF.

    Args:
        pdf_folder (str): Путь к папке с PDF
        db_conn: SQLite connection object (если None, возвращает DataFrame без сохранения)

    Returns:
        pd.DataFrame: DataFrame со статьями
    """
    parser = IMTArticleParser()
    return parser.parse_folder(Path(pdf_folder), db_conn)


if __name__ == "__main__":
    from pathlib import Path

    pdf_folder = Path("project_data/pdfs/imt")
    output_csv = Path("project_data/dataset_IMT.csv")

    df = parse_imt_folder(str(pdf_folder), str(output_csv))
    print(f"\nПарсинг завершен: {len(df)} статей")
