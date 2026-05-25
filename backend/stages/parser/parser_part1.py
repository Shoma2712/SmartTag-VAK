# БЛОК 3 - ЧАСТЬ 1: Обработка текста и парсинг УДК
import re
import pymorphy3
from nltk.corpus import stopwords


class TextProcessor:
    """
    Класс для обработки и очистки текста научных статей.
    """

    def __init__(self):
        """
        Инициализация процессора текста.
        """
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))

        # Добавляем специфичные для научных текстов стоп-слова
        academic_stopwords = [
            'это', 'наш', 'год', 'рис', 'табл', 'стр', 'см', 'статья', 'работа', 'исследование',
            'данные', 'результат', 'метод', 'основа', 'использование', 'показать', 'рассмотреть',
            'являться', 'предложить', 'провести', 'задача', 'рисунок', 'таблица', 'схема', 'цель',
            'проблема', 'решение', 'пример', 'случай', 'вид', 'тип', 'сравнение', 'оценка', 'вопрос',
            'подход', 'номер', 'онтология', 'применение', 'область', 'научный', 'новый', 'современный',
            'различный', 'получить', 'выполнить', 'представить', 'описать', 'существовать',
            'позволять', 'введение', 'заключение', 'вывод', 'список', 'литература',
            'библиография', 'аннотация', 'ключевые', 'слова', 'удк', 'doi', 'том', 'выпуск',
            'автор', 'данный', 'использовать', 'свойство', 'разработка', 'определение',
            'параметр', 'ключевой', 'который', 'свой', 'этот', 'также', 'каждый', 'мочь', 'весь'
        ]
        self.stop_words.update(academic_stopwords)

    def tokenize_for_lda(self, text):
        """
        Токенизация и лемматизация текста для LDA моделирования.

        Args:
            text (str): Исходный текст статьи

        Returns:
            List[str]: Список лемматизированных токенов (только существительные и прилагательные)
        """
        if not text:
            return []

        # Убираем переносы строк в словах
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Оставляем только кириллические символы
        text = re.sub(r'[^а-яёА-ЯЁ\s]', ' ', text)
        # Приводим к нижнему регистру и нормализуем пробелы
        text = re.sub(r'\s+', ' ', text).strip().lower()

        tokens = []
        for word in text.split():
            if len(word) <= 2:
                continue

            # Лемматизация
            p = self.morph.parse(word)[0]

            # Оставляем только существительные и прилагательные
            if p.tag.POS not in {'NOUN', 'ADJF'}:
                continue

            lemma = p.normal_form

            # Удаляем стоп-слова
            if lemma not in self.stop_words:
                tokens.append(lemma)

        return tokens

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Нормализация текста - удаление лишних пробелов и переносов.

        Args:
            text (str): Исходный текст

        Returns:
            str: Нормализованный текст
        """
        # Склеиваем слова, разорванные переносом строки
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        text = re.sub(r"\b([а-яё]{2,})-\s+([а-яё]{2,})\b", r"\1\2", text, flags=re.IGNORECASE)
        # Заменяем переносы строк на пробелы
        text = re.sub(r"\n+", " ", text)
        # Схлопываем множественные пробелы
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def strip_private(text: str) -> str:
        """
        Удаляет приватные символы Unicode.

        Args:
            text (str): Исходный текст

        Returns:
            str: Текст без приватных символов
        """
        return re.sub(r"[-]", " ", text)

    @staticmethod
    def is_garbage_line(line: str) -> bool:
        """
        Определяет, является ли строка мусором (метаданные, служебная информация).

        Args:
            line (str): Строка для проверки

        Returns:
            bool: True если строка является мусором
        """
        if not line or len(line.strip()) < 5:
            return True
        low = line.lower()
        return any(x in low for x in ["issn", "doi", "@", "orcid", "journal", "copyright"])

    @staticmethod
    def clean_main_text(text: str) -> str:
        """
        Глубокая очистка основного текста статьи от шума.

        Args:
            text (str): Исходный текст статьи

        Returns:
            str: Очищенный текст
        """
        # Убираем приватные символы
        text = TextProcessor.strip_private(text)
        # Удаляем строки с номерами страниц
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
        # Схлопываем множественные переносы строк
        text = re.sub(r"\n{2,}", "\n", text)

        # Фильтруем строки
        out = []
        for raw in text.split("\n"):
            line = raw.strip()
            if not line:
                continue
            out.append(line)

        text = "\n".join(out)
        # Удаляем специальные символы (кроме базовой пунктуации)
        text = re.sub(r"[^\w\s.,;:!?()\-«»\"'\/\\]", " ", text)
        # Нормализуем пробелы
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _looks_like_reference_line(line: str) -> bool:
        """
        Проверяет, похожа ли строка на запись из списка литературы.

        Args:
            line (str): Строка для проверки

        Returns:
            bool: True если строка похожа на ссылку
        """
        line = line.strip()
        # Проверяем формат нумерации
        if not re.match(r"^(?:\[\d{1,3}\]|\d{1,3}\.)\s+", line):
            return False

        body = re.sub(r"^(?:\[\d{1,3}\]|\d{1,3}\.)\s+", "", line).strip()
        if len(body) < 28:
            return False

        # Ищем признаки библиографической записи
        hints = 0
        if re.search(r"\b(?:19|20)\d{2}\b", body):
            hints += 1
        if re.search(r"\bdoi\b|https?://|www\.", body, flags=re.IGNORECASE):
            hints += 1
        if re.search(r"[A-ZА-ЯЁ]\.\s*[A-ZА-ЯЁ]\.", body):
            hints += 1
        if re.search(r"\b(?:journal|vol\.?|pp\.?|№|том|вып\.?|изд)\b", body, flags=re.IGNORECASE):
            hints += 1
        return hints > 0

    @staticmethod
    def cut_references_tail(text: str) -> str:
        """
        Обрезает список литературы в конце статьи.

        Args:
            text (str): Полный текст статьи

        Returns:
            str: Текст без списка литературы
        """
        if not text:
            return text

        # Паттерны для поиска начала списка литературы
        patterns = [
            r"(?im)^\s*СПИСОК\s+ЛИТЕРАТУРЫ\b",
            r"(?im)^\s*Список\s+литературы\b",
            r"(?im)^\s*СПИСОК\s+ИСТОЧНИКОВ\b",
            r"(?im)^\s*Список\s+источников\b",
            r"(?im)^\s*ЛИТЕРАТУРА\b",
            r"(?im)^\s*Литература\b",
            r"(?im)^\s*Библиография\b",
            r"(?im)^\s*Об\s+авторах\b",
            r"(?im)^\s*Сведения\s+об\s+авторах\b",
        ]

        cut_positions = []
        for pat in patterns:
            for m in re.finditer(pat, text):
                # Проверяем, что маркер во второй половине текста
                if m.start() > len(text) * 0.45:
                    cut_positions.append(m.start())

        # Ищем последовательность нумерованных строк в конце текста
        tail_start = int(len(text) * 0.75)
        numbered_line_pattern = r"(?m)^\s*(?:\[\d{1,3}\]|\d{1,3}\.)\s+[^\n]{20,}$"
        tail_fragment = text[tail_start:]

        reference_like_positions = []
        for m in re.finditer(numbered_line_pattern, tail_fragment):
            line = m.group(0)
            if TextProcessor._looks_like_reference_line(line):
                reference_like_positions.append(tail_start + m.start())

        # Если нашли 3+ последовательных ссылки, обрезаем
        if len(reference_like_positions) >= 3:
            for i in range(len(reference_like_positions) - 2):
                if reference_like_positions[i + 2] - reference_like_positions[i] < 1400:
                    cut_positions.append(reference_like_positions[i])
                    break

        if cut_positions:
            return text[:min(cut_positions)]
        return text


class UDCParser:
    """
    Класс для работы с УДК (Универсальная десятичная классификация).
    """

    @staticmethod
    def parse_udc(raw: str) -> str:
        """
        Извлекает первые цифры из строки УДК.

        Args:
            raw (str): Исходная строка УДК (например, "004.8+519.7")

        Returns:
            str: Первые цифры УДК (например, "004") или "N/A"
        """
        if not raw or raw == "N/A":
            return "N/A"
        m = re.match(r"(\d+)", raw.strip())
        return m.group(1) if m else raw.strip()


def build_text_for_embedding(title: str, annotation: str, keywords: str, main_text: str) -> str:
    """
    Строит полный текст для создания эмбеддингов.

    Args:
        title (str): Заголовок
        annotation (str): Аннотация
        keywords (str): Ключевые слова
        main_text (str): Основной текст

    Returns:
        str: Объединенный текст
    """
    parts = []
    if title:
        parts.append(TextProcessor.normalize_text(title))
    if annotation:
        parts.append(TextProcessor.normalize_text(annotation))
    if keywords:
        parts.append(TextProcessor.normalize_text(keywords))
    if main_text:
        parts.append(TextProcessor.normalize_text(main_text))
    return "\n\n".join(parts).strip()


def build_clean_text(text: str) -> str:
    """
    Строит очищенный текст без формул и специальных символов.

    Args:
        text (str): Исходный текст

    Returns:
        str: Очищенный текст
    """
    t = TextProcessor.normalize_text(text)
    if not t:
        return ""

    # Удаляем греческие символы
    t = re.sub(r"[Ͱ-Ͽἀ-῿]", " ", t)
    # Удаляем латиницу и цифры
    t = re.sub(r"[A-Za-z0-9]", " ", t)
    # Удаляем скобки
    t = re.sub(r"[(){}\[\]<>]", " ", t)
    # Удаляем математические символы
    t = re.sub(r"[=+*^_|~`#%$@&]", " ", t)

    # Очистка пунктуации
    t = re.sub(r"(?:/|\\|-)\s*\.", ".", t)
    t = re.sub(r"\.\s*(?:/|\\|-)", ".", t)
    t = re.sub(r"(?:\b[^\W\d_Ѐ-ӿԀ-ԯ]\b\s*[.,;:]\s*){2,}", " ", t, flags=re.UNICODE)
    t = re.sub(r"(?:\s*\.\s*){2,}", ". ", t)
    t = re.sub(r"([,;:!?])(?:\s*\1)+", r"\1", t)
    t = re.sub(r"[.,;:!?/\\-]{2,}", ". ", t)
    t = re.sub(r"(?:(?<=\s)|^)[.,;:!?/\\-]+(?=\s|$)", " ", t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"([.,;:!?])(?=\S)", r"\1 ", t)
    t = re.sub(r"(?:\s*\.\s*){2,}", ". ", t)

    t = TextProcessor.normalize_text(t)
    return t.strip(" .,:;!?/-")
