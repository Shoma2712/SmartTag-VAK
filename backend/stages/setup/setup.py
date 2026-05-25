# БЛОК 1: Настройка окружения и инициализация проекта
import os
import warnings
import logging
import nltk
from pathlib import Path

# Отключение предупреждений
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*datetime.utcnow.*")
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", message=".*multi-threaded, use of fork().*")

# Настройка логирования
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdffont").setLevel(logging.ERROR)


def download_nltk_resources():
    """
    Загружает необходимые ресурсы NLTK для обработки текста.

    Returns:
        bool: True если ресурсы загружены успешно
    """
    try:
        nltk.data.find('corpora/stopwords')
        print("✓ NLTK stopwords уже загружены")
        return True
    except LookupError:
        print("Загрузка NLTK stopwords...")
        nltk.download('stopwords')
        print("✓ NLTK stopwords загружены")
        return True


def setup_project_structure(base_dir=None):
    """
    Создает структуру папок для хранения данных проекта.

    Args:
        base_dir (str | Path | None): Базовая директория project_data (по умолчанию — корень репозитория)

    Returns:
        dict: Словарь с путями к основным директориям
    """
    if base_dir is None:
        from backend.config import PROJECT_ROOT

        BASE_DIR = str(PROJECT_ROOT / "project_data")
    else:
        BASE_DIR = str(base_dir)
    PDF_DIR = os.path.join(BASE_DIR, "pdfs")
    PDF_IMT = os.path.join(PDF_DIR, "imt")
    OUTPUT_IMT = os.path.join(BASE_DIR, "dataset_IMT.csv")

    # Создаем директории
    os.makedirs(PDF_IMT, exist_ok=True)

    print(f"✓ Рабочая папка: {BASE_DIR}")
    print(f"✓ IMT PDF: {PDF_IMT}")

    return {
        'BASE_DIR': BASE_DIR,
        'PDF_DIR': PDF_DIR,
        'PDF_IMT': PDF_IMT,
        'OUTPUT_IMT': OUTPUT_IMT,
    }


def initialize_project():
    """
    Главная функция инициализации проекта.

    Returns:
        dict: Словарь с путями к основным директориям
    """
    print("=" * 80)
    print("ИНИЦИАЛИЗАЦИЯ ПРОЕКТА")
    print("=" * 80)
    print()

    # Загружаем ресурсы NLTK
    download_nltk_resources()

    # Создаем структуру папок
    paths = setup_project_structure()

    print()
    print("✓ Проект инициализирован успешно!")
    print("=" * 80)

    return paths


if __name__ == "__main__":
    paths = initialize_project()
