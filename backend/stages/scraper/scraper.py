# БЛОК 2: Скрейпер для сбора PDF файлов с сайта IMT Journal
import os
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin


class IMTScraper:
    """
    Класс для скачивания PDF файлов с сайта журнала IMT.
    """

    def __init__(self, pdf_dir: str):
        """
        Инициализация скрейпера.

        Аргументы:
            pdf_dir (str): Путь к директории для сохранения PDF файлов
        """
        self.base_url = "https://imt-journal.ru"
        self.archive_url = f"{self.base_url}/archive"
        self.pdf_dir = pdf_dir
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def download_pdf(self, pdf_url: str) -> bool:
        """
        Скачивает PDF файл по указанному URL.

        Аргументы:
            pdf_url (str): URL PDF файла

        Возвращает:
            bool: True если файл скачан, False если уже существует или произошла ошибка
        """
        filename = os.path.join(self.pdf_dir, pdf_url.split("/")[-1])

        # Проверяем, существует ли файл
        if os.path.exists(filename):
            print(f"Файл уже есть: {filename}")
            return False

        try:
            print(f"Скачивание: {pdf_url.split('/')[-1]}")
            response = requests.get(pdf_url, headers=self.headers, timeout=30)
            response.raise_for_status()

            with open(filename, "wb") as f:
                f.write(response.content)

            return True

        except Exception as e:
            print(f"Ошибка скачивания {pdf_url}: {e}")
            return False

    def get_pdfs_from_page(self, page_url: str) -> list:
        """
        Извлекает все ссылки на PDF файлы со страницы.

        Аргументы:
            page_url (str): URL страницы для парсинга

        Возвращает:
            list: Список URL-ов PDF файлов
        """
        pdf_urls = []

        try:
            response = requests.get(page_url, headers=self.headers, timeout=30)
            soup = BeautifulSoup(response.text, "html.parser")

            for a in soup.find_all("a", href=True):
                if a['href'].endswith(".pdf"):
                    full_url = urljoin(self.base_url, a['href'])
                    if full_url not in pdf_urls:
                        pdf_urls.append(full_url)

        except Exception as e:
            print(f"Ошибка страницы {page_url}: {e}")

        return pdf_urls

    def find_next_page(self, current_page_url: str):
        """
        Находит ссылку на следующую страницу пагинации.

        Аргументы:
            current_page_url (str): URL текущей страницы

        Возвращает:
            str: URL следующей страницы или None если следующей страницы нет
        """
        try:
            response = requests.get(current_page_url, headers=self.headers, timeout=30)
            soup = BeautifulSoup(response.text, "html.parser")

            # Ищем пагинацию
            next_link = None
            pagination = soup.find(class_='pagination')

            if pagination:
                curr = pagination.find('span', class_='current')
                if curr and curr.find_next_sibling('a'):
                    next_link = curr.find_next_sibling('a')['href']

            # Альтернативный поиск
            if not next_link:
                for a in soup.find_all('a', href=True):
                    if a.get_text(strip=True) in ['>', '»', 'Вперед', 'Next', 'Следующая']:
                        next_link = a['href']
                        break

            if next_link:
                return urljoin(current_page_url, next_link)

        except Exception as e:
            print(f"Ошибка пагинации: {e}")

        return None

    def run(self, max_pages: int = 50, delay: float = 0.5):
        """
        Запускает процесс сбора PDF файлов.

        Аргументы:
            max_pages (int): Максимальное количество страниц для обхода
            delay (float): Задержка между запросами в секундах

        Возвращает:
            dict: Статистика сбора (количество страниц, скачанных файлов)
        """
        print("=" * 80)
        print("ЗАПУСК СБОРА PDF")
        print("=" * 80)

        visited_pages = set()
        current_page = self.archive_url
        page_num = 1
        total_downloaded = 0

        while current_page and current_page not in visited_pages and page_num <= max_pages:
            print(f"\nСтраница {page_num}: {current_page}")
            visited_pages.add(current_page)

            # Получаем PDF со страницы
            pdfs = self.get_pdfs_from_page(current_page)
            print(f"Найдено PDF: {len(pdfs)}")

            # Скачиваем каждый PDF
            for pdf_url in pdfs:
                if self.download_pdf(pdf_url):
                    total_downloaded += 1
                time.sleep(delay)  # Пауза, чтобы не перегружать сервер

            # Ищем следующую страницу
            next_page = self.find_next_page(current_page)

            if next_page and next_page != current_page and next_page not in visited_pages:
                current_page = next_page
                page_num += 1
            else:
                print("Последняя страница.")
                break

        print("\n" + "=" * 80)
        print("СБОР ЗАВЕРШЕН")
        print("=" * 80)
        print(f"Обработано страниц: {page_num}")
        print(f"Скачано новых файлов: {total_downloaded}")

        return {
            'pages_processed': page_num,
            'files_downloaded': total_downloaded
        }


def run_scraper(pdf_dir: str, max_pages: int = 50):
    """
    Удобная функция для запуска скрейпера.

    Аргументы:
        pdf_dir (str): Путь к директории для сохранения PDF
        max_pages (int): Максимальное количество страниц

    Возвращает:
        dict: Статистика сбора
    """
    scraper = IMTScraper(pdf_dir)
    return scraper.run(max_pages=max_pages)


if __name__ == "__main__":
    # Пример использования
    from pathlib import Path

    pdf_dir = Path("project_data/pdfs/imt")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    stats = run_scraper(str(pdf_dir), max_pages=50)
    print(f"\nИтого: {stats}")
