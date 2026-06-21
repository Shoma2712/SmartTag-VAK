# БЛОК 4: Тематическое моделирование LDA
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging
import sqlite3

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.phrases import Phrases, Phraser

# Импорт database functions
from backend.database import get_connection, get_all_articles, insert_lda_results_batch
from backend.config import DEFAULT_DATABASE_PATH

# Отключаем логирование gensim
logging.getLogger("gensim").setLevel(logging.ERROR)


def ensure_tokens(x):
    """
    Преобразует строковое представление списка в список.

    Аргументы:
        x: Значение (может быть list или str)

    Возвращает:
        list: Список токенов
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def build_ngrams(docs, min_count_bigram=5, threshold_bigram=20, min_count_trigram=3, threshold_trigram=30):
    """
    Строит биграммы и триграммы из документов.

    Аргументы:
        docs (list): Список документов (каждый документ - список токенов)
        min_count_bigram (int): Минимальная частота для биграмм
        threshold_bigram (int): Порог для биграмм
        min_count_trigram (int): Минимальная частота для триграмм
        threshold_trigram (int): Порог для триграмм

    Возвращает:
        tuple: (документы_с_nграммами, bigram_модель, trigram_модель)
    """
    bigram = Phrases(docs, min_count=min_count_bigram, threshold=threshold_bigram)
    bigram_mod = Phraser(bigram)

    trigram = Phrases(bigram_mod[docs], min_count=min_count_trigram, threshold=threshold_trigram)
    trigram_mod = Phraser(trigram)

    docs_ngrams = [trigram_mod[bigram_mod[doc]] for doc in docs]
    return docs_ngrams, bigram_mod, trigram_mod


def compute_coherence_grid(dictionary, corpus, texts, start=4, limit=20, step=1):
    """
    Вычисляет coherence для разного количества тем.

    Аргументы:
        dictionary: Словарь gensim
        corpus: Корпус документов
        texts: Тексты для вычисления coherence
        start (int): Начальное количество тем
        limit (int): Максимальное количество тем
        step (int): Шаг

    Возвращает:
        tuple: (список_количеств_тем, список_моделей, список_coherence)
    """
    model_list = []
    coherence_values = []
    topic_range = list(range(start, limit + 1, step))

    for k in topic_range:
        print(f"Обучение LDA: {k} тем")
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            chunksize=2000,
            passes=15,
            iterations=300,
            alpha="asymmetric",
            eta="auto",
            workers=4,
            per_word_topics=False,
        )

        cm = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )

        coherence = cm.get_coherence()
        print(f"  coherence(c_v)={coherence:.4f}")

        model_list.append(model)
        coherence_values.append(coherence)

    return topic_range, model_list, coherence_values


class LDATopicModeler:
    """
    Класс для тематического моделирования с помощью LDA.
    """

    def __init__(self, df, tokens_column='lda_tokens', max_tokens_per_doc=800):
        """
        Инициализация модели LDA.

        Аргументы:
            df (pd.DataFrame): DataFrame с данными
            tokens_column (str): Название колонки с токенами
            max_tokens_per_doc (int): Максимальное количество токенов на документ
        """
        self.df = df.copy()
        self.tokens_column = tokens_column
        self.max_tokens_per_doc = max_tokens_per_doc

        # Подготовка данных
        self._prepare_data()

    def _prepare_data(self):
        """
        Подготовка данных для LDA.
        """
        print("Подготовка данных...")

        # Преобразуем токены
        self.df[self.tokens_column] = self.df[self.tokens_column].apply(ensure_tokens)

        # Фильтруем документы с малым количеством токенов
        self.df = self.df[self.df[self.tokens_column].apply(
            lambda t: isinstance(t, list) and len(t) > 20
        )].reset_index(drop=True)

        if self.df.empty:
            raise ValueError("После фильтрации по токенам не осталось статей для LDA")

        print(f"Статей для LDA: {len(self.df)}")

        # Ограничиваем количество токенов
        self.texts = [doc[:self.max_tokens_per_doc] for doc in self.df[self.tokens_column]]

    def build_dictionary_and_corpus(self, no_below=5, no_above=0.6, keep_n=20000):
        """
        Строит словарь и корпус для LDA.

        Аргументы:
            no_below (int): Минимальная частота слова в документах
            no_above (float): Максимальная доля документов со словом
            keep_n (int): Максимальный размер словаря

        Возвращает:
            tuple: (словарь, корпус, тексты_с_nграммами)
        """
        print("Построение биграмм и триграмм...")
        data_ready, self.bigram_mod, self.trigram_mod = build_ngrams(self.texts)

        print("Построение словаря и корпуса...")
        self.id2word = corpora.Dictionary(data_ready)
        self.id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        self.corpus = [self.id2word.doc2bow(doc) for doc in data_ready]
        self.data_ready = data_ready

        if len(self.id2word) == 0:
            raise ValueError("Словарь пуст после filter_extremes, ослабьте фильтры no_below/no_above")

        print(f"Размер словаря: {len(self.id2word)}")

        return self.id2word, self.corpus, self.data_ready

    def find_optimal_topics(self, start=4, limit=20, step=1):
        """
        Находит оптимальное количество тем по coherence.

        Аргументы:
            start (int): Начальное количество тем
            limit (int): Максимальное количество тем
            step (int): Шаг

        Возвращает:
            tuple: (topic_range, model_list, coherence_values)
        """
        print("Поиск оптимального числа тем...")
        topic_range, model_list, coherence_values = compute_coherence_grid(
            self.id2word,
            self.corpus,
            self.data_ready,
            start=start,
            limit=limit,
            step=step,
        )

        self.topic_range = topic_range
        self.model_list = model_list
        self.coherence_values = coherence_values

        # Выбираем лучшую модель
        best_idx = int(np.argmax(coherence_values))
        self.best_num_topics = topic_range[best_idx]
        self.best_coherence = float(coherence_values[best_idx])
        self.lda_model = model_list[best_idx]

        print("\nЛУЧШАЯ МОДЕЛЬ")
        print(f"- Количество тем: {self.best_num_topics}")
        print(f"- Coherence (c_v): {self.best_coherence:.4f}")

        return topic_range, model_list, coherence_values

    def plot_coherence(self, save_path=None):
        """
        Строит график coherence от количества тем.

        Аргументы:
            save_path (str, optional): Путь для сохранения графика
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.topic_range, self.coherence_values, marker="o")
        plt.xticks(self.topic_range)
        plt.xlabel("Количество тем")
        plt.ylabel("Coherence (c_v)")
        plt.title("Подбор числа тем для LDA")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()

    def print_topics(self, num_words=10):
        """
        Выводит ключевые слова по темам.

        Аргументы:
            num_words (int): Количество слов на тему
        """
        print("\nКлючевые слова по темам:")
        for tid, tdesc in self.lda_model.print_topics(num_topics=self.best_num_topics, num_words=num_words):
            print(f"Тема {tid}: {tdesc}")

    def plot_wordclouds(self, save_path=None):
        """
        Строит облака слов для каждой темы.

        Аргументы:
            save_path (str, optional): Путь для сохранения графика
        """
        cols = 2
        rows = (self.best_num_topics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
        axes = np.array(axes).reshape(-1)

        for i in range(self.best_num_topics):
            topic_words = dict(self.lda_model.show_topic(i, topn=30))
            wc = WordCloud(
                width=700,
                height=400,
                background_color="white",
                colormap="tab10"
            ).generate_from_frequencies(topic_words)

            axes[i].imshow(wc, interpolation="bilinear")
            axes[i].set_title(f"Тема {i}")
            axes[i].axis("off")

        for j in range(self.best_num_topics, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Облака слов сохранены: {save_path}")

        plt.show()

    def get_model_info(self):
        """
        Возвращает информацию о модели.

        Возвращает:
            dict: Информация о модели
        """
        return {
            'best_num_topics': self.best_num_topics,
            'best_coherence': self.best_coherence,
            'vocabulary_size': len(self.id2word),
            'num_documents': len(self.corpus),
        }

    def assign_topics_to_documents(self):
        """
        Назначает темы документам на основе обученной LDA модели.

        Возвращает:
            pd.DataFrame: DataFrame с колонками id, lda_tokens, lda_topic
        """
        print("\nНазначение тем документам...")
        
        # Получаем доминантную тему для каждого документа
        topics = []
        for doc_bow in self.corpus:
            topic_dist = self.lda_model.get_document_topics(doc_bow)
            if topic_dist:
                # Выбираем тему с максимальной вероятностью
                dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
            else:
                dominant_topic = -1  # Если не удалось определить тему
            topics.append(dominant_topic)
        
        # Добавляем темы в DataFrame
        self.df['lda_topic'] = topics
        
        # Преобразуем токены в строку для сохранения
        self.df['lda_tokens_str'] = self.df[self.tokens_column].apply(
            lambda x: str(x) if isinstance(x, list) else x
        )
        
        print(f"Назначено тем: {len(topics)}")
        print(f"Распределение по темам:")
        print(self.df['lda_topic'].value_counts().sort_index())
        
        return self.df[['id', 'lda_tokens_str', 'lda_topic']].copy()

    def save_results_to_database(self, conn):
        """
        Сохраняет результаты LDA в базу данных.

        Аргументы:
            conn: SQLite объект соединения

        Возвращает:
            int: Количество сохраненных записей
        """
        print("\nСохранение результатов LDA в базу данных...")
        
        # Назначаем темы документам
        results_df = self.assign_topics_to_documents()
        
        # Получаем ключевые слова для каждого топика
        topic_keywords_map = {}
        for topic_id in range(self.best_num_topics):
            # Получаем топ-7 слов для топика
            topic_words = self.lda_model.show_topic(topic_id, topn=7)
            keywords = ", ".join([word for word, _ in topic_words])
            topic_keywords_map[topic_id] = keywords
        
        # Подготавливаем данные для batch insert
        results_list = []
        for _, row in results_df.iterrows():
            topic_id = int(row['lda_topic'])
            # Преобразуем номер топика в ключевые слова
            topic_keywords = topic_keywords_map.get(topic_id, f"Топик {topic_id}")
            
            results_list.append({
                'article_id': int(row['id']),
                'lda_tokens': row['lda_tokens_str'],
                'lda_topic_keywords': topic_keywords
            })
        
        # Вставляем результаты в базу данных
        try:
            # Очистка старых результатов перед записью новых (UNIQUE на article_id)
            conn.cursor().execute("DELETE FROM lda_results")
            conn.commit()
            rowcount = insert_lda_results_batch(conn, results_list)
            print(f"Сохранено {rowcount} результатов LDA в базу данных")
            return rowcount
        except sqlite3.Error as e:
            print(f"Ошибка при сохранении результатов LDA: {e}")
            raise


def run_lda_modeling(df=None, conn=None, tokens_column='lda_tokens', start_topics=4, limit_topics=20, output_dir=None, db_path=None):
    """
    Упрощенная функция для запуска LDA моделирования.

    Аргументы:
        df (pd.DataFrame, optional): DataFrame с данными (если None, загружается из БД)
        conn (sqlite3.Connection, optional): Соединение с базой данных
        tokens_column (str): Название колонки с токенами
        start_topics (int): Начальное количество тем
        limit_topics (int): Максимальное количество тем
        output_dir (str, optional): Директория для сохранения результатов
        db_path (str, optional): Путь к базе данных (если conn не предоставлен)

    Возвращает:
        LDATopicModeler: Обученная модель
    """
    # Если DataFrame не предоставлен, загружаем из базы данных
    if df is None:
        if conn is None:
            if db_path is None:
                db_path = DEFAULT_DATABASE_PATH
            conn = get_connection(str(db_path))
            close_conn = True
        else:
            close_conn = False
        
        print("Загрузка статей из базы данных...")
        df = get_all_articles(conn)
        print(f"Загружено {len(df)} статей")
    else:
        close_conn = False
    
    # Инициализация
    modeler = LDATopicModeler(df, tokens_column=tokens_column)

    # Построение словаря и корпуса
    modeler.build_dictionary_and_corpus()

    # Поиск оптимального количества тем
    modeler.find_optimal_topics(start=start_topics, limit=limit_topics)

    # Вывод тем
    modeler.print_topics()

    # Графики
    if output_dir:
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        modeler.plot_coherence(save_path=output_dir / "lda_coherence.png")
        modeler.plot_wordclouds(save_path=output_dir / "lda_wordclouds.png")
    else:
        modeler.plot_coherence()
        modeler.plot_wordclouds()

    # Сохраняем результаты в базу данных, если соединение предоставлено
    if conn is not None:
        modeler.save_results_to_database(conn)
    
    # Закрываем соединение, если мы его открыли
    if close_conn:
        conn.close()

    return modeler


if __name__ == "__main__":
    # Пример использования с базой данных
    print("Запуск LDA моделирования с использованием базы данных...")
    
    # Вариант 1: Автоматическая загрузка из базы данных
    modeler = run_lda_modeling(
        conn=None,  # Автоматически создаст соединение
        db_path=DEFAULT_DATABASE_PATH,
        tokens_column='lda_tokens',
        start_topics=4,
        limit_topics=20,
        output_dir="project_data/lda_results"
    )

    print("\nИнформация о модели:")
    print(modeler.get_model_info())
    
    # Вариант 2: Использование существующего соединения
    # conn = get_connection(str(DEFAULT_DATABASE_PATH))
    # modeler = run_lda_modeling(
    #     conn=conn,
    #     tokens_column='lda_tokens',
    #     start_topics=4,
    #     limit_topics=20,
    #     output_dir="project_data/lda_results"
    # )
    # conn.close()
