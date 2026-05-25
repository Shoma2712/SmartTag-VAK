# БЛОК 4: Тематическое моделирование LDA
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.phrases import Phrases, Phraser

# Отключаем логирование gensim
logging.getLogger("gensim").setLevel(logging.ERROR)


def ensure_tokens(x):
    """
    Преобразует строковое представление списка в список.

    Args:
        x: Значение (может быть list или str)

    Returns:
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

    Args:
        docs (list): Список документов (каждый документ - список токенов)
        min_count_bigram (int): Минимальная частота для биграмм
        threshold_bigram (int): Порог для биграмм
        min_count_trigram (int): Минимальная частота для триграмм
        threshold_trigram (int): Порог для триграмм

    Returns:
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

    Args:
        dictionary: Словарь gensim
        corpus: Корпус документов
        texts: Тексты для вычисления coherence
        start (int): Начальное количество тем
        limit (int): Максимальное количество тем
        step (int): Шаг

    Returns:
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

        Args:
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

        Args:
            no_below (int): Минимальная частота слова в документах
            no_above (float): Максимальная доля документов со словом
            keep_n (int): Максимальный размер словаря

        Returns:
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

        Args:
            start (int): Начальное количество тем
            limit (int): Максимальное количество тем
            step (int): Шаг

        Returns:
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

        Args:
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

        Args:
            num_words (int): Количество слов на тему
        """
        print("\nКлючевые слова по темам:")
        for tid, tdesc in self.lda_model.print_topics(num_topics=self.best_num_topics, num_words=num_words):
            print(f"Тема {tid}: {tdesc}")

    def plot_wordclouds(self, save_path=None):
        """
        Строит облака слов для каждой темы.

        Args:
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

        Returns:
            dict: Информация о модели
        """
        return {
            'best_num_topics': self.best_num_topics,
            'best_coherence': self.best_coherence,
            'vocabulary_size': len(self.id2word),
            'num_documents': len(self.corpus),
        }


def run_lda_modeling(df, tokens_column='lda_tokens', start_topics=4, limit_topics=20, output_dir=None):
    """
    Упрощенная функция для запуска LDA моделирования.

    Args:
        df (pd.DataFrame): DataFrame с данными
        tokens_column (str): Название колонки с токенами
        start_topics (int): Начальное количество тем
        limit_topics (int): Максимальное количество тем
        output_dir (str, optional): Директория для сохранения результатов

    Returns:
        LDATopicModeler: Обученная модель
    """
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

    return modeler


if __name__ == "__main__":
    # Пример использования
    df = pd.read_csv("project_data/dataset_IMT.csv")

    modeler = run_lda_modeling(
        df,
        tokens_column='lda_tokens',
        start_topics=4,
        limit_topics=20,
        output_dir="project_data/lda_results"
    )

    print("\nИнформация о модели:")
    print(modeler.get_model_info())
