import sqlite3
from pathlib import Path
import pandas as pd


def init_database(db_path):
    """Инициализация схемы базы данных со всеми таблицами и индексами."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Создание таблицы articles
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT NOT NULL UNIQUE,
            annotation TEXT,
            keywords TEXT,
            main_text TEXT,
            udc TEXT,
            authors TEXT,
            lda_tokens TEXT,
            topic TEXT
        )
    """)
    
    # Создание таблицы lda_results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lda_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            lda_tokens TEXT,
            lda_topic_keywords TEXT,
            FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
        )
    """)
    
    # Создание таблицы clustering_results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clustering_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            cluster_name TEXT,
            FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
        )
    """)
    
    # Создание таблицы vak_results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vak_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            vak_embed_code TEXT,
            embed_cosine REAL,
            vak_tfidf_code TEXT,
            tfidf_cosine REAL,
            is_ambiguous INTEGER DEFAULT 0,
            has_discrepancy INTEGER DEFAULT 0,
            FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
        )
    """)
    
    # Создание indexes on foreign keys
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lda_results_article_id 
        ON lda_results(article_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_clustering_results_article_id 
        ON clustering_results(article_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_vak_results_article_id 
        ON vak_results(article_id)
    """)
    
    # Создание indexes on frequently queried columns
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_articles_title 
        ON articles(title)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_articles_source 
        ON articles(source)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_clustering_results_cluster_name 
        ON clustering_results(cluster_name)
    """)
    
    # Создание таблицы saved_classifications для сохранения результатов из интерфейса
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            title TEXT NOT NULL,
            annotation TEXT,
            keywords TEXT,
            main_text TEXT,
            vak_tfidf_code TEXT,
            vak_tfidf_title TEXT,
            vak_tfidf_score REAL,
            vak_embed_code TEXT,
            vak_embed_title TEXT,
            vak_embed_score REAL,
            top3_json TEXT,
            is_ambiguous INTEGER DEFAULT 0,
            has_discrepancy INTEGER DEFAULT 0,
            notes TEXT
        )
    """)
    
    # Создание индексов для saved_classifications
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_saved_classifications_timestamp 
        ON saved_classifications(timestamp)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_saved_classifications_title 
        ON saved_classifications(title)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_saved_classifications_vak_tfidf 
        ON saved_classifications(vak_tfidf_code)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_saved_classifications_vak_embed 
        ON saved_classifications(vak_embed_code)
    """)
    
    conn.commit()
    return conn


def get_connection(db_path):
    """Получение соединения с базой данных с включенными внешними ключами."""
    conn = sqlite3.Connection(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    return conn


# Операции CRUD для статей

def insert_article(conn, article_data):
    """
    Вставка одной статьи в базу данных.
    
    Аргументы:
        conn: SQLite объект соединения
        article_data: словарь с ключами, соответствующими articles столбцам таблицы
                     (source, title, annotation, keywords, main_text, udc, authors, lda_tokens, topic)
    
    Возвращает:
        int: lastrowid вставленной article
    
    Исключения:
        sqlite3.IntegrityError: если статья с таким заголовком уже существует
        sqlite3.Error: for other ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO articles (source, title, annotation, keywords, main_text, udc, authors, lda_tokens, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article_data.get('source'),
            article_data.get('title'),
            article_data.get('annotation'),
            article_data.get('keywords'),
            article_data.get('main_text'),
            article_data.get('udc'),
            article_data.get('authors'),
            article_data.get('lda_tokens'),
            article_data.get('topic')
        ))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise sqlite3.IntegrityError(f"Article with title '{article_data.get('title')}' already exists") from e
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error inserting article: {e}") from e


def insert_articles_batch(conn, articles_list):
    """
    Вставка нескольких статей пакетным запросом.
    
    Аргументы:
        conn: SQLite объект соединения
        articles_list: список словарей, каждый с ключами matching articles столбцам таблицы
    
    Возвращает:
        int: количество строк (количество вставленных строк)
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        inserted_count = 0
        skipped_count = 0
        
        for article in articles_list:
            try:
                # Преобразуем lda_tokens в строку если это список
                lda_tokens = article.get('lda_tokens')
                if isinstance(lda_tokens, list):
                    lda_tokens = str(lda_tokens)
                
                cursor.execute("""
                    INSERT INTO articles (source, title, annotation, keywords, main_text, udc, authors, lda_tokens, topic)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.get('source'),
                    article.get('title'),
                    article.get('annotation'),
                    article.get('keywords'),
                    article.get('main_text'),
                    article.get('udc'),
                    article.get('authors'),
                    lda_tokens,
                    article.get('topic')
                ))
                inserted_count += 1
            except sqlite3.IntegrityError:
                # Пропускаем дубликаты (статья с таким title уже существует)
                skipped_count += 1
                continue
        
        conn.commit()
        if skipped_count > 0:
            print(f"⚠️  Пропущено дубликатов: {skipped_count}")
        return inserted_count
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error batch inserting articles: {e}") from e


def update_article(conn, article_id, fields):
    """
    Обновление определенных полей статьи.
    
    Аргументы:
        conn: SQLite объект соединения
        article_id: int, ID статьи to update
        fields: dict with имена столбцов в качестве ключей and new values
    
    Возвращает:
        int: количество строк (1 if updated, 0 if article not found)
    
    Исключения:
        ValueError: если словарь полей пуст
        sqlite3.Error: for ошибки базы данных
    """
    if not fields:
        raise ValueError("словарь полей не может быть пустым")
    
    try:
        cursor = conn.cursor()
        set_clause = ", ".join([f"{key} = ?" for key in fields.keys()])
        values = list(fields.values()) + [article_id]
        
        cursor.execute(f"""
            UPDATE articles
            SET {set_clause}
            WHERE id = ?
        """, values)
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error updating article {article_id}: {e}") from e


def get_article_by_id(conn, article_id):
    """
    Получение статьи по ID.
    
    Аргументы:
        conn: SQLite объект соединения
        article_id: int, ID статьи
    
    Возвращает:
        dict: article data with имена столбцов в качестве ключей, или None если не найдено
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching article by ID {article_id}: {e}") from e


def get_article_by_title(conn, title):
    """
    Получение статьи по заголовку.
    
    Аргументы:
        conn: SQLite объект соединения
        title: str, заголовок статьи
    
    Возвращает:
        dict: article data with имена столбцов в качестве ключей, или None если не найдено
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM articles WHERE title = ?", (title,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching article by title '{title}': {e}") from e


def get_all_articles(conn):
    """
    Получение всех статей из базы данных.
    
    Аргументы:
        conn: SQLite объект соединения
    
    Возвращает:
        pandas.DataFrame: DataFrame с all articles
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM articles")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching all articles: {e}") from e


# Операции с результатами ВАК

def insert_vak_result(conn, article_id, vak_data):
    """
    Вставка одного результата ВАК в базу данных.
    
    Аргументы:
        conn: SQLite объект соединения
        article_id: int, ID статьи
        vak_data: dict with keys: vak_embed_code, embed_cosine, vak_tfidf_code, tfidf_cosine, is_ambiguous, has_discrepancy
    
    Возвращает:
        int: lastrowid вставленной VAK result
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO vak_results (article_id, vak_embed_code, embed_cosine, vak_tfidf_code, tfidf_cosine, is_ambiguous, has_discrepancy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            article_id,
            vak_data.get('vak_embed_code'),
            vak_data.get('embed_cosine'),
            vak_data.get('vak_tfidf_code'),
            vak_data.get('tfidf_cosine'),
            vak_data.get('is_ambiguous', 0),
            vak_data.get('has_discrepancy', 0)
        ))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error inserting VAK result for article {article_id}: {e}") from e


def insert_vak_results_batch(conn, results_list):
    """
    Вставка нескольких результатов ВАК пакетным запросом.
    
    Аргументы:
        conn: SQLite объект соединения
        results_list: список словарей, каждый с ключами: article_id, vak_embed_code, embed_cosine, vak_tfidf_code, tfidf_cosine, is_ambiguous, has_discrepancy
    
    Возвращает:
        int: количество строк (количество вставленных строк)
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        data = [
            (
                result.get('article_id'),
                result.get('vak_embed_code'),
                result.get('embed_cosine'),
                result.get('vak_tfidf_code'),
                result.get('tfidf_cosine'),
                result.get('is_ambiguous', 0),
                result.get('has_discrepancy', 0)
            )
            for result in results_list
        ]
        cursor.executemany("""
            INSERT INTO vak_results (article_id, vak_embed_code, embed_cosine, vak_tfidf_code, tfidf_cosine, is_ambiguous, has_discrepancy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data)
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error batch inserting VAK results: {e}") from e


# Операции с результатами кластеризации

def insert_clustering_result(conn, article_id, cluster_data):
    """
    Вставка одного результата кластеризации в базу данных.
    
    Аргументы:
        conn: SQLite объект соединения
        article_id: int, ID статьи
        cluster_data: dict with keys: cluster_name
    
    Возвращает:
        int: lastrowid вставленной clustering result
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO clustering_results (article_id, cluster_name)
            VALUES (?, ?)
        """, (
            article_id,
            cluster_data.get('cluster_name')
        ))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error inserting clustering result for article {article_id}: {e}") from e


def insert_clustering_results_batch(conn, results_list):
    """
    Вставка нескольких результатов кластеризации пакетным запросом.
    
    Аргументы:
        conn: SQLite объект соединения
        results_list: список словарей, каждый с ключами: article_id, cluster_name
    
    Возвращает:
        int: количество строк (количество вставленных строк)
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        data = [
            (
                result.get('article_id'),
                result.get('cluster_name')
            )
            for result in results_list
        ]
        cursor.executemany("""
            INSERT INTO clustering_results (article_id, cluster_name)
            VALUES (?, ?)
        """, data)
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error batch inserting clustering results: {e}") from e


def insert_cluster_topic_matrix(conn, matrix_data):
    """
    Вставка данных матрицы кластер-тема пакетным запросом.
    
    Аргументы:
        conn: SQLite объект соединения
        matrix_data: список словарей, каждый с ключами: cluster, topic, count
    
    Возвращает:
        int: количество строк (количество вставленных строк)
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        data = [
            (
                item.get('cluster'),
                item.get('topic'),
                item.get('count')
            )
            for item in matrix_data
        ]
        cursor.executemany("""
            INSERT INTO cluster_topic_matrix (cluster, topic, count)
            VALUES (?, ?, ?)
        """, data)
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error inserting cluster topic matrix: {e}") from e


# Операции с результатами LDA

def insert_lda_result(conn, article_id, lda_data):
    """
    Вставка одного результата LDA в базу данных.
    
    Аргументы:
        conn: SQLite объект соединения
        article_id: int, ID статьи
        lda_data: dict with keys: lda_tokens, lda_topic_keywords
    
    Возвращает:
        int: lastrowid вставленной LDA result
    
    Исключения:
        sqlite3.IntegrityError: если article_id не существует (ограничение внешнего ключа)
        sqlite3.Error: for other ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO lda_results (article_id, lda_tokens, lda_topic_keywords)
            VALUES (?, ?, ?)
        """, (
            article_id,
            lda_data.get('lda_tokens'),
            lda_data.get('lda_topic_keywords')
        ))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise sqlite3.IntegrityError(f"Foreign key constraint failed: article_id {article_id} does not exist") from e
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error inserting LDA result: {e}") from e


def insert_lda_results_batch(conn, results_list):
    """
    Вставка нескольких результатов LDA пакетным запросом.
    
    Аргументы:
        conn: SQLite объект соединения
        results_list: список словарей, каждый с ключами: article_id, lda_tokens, lda_topic_keywords
    
    Возвращает:
        int: количество строк (количество вставленных строк)
    
    Исключения:
        sqlite3.IntegrityError: if any article_id doesn't exist (ограничение внешнего ключа)
        sqlite3.Error: for other ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        data = [
            (
                result.get('article_id'),
                result.get('lda_tokens'),
                result.get('lda_topic_keywords')
            )
            for result in results_list
        ]
        cursor.executemany("""
            INSERT INTO lda_results (article_id, lda_tokens, lda_topic_keywords)
            VALUES (?, ?, ?)
        """, data)
        conn.commit()
        return cursor.rowcount
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise sqlite3.IntegrityError(f"Foreign key constraint failed: одно или несколько значений article_id не существуют") from e
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error batch inserting LDA results: {e}") from e


# Операции с JOIN запросами

def get_articles_with_lda(conn):
    """
    Получение всех статей с результатами LDA через LEFT JOIN.
    
    Аргументы:
        conn: SQLite объект соединения
    
    Возвращает:
        pandas.DataFrame: DataFrame с articles and LDA results
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                a.*,
                l.lda_topic_keywords
            FROM articles a
            LEFT JOIN lda_results l ON a.id = l.article_id
        """)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching articles with LDA: {e}") from e


def get_articles_with_clustering(conn):
    """
    Получение всех статей с результатами кластеризации через LEFT JOIN.
    
    Аргументы:
        conn: SQLite объект соединения
    
    Возвращает:
        pandas.DataFrame: DataFrame с articles and clustering results
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                a.*,
                c.cluster,
                c.tsne_x,
                c.tsne_y
            FROM articles a
            LEFT JOIN clustering_results c ON a.id = c.article_id
        """)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching articles with clustering: {e}") from e


def get_articles_with_vak(conn):
    """
    Получение всех статей с результатами классификации ВАК через LEFT JOIN.
    
    Аргументы:
        conn: SQLite объект соединения
    
    Возвращает:
        pandas.DataFrame: DataFrame с articles and VAK results
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                a.*,
                v.vak_ui_json,
                v.vak_tfidf_code,
                v.vak_embed_code
            FROM articles a
            LEFT JOIN vak_results v ON a.id = v.article_id
        """)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching articles with VAK: {e}") from e


# Операции с распределением кластер-ВАК

def calculate_cluster_vak_distribution(conn):
    """
    Вычисление распределения специальностей ВАК по кластерам через JOIN и GROUP BY.
    
    Аргументы:
        conn: SQLite объект соединения
    
    Возвращает:
        pandas.DataFrame: DataFrame с columns: cluster, vak_code, article_count
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        
        # Вычисление distribution through JOIN and GROUP BY
        cursor.execute("""
            SELECT 
                cr.cluster,
                vr.vak_tfidf_code AS vak_code,
                COUNT(*) AS article_count
            FROM clustering_results cr
            JOIN vak_results vr ON cr.article_id = vr.article_id
            WHERE vr.vak_tfidf_code IS NOT NULL AND vr.vak_tfidf_code != ''
            GROUP BY cr.cluster, vr.vak_tfidf_code
            ORDER BY cr.cluster, article_count DESC
        """)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        
        # Очистка existing distribution data
        cursor.execute("DELETE FROM cluster_vak_distribution")
        
        # Вставка new distribution data
        if not df.empty:
            data = [
                (row['cluster'], row['vak_code'], row['article_count'])
                for _, row in df.iterrows()
            ]
            cursor.executemany("""
                INSERT INTO cluster_vak_distribution (cluster, vak_code, article_count)
                VALUES (?, ?, ?)
            """, data)
        
        conn.commit()
        return df
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error calculating cluster-VAK distribution: {e}") from e


def insert_cluster_vak_distribution(conn, distribution_data):
    """
    Вставка данных распределения кластер-ВАК пакетным запросом.
    
    Аргументы:
        conn: SQLite объект соединения
        distribution_data: список словарей, каждый с ключами: cluster, vak_code, vak_title, article_count
    
    Возвращает:
        int: количество строк (количество вставленных строк)
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        data = [
            (
                item.get('cluster'),
                item.get('vak_code'),
                item.get('vak_title'),
                item.get('article_count')
            )
            for item in distribution_data
        ]
        cursor.executemany("""
            INSERT INTO cluster_vak_distribution (cluster, vak_code, vak_title, article_count)
            VALUES (?, ?, ?, ?)
        """, data)
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error inserting cluster-VAK distribution: {e}") from e


def get_cluster_vak_distribution(conn, cluster_id=None):
    """
    Получение статистики распределения кластер-ВАК.
    
    Аргументы:
        conn: SQLite объект соединения
        cluster_id: int, optional cluster ID to filter by
    
    Возвращает:
        pandas.DataFrame: DataFrame с columns: cluster, vak_code, vak_title, article_count
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        if cluster_id is not None:
            cursor.execute("""
                SELECT cluster, vak_code, vak_title, article_count
                FROM cluster_vak_distribution
                WHERE cluster = ?
                ORDER BY article_count DESC
            """, (cluster_id,))
        else:
            cursor.execute("""
                SELECT cluster, vak_code, vak_title, article_count
                FROM cluster_vak_distribution
                ORDER BY cluster, article_count DESC
            """)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching cluster-VAK distribution: {e}") from e


def get_vak_cluster_distribution(conn, vak_code=None):
    """
    Получение кластеров, содержащих определенную специальность ВАК.
    
    Аргументы:
        conn: SQLite объект соединения
        vak_code: str, optional VAK code to filter by
    
    Возвращает:
        pandas.DataFrame: DataFrame с distribution information
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        if vak_code is not None:
            cursor.execute("""
                SELECT cluster, vak_code, vak_title, article_count
                FROM cluster_vak_distribution
                WHERE vak_code = ?
                ORDER BY article_count DESC
            """, (vak_code,))
        else:
            cursor.execute("""
                SELECT 
                    vak_code,
                    vak_title,
                    COUNT(DISTINCT cluster) as cluster_count,
                    SUM(article_count) as total_articles
                FROM cluster_vak_distribution
                GROUP BY vak_code, vak_title
                ORDER BY total_articles DESC
            """)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching VAK-cluster distribution: {e}") from e


# Операции с сохранёнными классификациями

def insert_saved_classification(conn, classification_data):
    """
    Вставка одной сохранённой классификации в базу данных.
    
    Аргументы:
        conn: SQLite объект соединения
        classification_data: dict with keys:
            - title, annotation, keywords, main_text (текстовые поля)
            - vak_tfidf_code, vak_tfidf_title, vak_tfidf_score (TF-IDF результат)
            - vak_embed_code, vak_embed_title, vak_embed_score (Embedding результат)
            - top3_json (JSON с топ-3 специальностями)
            - is_ambiguous, has_discrepancy (флаги)
            - notes (опциональные заметки пользователя)
    
    Возвращает:
        int: lastrowid вставленной saved classification
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO saved_classifications (
                title, annotation, keywords, main_text,
                vak_tfidf_code, vak_tfidf_title, vak_tfidf_score,
                vak_embed_code, vak_embed_title, vak_embed_score,
                top3_json, is_ambiguous, has_discrepancy, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            classification_data.get('title'),
            classification_data.get('annotation'),
            classification_data.get('keywords'),
            classification_data.get('main_text'),
            classification_data.get('vak_tfidf_code'),
            classification_data.get('vak_tfidf_title'),
            classification_data.get('vak_tfidf_score'),
            classification_data.get('vak_embed_code'),
            classification_data.get('vak_embed_title'),
            classification_data.get('vak_embed_score'),
            classification_data.get('top3_json'),
            classification_data.get('is_ambiguous', 0),
            classification_data.get('has_discrepancy', 0),
            classification_data.get('notes')
        ))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error inserting saved classification: {e}") from e


def get_saved_classifications(conn, limit=100, vak_code=None):
    """
    Получение сохранённых классификаций с фильтрацией.
    
    Аргументы:
        conn: SQLite объект соединения
        limit: int, максимальное количество записей (по умолчанию 100)
        vak_code: str, optional код ВАК для фильтрации (ищет в обоих методах)
    
    Возвращает:
        pandas.DataFrame: DataFrame с saved classifications
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        
        query = "SELECT * FROM saved_classifications WHERE 1=1"
        params = []
        
        if vak_code is not None:
            query += " AND (vak_tfidf_code = ? OR vak_embed_code = ?)"
            params.extend([vak_code, vak_code])
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching saved classifications: {e}") from e


def get_saved_classification_by_id(conn, classification_id):
    """
    Получение сохранённой классификации по ID.
    
    Аргументы:
        conn: SQLite объект соединения
        classification_id: int, ID классификации
    
    Возвращает:
        dict: classification data with имена столбцов в качестве ключей, или None если не найдено
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM saved_classifications WHERE id = ?", (classification_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching saved classification by ID {classification_id}: {e}") from e


def update_saved_classification_notes(conn, classification_id, notes):
    """
    Обновление заметок для сохранённой классификации.
    
    Аргументы:
        conn: SQLite объект соединения
        classification_id: int, ID классификации
        notes: str, новые заметки
    
    Возвращает:
        int: количество строк (1 if updated, 0 if not found)
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE saved_classifications
            SET notes = ?
            WHERE id = ?
        """, (notes, classification_id))
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error updating notes for classification {classification_id}: {e}") from e


def delete_saved_classification(conn, classification_id):
    """
    Удаление сохранённой классификации.
    
    Аргументы:
        conn: SQLite объект соединения
        classification_id: int, ID классификации
    
    Возвращает:
        int: количество удалённых строк
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM saved_classifications WHERE id = ?", (classification_id,))
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"Error deleting saved classification {classification_id}: {e}") from e


def get_saved_classifications_statistics(conn):
    """
    Получение статистики сохранённых классификаций.
    
    Аргументы:
        conn: SQLite объект соединения
    
    Возвращает:
        dict: статистика с ключами:
            - total_count: общее количество
            - ambiguous_count: количество неоднозначных
            - discrepancy_count: количество с расхождением методов
            - top_vak_tfidf: топ-5 специальностей по TF-IDF
            - top_vak_embed: топ-5 специальностей по Embedding
    
    Исключения:
        sqlite3.Error: for ошибки базы данных
    """
    try:
        cursor = conn.cursor()
        
        # Общая статистика
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(is_ambiguous) as ambiguous,
                SUM(has_discrepancy) as discrepancy
            FROM saved_classifications
        """)
        total, ambiguous, discrepancy = cursor.fetchone()
        
        # Топ-5 специальностей TF-IDF
        cursor.execute("""
            SELECT vak_tfidf_code, vak_tfidf_title, COUNT(*) as count
            FROM saved_classifications
            WHERE vak_tfidf_code IS NOT NULL
            GROUP BY vak_tfidf_code, vak_tfidf_title
            ORDER BY count DESC
            LIMIT 5
        """)
        top_tfidf = cursor.fetchall()
        
        # Топ-5 специальностей Embedding
        cursor.execute("""
            SELECT vak_embed_code, vak_embed_title, COUNT(*) as count
            FROM saved_classifications
            WHERE vak_embed_code IS NOT NULL
            GROUP BY vak_embed_code, vak_embed_title
            ORDER BY count DESC
            LIMIT 5
        """)
        top_embed = cursor.fetchall()
        
        return {
            'total_count': total or 0,
            'ambiguous_count': ambiguous or 0,
            'discrepancy_count': discrepancy or 0,
            'top_vak_tfidf': top_tfidf,
            'top_vak_embed': top_embed
        }
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error fetching saved classifications statistics: {e}") from e
