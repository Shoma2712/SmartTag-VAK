# -*- coding: utf-8 -*-
"""
Модуль для создания таблиц распределения кластеров и VAK специальностей

Создает две таблицы:
1. cluster_vak_tfidf_distribution - для TF-IDF метода
2. cluster_vak_embed_distribution - для Embedding метода
"""

import sqlite3
from typing import Tuple


def create_cluster_vak_distributions(conn: sqlite3.Connection) -> bool:
    """
    Создать и заполнить таблицы распределения кластеров и VAK специальностей
    
    Args:
        conn: Соединение с базой данных SQLite
        
    Returns:
        True если успешно, False при ошибке
    """
    print("\n" + "="*80)
    print("СОЗДАНИЕ ТАБЛИЦ CLUSTER-VAK DISTRIBUTION")
    print("="*80)
    
    cursor = conn.cursor()
    
    try:
        # Создание таблицы для TF-IDF метода
        print("\n1. Создание таблицы cluster_vak_tfidf_distribution...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_vak_tfidf_distribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_name TEXT NOT NULL,
                vak_code TEXT NOT NULL,
                article_count INTEGER NOT NULL,
                percentage REAL,
                UNIQUE(cluster_name, vak_code)
            )
        """)
        
        # Создание индексов
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cluster_vak_tfidf_cluster 
            ON cluster_vak_tfidf_distribution(cluster_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cluster_vak_tfidf_vak 
            ON cluster_vak_tfidf_distribution(vak_code)
        """)
        
        print("✅ Таблица cluster_vak_tfidf_distribution создана")
        
        # Создание таблицы для Embedding метода
        print("\n2. Создание таблицы cluster_vak_embed_distribution...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_vak_embed_distribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_name TEXT NOT NULL,
                vak_code TEXT NOT NULL,
                article_count INTEGER NOT NULL,
                percentage REAL,
                UNIQUE(cluster_name, vak_code)
            )
        """)
        
        # Создание индексов
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cluster_vak_embed_cluster 
            ON cluster_vak_embed_distribution(cluster_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cluster_vak_embed_vak 
            ON cluster_vak_embed_distribution(vak_code)
        """)
        
        print("✅ Таблица cluster_vak_embed_distribution создана")
        
        conn.commit()
        
        # Заполнение таблиц данными
        _populate_distributions(conn)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False


def _populate_distributions(conn: sqlite3.Connection) -> None:
    """
    Вычислить и заполнить распределения кластеров и VAK специальностей
    
    Args:
        conn: Соединение с базой данных SQLite
    """
    print("\n" + "="*80)
    print("ВЫЧИСЛЕНИЕ И ЗАПОЛНЕНИЕ РАСПРЕДЕЛЕНИЙ")
    print("="*80)
    
    cursor = conn.cursor()
    
    # Очистка таблиц перед заполнением
    cursor.execute("DELETE FROM cluster_vak_tfidf_distribution")
    cursor.execute("DELETE FROM cluster_vak_embed_distribution")
    
    # Получение данных для TF-IDF метода
    print("\n1. Вычисление распределения для TF-IDF метода...")
    cursor.execute("""
        SELECT 
            c.cluster_name,
            v.vak_tfidf_code,
            COUNT(*) as article_count
        FROM clustering_results c
        JOIN vak_results v ON c.article_id = v.article_id
        WHERE v.vak_tfidf_code IS NOT NULL
        GROUP BY c.cluster_name, v.vak_tfidf_code
        ORDER BY c.cluster_name, article_count DESC
    """)
    
    tfidf_results = cursor.fetchall()
    
    # Вычисление процентов для каждого кластера
    cluster_totals_tfidf = {}
    for cluster_name, vak_code, count in tfidf_results:
        if cluster_name not in cluster_totals_tfidf:
            cluster_totals_tfidf[cluster_name] = 0
        cluster_totals_tfidf[cluster_name] += count
    
    # Вставка данных с процентами
    tfidf_data = []
    for cluster_name, vak_code, count in tfidf_results:
        total = cluster_totals_tfidf[cluster_name]
        percentage = (count / total * 100) if total > 0 else 0
        tfidf_data.append((cluster_name, vak_code, count, percentage))
    
    cursor.executemany("""
        INSERT INTO cluster_vak_tfidf_distribution 
        (cluster_name, vak_code, article_count, percentage)
        VALUES (?, ?, ?, ?)
    """, tfidf_data)
    
    print(f"✅ Вставлено {len(tfidf_data)} записей для TF-IDF метода")
    
    # Получение данных для Embedding метода
    print("\n2. Вычисление распределения для Embedding метода...")
    cursor.execute("""
        SELECT 
            c.cluster_name,
            v.vak_embed_code,
            COUNT(*) as article_count
        FROM clustering_results c
        JOIN vak_results v ON c.article_id = v.article_id
        WHERE v.vak_embed_code IS NOT NULL
        GROUP BY c.cluster_name, v.vak_embed_code
        ORDER BY c.cluster_name, article_count DESC
    """)
    
    embed_results = cursor.fetchall()
    
    # Вычисление процентов для каждого кластера
    cluster_totals_embed = {}
    for cluster_name, vak_code, count in embed_results:
        if cluster_name not in cluster_totals_embed:
            cluster_totals_embed[cluster_name] = 0
        cluster_totals_embed[cluster_name] += count
    
    # Вставка данных с процентами
    embed_data = []
    for cluster_name, vak_code, count in embed_results:
        total = cluster_totals_embed[cluster_name]
        percentage = (count / total * 100) if total > 0 else 0
        embed_data.append((cluster_name, vak_code, count, percentage))
    
    cursor.executemany("""
        INSERT INTO cluster_vak_embed_distribution 
        (cluster_name, vak_code, article_count, percentage)
        VALUES (?, ?, ?, ?)
    """, embed_data)
    
    print(f"✅ Вставлено {len(embed_data)} записей для Embedding метода")
    
    conn.commit()
    
    # Показ статистики
    _print_statistics(conn)


def _print_statistics(conn: sqlite3.Connection) -> None:
    """
    Вывести статистику распределений
    
    Args:
        conn: Соединение с базой данных SQLite
    """
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("СТАТИСТИКА РАСПРЕДЕЛЕНИЙ")
    print("="*80)
    
    print("\n📊 TF-IDF метод:")
    cursor.execute("""
        SELECT cluster_name, COUNT(DISTINCT vak_code) as vak_count, SUM(article_count) as total_articles
        FROM cluster_vak_tfidf_distribution
        GROUP BY cluster_name
        ORDER BY cluster_name
    """)
    
    for cluster_name, vak_count, total_articles in cursor.fetchall():
        print(f"  {cluster_name}")
        print(f"    Уникальных VAK кодов: {vak_count}")
        print(f"    Всего статей: {total_articles}")
    
    print("\n📊 Embedding метод:")
    cursor.execute("""
        SELECT cluster_name, COUNT(DISTINCT vak_code) as vak_count, SUM(article_count) as total_articles
        FROM cluster_vak_embed_distribution
        GROUP BY cluster_name
        ORDER BY cluster_name
    """)
    
    for cluster_name, vak_count, total_articles in cursor.fetchall():
        print(f"  {cluster_name}")
        print(f"    Уникальных VAK кодов: {vak_count}")
        print(f"    Всего статей: {total_articles}")
    
    # Показ примеров топ-3 VAK для каждого кластера
    print("\n" + "="*80)
    print("ТОП-3 VAK СПЕЦИАЛЬНОСТИ ПО КЛАСТЕРАМ")
    print("="*80)
    
    cursor.execute("SELECT DISTINCT cluster_name FROM cluster_vak_tfidf_distribution ORDER BY cluster_name")
    clusters = [row[0] for row in cursor.fetchall()]
    
    for cluster_name in clusters:
        print(f"\n🔹 {cluster_name}")
        
        # TF-IDF топ-3
        print("  TF-IDF метод:")
        cursor.execute("""
            SELECT vak_code, article_count, percentage
            FROM cluster_vak_tfidf_distribution
            WHERE cluster_name = ?
            ORDER BY article_count DESC
            LIMIT 3
        """, (cluster_name,))
        
        for vak_code, count, pct in cursor.fetchall():
            print(f"    {vak_code}: {count} статей ({pct:.1f}%)")
        
        # Embedding топ-3
        print("  Embedding метод:")
        cursor.execute("""
            SELECT vak_code, article_count, percentage
            FROM cluster_vak_embed_distribution
            WHERE cluster_name = ?
            ORDER BY article_count DESC
            LIMIT 3
        """, (cluster_name,))
        
        for vak_code, count, pct in cursor.fetchall():
            print(f"    {vak_code}: {count} статей ({pct:.1f}%)")
    
    print("\n💡 Примеры запросов:")
    print("\n1. Все VAK специальности в кластере (TF-IDF):")
    print("   SELECT * FROM cluster_vak_tfidf_distribution WHERE cluster_name = 'Математическое моделирование' ORDER BY article_count DESC;")
    
    print("\n2. Все кластеры для VAK специальности (Embedding):")
    print("   SELECT * FROM cluster_vak_embed_distribution WHERE vak_code LIKE '%1.2.2%' ORDER BY article_count DESC;")
    
    print("\n3. Сравнение методов для кластера:")
    print("   SELECT 'TF-IDF' as method, vak_code, article_count FROM cluster_vak_tfidf_distribution WHERE cluster_name = 'Программные системы и комплексы'")
    print("   UNION ALL")
    print("   SELECT 'Embedding' as method, vak_code, article_count FROM cluster_vak_embed_distribution WHERE cluster_name = 'Программные системы и комплексы'")
    print("   ORDER BY method, article_count DESC;")
