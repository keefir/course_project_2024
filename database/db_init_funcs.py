import sqlite3
import os


def init_db(cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    image BLOB NOT NULL,
    class_name TEXT NOT NULL
    )
    ''')


def connect_db(path):
    connection = sqlite3.connect(os.path.join(path, 'labels.db'))
    cursor = connection.cursor()
    init_db(cursor)

    return connection, cursor
