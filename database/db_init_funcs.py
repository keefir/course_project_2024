import sqlite3
import os


def init_images_db(cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    image BLOB NOT NULL UNIQUE,
    class_name TEXT NOT NULL,
    c INTEGER NOT NULL,
    h INTEGER NOT NULL,
    w INTEGER NOT NULL
    )
    ''')


def init_classes_db(cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY,
    class_name TEXT NOT NULL UNIQUE
    )
    ''')


def count_rows(cursor):
    cursor.execute('''
    SELECT Count(*) FROM images''')

    return cursor.fetchall()


# def init_first_rows(cursor):
#     cursor.execute('''
#     INSERT INTO images VALUES ''')


def connect_db(path, db_name):
    if db_name != "images.db":
        raise NameError(f"database {db_name} does not exist")

    connection = sqlite3.connect(os.path.join(path, db_name))
    cursor = connection.cursor()

    init_images_db(cursor)
    init_classes_db(cursor)
    # num_rows = count_rows(cursor)

    # if db_name == "images.db":
    #     init_images_db(cursor)
    # elif db_name == "classes.db":
    #     init_classes_db(cursor)
    # else:
    #     raise NameError("wtf")

    return connection, cursor
