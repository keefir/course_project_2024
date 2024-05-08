import os
import sqlite3
import pickle
import torch


def add_image(connection, cursor, byte_image, class_name, image_size):
    query = f"INSERT INTO images (image, class_name, c, h, w) VALUES (?, ?, ?, ?, ?)"
    c, h, w = image_size[0], image_size[1], image_size[2]
    data_tuple = (byte_image, class_name, c, h, w)
    all_images = get_all_images(connection, cursor)
    for row in all_images:  # checking if image already exists
        if torch.all(pickle.loads(row[1]) == pickle.loads(byte_image)):
            return
    cursor.execute(query, data_tuple)

    connection.commit()


def add_class(connection, cursor, class_name):
    query = f"INSERT or IGNORE INTO classes (class_name) VALUES (?)"
    data_tuple = (class_name,)
    cursor.execute(query, data_tuple)

    connection.commit()


def get_all_images(connection, cursor):
    query = "SELECT * FROM images"
    cursor.execute(query)

    connection.commit()

    return cursor.fetchall()


def get_all_classes(connection, cursor):
    query = "SELECT * FROM classes"
    cursor.execute(query)
    connection.commit()

    return cursor.fetchall


def get_images_with_certain_class(connection, cursor, class_name):
    query = f"SELECT image FROM images WHERE class_name = ?"
    data_tuple = (class_name,)
    cursor.execute(query, data_tuple)

    connection.commit()

    return cursor.fetchall()


def get_images_with_certain_class_label(connection, cursor, class_label):
    query = f"SELECT image FROM images WHERE class = ?"
    # TODO


def get_class_label(connection, cursor, class_name):
    query = f"SELECT (id) FROM classes WHERE class_name = ?"
    data_tuple = (class_name,)
    cursor.execute(query, data_tuple)
    connection.commit()

    return cursor.fetchall()[0][0]


def get_class_name(connection, cursor, class_label):
    query = f"SELECT (class_name) FROM classes WHERE id = ?"
    # print(str(class_label)
    cursor.execute(query, str(class_label))
    connection.commit()
    # print(cursor.fetchall()[0])
    return cursor.fetchall()[0][0]


def get_all_labels(connection, cursor):
    cursor.execute("select distinct class_name from images")

    return cursor.fetchall()
