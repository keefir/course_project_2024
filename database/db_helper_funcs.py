import os
import sqlite3


def add_image(connection, cursor, byte_image, class_name):

    query = f"INSERT INTO images (image, class_name) VALUES (?, ?)"
    data_tuple = (byte_image, class_name)
    cursor.execute(query, data_tuple)

    connection.commit()

