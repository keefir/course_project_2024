from helper_funcs import start_app
from database.db_init_funcs import connect_db
from database.db_helper_funcs import get_all_images, get_all_labels
import torch
import io
import numpy as np
import pickle
from recognition import start_recognition_pipeline
import os


def main():
    start_app()


main()
