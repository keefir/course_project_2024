from detection import start_loading_pipeline
from recognition import start_recognition_pipeline


def start_app():
    while True:
        mode = input("What would you like to do?\n\nLoad labeled images - L\nRecognize faces - R\nFinish - F\n").lower()
        while mode != "l" and mode != "r" and mode != 'f':
            mode = input("Wrong parameter.\n")

        if mode == "l":
            start_loading_pipeline('./data/to_label')
        elif mode == "r":
            start_recognition_pipeline('./data/to_recognize')
        else:
            break
