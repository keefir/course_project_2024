from detection import loading_pipeline
from database.db_init_funcs import connect_db


def main():
    loading_pipeline('./data')
    # yolo_results = yolo_detect('./data')
    # save_images_with_boxes(yolo_results, './boxes_img.png')

main()
