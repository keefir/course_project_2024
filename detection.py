import os
import torch
import io
import pickle

from ultralytics import YOLO
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, crop
from database.db_init_funcs import connect_db
from database.db_helper_funcs import add_image, add_class
from torchvision.transforms import Resize


def yolo_detect(path, verbose=False):
    """
    :param path: Path to folder with images
    :param verbose: Whether to print logs
    :return: YOLO detection results object (UNSORTED)
    """
    print("Starting face detection...")

    imgs = [os.path.join(path, img_path) for img_path in os.listdir(path)]

    if len(imgs) <= 0:
        print("The directory with images is empty.")
        return None

    print(f"The directory contains {len(imgs)} image(s).")

    yolo = YOLO("./pretrained_weights/yolov8n-face.pt")
    result = yolo(imgs, verbose=verbose)

    print("Face detection complete.")
    return result


def start_loading_pipeline(path):
    label_input_mode = str(
        input("Choose how you would like to enter labels:\n\nManually - M\n")).lower()

    while label_input_mode != "m" and label_input_mode != "f":
        label_input_mode = str(input("Wrong parameter\n")).lower()

    connection, cursor = connect_db('./', 'images.db')
    # connection2, cursor2 = connect_db('./', 'classes.db')
    rs = Resize((128, 128))
    if label_input_mode == "m":
        yolo_results = yolo_detect(path, False)

        if yolo_results is None:  # Empty directory
            return

        for i in range(len(yolo_results)):
            print(f"Found {len(yolo_results[i].boxes)} face(s) on image {yolo_results[i].path}.")

            # Sorting boxes by xmin. We are doing this to conveniently assign classes in the future.
            sorted_xyxy = yolo_results[i].boxes.xyxy[yolo_results[i].boxes.xyxy[:, 0].argsort()]

            for j in range(len(sorted_xyxy)):
                class_name = input(f"Enter the class name for face {j + 1} (left to right):\n")

                while class_name == "":
                    class_name = input("Class name must contain at least one character.\n")

                box = sorted_xyxy[j]
                cropped_face = pil_to_tensor(crop(to_pil_image(yolo_results[i].orig_img, mode="RGB"),
                                                  int(box[1].item()), int(box[0].item()), int((box[3] - box[1]).item()),
                                                  int((box[2] - box[0]).item())))

                # For some reason, YOLO changes the order of color channels. The next line fixes this.
                cropped_face = torch.stack((cropped_face[2], cropped_face[1], cropped_face[0]))
                image_size = cropped_face.shape

                # LEGACY:

                # print(cropped_face.flatten().shape)
                # buff = io.BytesIO()
                # torch.save(cropped_face.flatten(), buff)
                # buff.seek(0)
                # blob_data = buff.read()
                # img = torch.frombuffer(blob_data, dtype=torch.uint8)
                # print(img.shape)

                blob_data = pickle.dumps(rs(cropped_face), protocol=0)
                add_image(connection, cursor, blob_data, class_name, image_size)
                add_class(connection, cursor, class_name)

        print(f"Loaded {len(yolo_results)} image(s) into database.")

    # elif label_input_mode == "f":
    #     return  # TODO
    #     label_file_name = str(input("Enter label file name (should be in app directory, csv format)\n"))
    #     while not os.path.isfile(os.path.join('../', label_file_name)):
    #         label_file_name = str(input(f"File {label_file_name} does not exist\n"))
    #     # TODO

    cursor.close()
