import os
import torch
import io

from ultralytics import YOLO
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, crop
from database.db_init_funcs import connect_db
from database.db_helper_funcs import add_image


def yolo_detect(path, verbose=False):
    """
    :param path: Path to folder with images
    :param verbose: Whether to print logs
    :return: YOLO detection results object (UNSORTED)
    """
    print("Starting face detection...")

    yolo = YOLO("./pretrained_weights/yolov8n-face.pt")
    imgs = [os.path.join(path, img_path) for img_path in os.listdir(path)]

    print(f"The directory contains {len(imgs)} image(s).")

    result = yolo(imgs, verbose=verbose)

    print("Face detection complete.")
    return result


def save_images_with_boxes(yolo_results, save_path):
    """
    :param yolo_results: YOLO detection results object
    :param save_path: Path to save images
    :return: None
    """
    # print(yolo_results[0].boxes.xyxy.sort(dim=0))
    for i in range(len(yolo_results)):
        boxed_img = draw_bounding_boxes(torch.tensor(yolo_results[i].orig_img).permute(2, 0, 1),
                                        yolo_results[0].boxes.xyxy,
                                        width=5, colors="green", fill=False)
        # print(boxed_img[0])
        to_pil_image(torch.stack((boxed_img[2], boxed_img[1], boxed_img[0])), mode='RGB').save(save_path)
        # to_pil_image(yolo_results[i].orig_img).save('./orig_img.jpg')

    print(f"Images saved to {save_path}.")


def loading_pipeline(path):
    label_input_mode = str(
        input("Choose how you would like to enter labels: Manually - M; From the file - F\n")).lower()

    while label_input_mode != "m" and label_input_mode != "f":
        label_input_mode = str(input("Wrong parameter\n")).lower()

    connection, cursor = connect_db('./')

    if label_input_mode == "m":
        yolo_results = yolo_detect(path, False)
        for i in range(len(yolo_results)):
            print(f"Found {len(yolo_results[i].boxes)} faces on image {yolo_results[i].path}.")

            for j in range(len(yolo_results[i].boxes.xyxy)):
                class_name = input(f"Enter the class name for face {j + 1} (left to right):\n")

                while class_name == "":
                    class_name = input("Class name must contain at least one character.\n")

                box = yolo_results[i].boxes.xyxy[j]
                cropped_face = pil_to_tensor(crop(to_pil_image(yolo_results[i].orig_img, mode="RGB"),
                                                  int(box[1].item()), int(box[0].item()), int((box[3] - box[1]).item()),
                                                  int((box[2] - box[0]).item())))

                # For some reason, YOLO changes the order of color channels. The next line fixes this.
                cropped_face = torch.stack((cropped_face[2], cropped_face[1], cropped_face[0]))

                buff = io.BytesIO()
                torch.save(cropped_face, buff)
                buff.seek(0)
                blob_data = buff.read()
                add_image(connection, cursor, blob_data, class_name)

    elif label_input_mode == "f":
        label_file_name = str(input("Enter label file name (should be in app directory, csv format)\n"))
        while not os.path.isfile(os.path.join('../', label_file_name)):
            label_file_name = str(input(f"File {label_file_name} does not exist\n"))
        # TODO

    cursor.close()
