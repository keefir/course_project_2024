from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from detection import yolo_detect
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, crop, to_tensor
from sklearn.neighbors import KNeighborsClassifier
import os
import torch
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pretrained_weights.resnet import Resnet34Triplet
from database.db_init_funcs import connect_db
from database.db_helper_funcs import get_all_images, get_class_name, get_class_label
from torchvision.utils import draw_bounding_boxes


class ResNet50EmbNorm(nn.Module):

    def __init__(self, emb_size=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, emb_size, bias=False)

    def forward(self, data):
        return F.normalize(self.model(data), p=2, dim=1)


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=140),  # Pre-trained model uses 140x140 input images
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.6071, 0.4609, 0.3944],  # Normalization settings for the model, the calculated mean and std values
        std=[0.2457, 0.2175, 0.2129]  # for the RGB channels of the tightly-cropped glint360k face dataset
    )
])


def fit_onn(neighs, model):
    connection, cursor = connect_db('./', 'images.db')
    data = get_all_images(connection, cursor)
    embs = []
    classes = []
    for row in data:
        # print(preprocess(pickle.loads(row[1])))
        embs.append(model(preprocess(pickle.loads(row[1]))[None, :, :, :]).detach().numpy()[0])  # Unblobbing
        classes.append(row[2])
    enc = LabelEncoder()
    enc.fit(classes)
    # print(enc.transform(classes))
    # print(embs[0])
    # print(embs[0][0])
    # print(enc.classes_)
    labels = []
    for class_name in classes:
        labels.append(get_class_label(connection, cursor, class_name))
    neighs.fit(embs, labels)

    cursor.close()


def save_image_with_boxes(imgs, boxes, labels, paths, save_path):
    """
    :param imgs:
    :param boxes:
    :param labels:
    :param save_path:
    :return: None
    """
    # print(yolo_results[0].boxes.xyxy.sort(dim=0))
    connection, cursor = connect_db('./', 'images.db')
    for i in range(len(imgs)):
        class_names = []
        for label in labels[i]:
            class_names.append(get_class_name(connection, cursor, label))
        # print(torch.tensor(imgs[i]).permute(2, 0, 1).shape)
        img = torch.tensor(imgs[i]).permute(2, 0, 1)
        boxed_img = draw_bounding_boxes(img,
                                        boxes[i], labels=class_names, width=int(img.shape[1] * 0.0053),
                                        font='./fonts/English.ttf',
                                        font_size=int(img.shape[1] * 0.04), colors="green", fill=False)
        # print(boxed_img[0])
        # to_pil_image(torch.stack((boxed_img[2], boxed_img[1], boxed_img[0])), mode='RGB').save(save_path)
        to_pil_image(torch.stack((boxed_img[2], boxed_img[1], boxed_img[0])),
                     mode='RGB').save(os.path.join(save_path, os.path.basename(paths[i])))
        # to_pil_image(yolo_results[i].orig_img).save('./orig_img.jpg')

    print(f"Images saved to {save_path}.")


def make_csv_with_labels(paths, labels):
    connection, cursor = connect_db('./', 'images.db')
    classes = []
    for i in range(len(paths)):
        class_names = []
        for label in labels[i]:
            class_names.append(get_class_name(connection, cursor, label))
        classes.append(class_names)
    df = pd.DataFrame({'image': paths, 'classes': classes}, columns=['image', 'classes'])

    df.to_csv('./data/result/csv_with_labels/result.csv')


def start_recognition_pipeline(path):
    checkpoint = torch.load('pretrained_weights/model_resnet34_triplet.pt', map_location="cpu")
    model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
    model.load_state_dict(checkpoint['model_state_dict'])
    # best_distance_threshold = checkpoint['best_distance_threshold']
    # print(os.listdir(path))
    yolo_results = yolo_detect(path, False)

    if yolo_results is None:
        return

    embs = []
    orig_imgs = []
    boxes = []
    paths = []
    model.eval()
    for i in range(len(yolo_results)):
        print(f"Found {len(yolo_results[i].boxes)} faces on image {yolo_results[i].path}.")
        orig_imgs.append(yolo_results[i].orig_img)
        # boxes.append(yolo_results[i].boxes.xyxy)
        sorted_xyxy = yolo_results[i].boxes.xyxy[yolo_results[i].boxes.xyxy[:, 0].argsort()]
        boxes.append(sorted_xyxy)
        paths.append(yolo_results[i].path)
        embs.append([])
        for j in range(len(sorted_xyxy)):
            box = sorted_xyxy[j]
            cropped_face = crop(to_tensor(to_pil_image(yolo_results[i].orig_img, mode="RGB")),
                                int(box[1].item()), int(box[0].item()), int((box[3] - box[1]).item()),
                                int((box[2] - box[0]).item()))
            cropped_face = preprocess(torch.stack((cropped_face[2], cropped_face[1], cropped_face[0]))).unsqueeze(0)
            embs[-1].append(model(cropped_face).detach().numpy()[0])

    neighs = KNeighborsClassifier(n_neighbors=1)
    fit_onn(neighs, model)

    preds = []
    for emb in embs:
        pred = neighs.predict(emb)
        preds.append(pred)

    save_image_with_boxes(orig_imgs, boxes, preds, paths, './data/result/images_with_boxes')
    make_csv_with_labels(paths, preds)

    return
