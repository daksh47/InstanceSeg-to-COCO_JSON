import copy
import cv2
import json, os
import numpy as np
from detectron2.detectron2.data import MetadataCatalog


def saveFile(data, label_path):
    """

    :param data: contains the COCO labels
    :param label_path: contains the path where the COCO labels will be saved
    :return: None

    this saves the COCO file in json format

    """
    with open(label_path, 'w') as f:
        json.dump(data, f, indent=4)


def detections_to_labels(instances, label_path, imgPth, cfg):
    """

    :param instances: the object that has the prediction data of an image
    :param label_path: it contains the path for saving the final COCO json file
    :param imgPth: contains the current image which is was used for prediction
    :param cfg: it contains the configuration data of the model
    :return: None

    this function creates ( if the data already exists then overwrite the variable ) a variable similar to a json data
    format,  the method appends the image data,and categories while taking care of not writing the same category twice,
    for multiple images, for the annotations, the prediction mask ( 0 & 1's ) is converted contour's to reduce the
    number of coords marking the segmentation mask, which improves the efficiency and significantly reduces the memory
    overhead, along with the reduction of annotation file size

    """
    with open(label_path, 'r') as f:
        data = {
            "images": [],
            "categories": [],
            "annotations": []
        }
        try:
            data = json.load(f)
            print("loaded : ", data)
        except Exception as e:
            print("Couldnt load", e)

        img = cv2.imread(imgPth)
        h, w, _ = img.shape
        pos = len(data["images"])
        data["images"].append({
            "height": h,
            "width": w,
            "id": pos,
            "file_name": os.path.basename(imgPth)
        })

        pre_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        pred_class = []
        for i in set(instances.pred_classes):
            pred_class.append(pre_classes[i])
        pred_class.sort()
        benchod = len(data["categories"])
        br = 0
        for i in pred_class:
            for j in data["categories"]:
                if i.strip() == j["name"].strip():
                    br = -1
                    break
            if br != -1:
                data["categories"].append({
                    "supercategory": i.strip(),
                    "id": benchod,
                    "name": i.strip()
                })
                benchod += 1
            br = 0

        annotation_id = len(data["annotations"]) + 1
        bboxes = (instances.pred_boxes.tensor.numpy()).astype(int)
        annotation_index = [i["name"] for i in data["categories"]]
        annotation_category = [pre_classes[i] for i in list(instances.pred_classes)]
        for i, j, k in zip(instances.pred_masks.numpy(), bboxes, annotation_category):
            b = (i * 255).astype(np.uint8)
            contours, hierarchy = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            epsilon = 0.01 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            reduced = []
            for ijk in approx:
                reduced.append(int(ijk[0][0]))
                reduced.append(int(ijk[0][1]))
            data["annotations"].append({
                "segmentation": [
                    reduced
                ],
                "iscrowd": 0,
                "area": 0,
                "image_id": pos,
                "bbox": j.tolist(),
                "category_id": annotation_index.index(k),
                "id": annotation_id
            })
            annotation_id += 1
        saveFile(data, label_path)
