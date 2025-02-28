import cv2
import pickle
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2 import model_zoo
from detectron2.detectron2.utils.visualizer import ColorMode, Visualizer

import files.detectionsCONVERTIONtoLABELS as dctl



def main(predictor, imagePath, outputPath):
    """

    :param predictor: the predictor object which will be used to make predictions
    :param imagePath: contains the current image path
    :param outputPath: contains the output json COCO label path
    :return: None

    make predictions to an image and then save it as COCO labels

    """
    im = cv2.imread(imagePath)
    outputs = predictor(im)
    instances = outputs["instances"]
    dctl.detections_to_labels(instances, outputPath, imagePath, cfg)
    print(f"saved {outputPath.split('/')[-1]}")


if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.25
    cfg.TEST.DETECTIONS_PER_IMAGE = 999999999
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    """
    The above is used to setup an model by loading weights and additional parameters which is ready to perform predictions
    """

    path = "train.json"  # output file path where you want to store the COCO labels
    open(path, 'w').close()  # remove the contents from the file if it exists
    main(predictor=predictor, imagePath="0.png", outputPath=path)  # to perform predictions to COCO labels on multiple images call this function in a loop with the same outputPath and different imagePath
