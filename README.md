The implementation for [this]( https://github.com/facebookresearch/detectron2/issues/5444 ) issue on detectron2

This python code uses detectron2 to perform instance segmentation on images, and these predictions are converted to labels/annotations in COCO JSON format

add your COCO JSON path and file name, as to where the files is located or has to be stored
add your image/images ( use loop to call the main function, multiple times for multiple images ) path

all the changes related to output and input files and data should be made in detectToLabelConverter.py file

# Sample Data

## Input image
![0](https://github.com/user-attachments/assets/0f977553-4e4d-4003-9384-ac52d1ead27f)

## Output file
[train.json](https://github.com/user-attachments/files/19029330/train.json)


## LABEL/ANNOTATION VISUALS
![image](https://github.com/user-attachments/assets/0c40f421-304c-49a6-a711-c847b9d3b8ec)


