from ultralytics import YOLO
import os
from PIL import Image

PATH = './datasets/everything/'
filenames = os.listdir(PATH)
for filename in filenames:
    image_path = PATH + filename
    model = YOLO("best.pt")
    result = model(image_path)
    image = Image.open(image_path)
    boxes = result[0].boxes.xyxy.to('cpu').numpy().astype(int)
    confidences = result[0].boxes.conf.to('cpu').numpy().astype(float)
    labels = result[0].boxes.cls.to('cpu').numpy().astype(int)
    for box, conf, label in zip(boxes, confidences, labels):
        if label == 0 and conf > 0.7:
            x_min, y_min, x_max, y_max = box
            image_crop = image.crop((x_min,y_min, x_max,y_max))
            image_crop = image_crop.convert('RGB')
            image_crop.save('./tests/cropped/' + filename)