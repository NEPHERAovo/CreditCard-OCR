from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("D:/Softwares/Python/CreditCard-OCR/runs/detect/train3/weights/last.pt")  # load a pretrained model (recommended for training
    # Use the model
    model.train(data="CreditCard.yaml", epochs=100, workers=1,resume=True)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # success = model.export(format="onnx")  # export the model to ONNX format