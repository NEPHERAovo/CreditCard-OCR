from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo_best.pt")  # load a pretrained model (recommended for training)
    # Use the model
    # model.train(data="CreditCard.yaml", epochs=100, workers=1, resume=True)  # train the model
    model.train(data="CreditCard.yaml", epochs=150, workers=1)
    metrics = model.val()  # evaluate model perfance on the validation set
    # success = model.export(format="onnx")  # export the model to ONNX format