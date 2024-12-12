from time import sleep
from ultralytics import YOLO

model = YOLO("models/best.onnx")
model.predict("tmp/tomato.jpg", show=True, conf=0.25)
sleep(100)