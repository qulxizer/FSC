from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("models/best.pt")

# Export the model to ONNX format
model.export(format="tflite")  # creates 'best.onnx'

