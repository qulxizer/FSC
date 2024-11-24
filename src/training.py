from detector.Detector import Detector
from ultralytics import YOLO


path_dataset_yaml="dataset/tomato_checker/data.yaml",
path_training_runs="dataset/tomato_checker/"


model = YOLO("yolo11n.pt") 
model.train(data=path_dataset_yaml, epochs=50, imgsz=640,device="cpu",  project=path_training_runs)