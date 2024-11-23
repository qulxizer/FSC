from ultralytics import YOLO
from roboflow import Roboflow


class Detector():
    """docstring for Detector"""

    def download_dataset(self, api_key:str,path_to_dataset:str) :
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("money-detection-xez0r").project("tomato-checker")
        version = project.version(1)
        dataset = version.download("yolov11", location=path_to_dataset)
        print(dataset.location)


    def train(self, path_dataset_yaml:str, path_training_runs):
        model = YOLO("yolo11n.pt") 
        model.train(data=path_dataset_yaml, epochs=50, imgsz=640,device="cpu",  project=path_training_runs)

    