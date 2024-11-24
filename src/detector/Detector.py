import time
import torch
import cv2 as cv
from ultralytics import YOLO

class Detector():
    """
    A simple class for object detection using a YOLO model.
    """


    def __init__(self, model_path:str):
        """
        Initializes the detector with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.yolo = YOLO(model_path)


    def detect(self, img:cv.typing.MatLike):
        """
        Runs object detection on an image.

        Args:
            img (cv.typing.MatLike): The input image.

        Prints:
            The detection results.
        """
        results = self.yolo.predict(img, show=True, conf=0.25)
        time.sleep(100)
        
        print(results)

    def benchmark(self):
        """
        Benchmarks the model to measure its speed and performance.
        """
        self.yolo.benchmark()
