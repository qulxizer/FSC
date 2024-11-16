# import threading
from dataclasses import dataclass
import cv2 as cv


@dataclass
class Camera:
    """Camera dataclass contains general information about
    the camera framerate cv_index. """
    # OpenCv index
    capture: cv.VideoCapture
    name: str
    cv_index: int
    framerate: int

    


class StereoCamera():
    """docstring for StereoCamera."""
    def __init__(self, left_camera:Camera, right_camera:Camera):
        self.left_camera = left_camera
        self.right_camera = right_camera



    # Capturing cameras and checking if they are accessable 
    def capture(self):
        print("Capturing stereo images...")
        right_cam = cv.VideoCapture(self.right_camera.cv_index)
        if not right_cam.isOpened():
            print("Error: Could not access right camera.")
            exit(1)
            
        left_cam = cv.VideoCapture(self.right_camera.cv_index)
        if not left_cam.isOpened():
            print("Error: Could not access left camera.")
            exit(1)
        
        

        