from camera.utils import Utils
from camera.stereo import Camera
import cv2 as cv

utils = Utils()

cam = Camera(
    capture=cv.VideoCapture(0),
    calibration_result=None, # type: ignore
    name="Right Camera",
    cv_index=0,
    framerate=30
)
utils.captureImageToDirectory( 
    camera=cam,
    directory="dataset/our_dataset/calibration2/right/",
    key=ord('s')
)
