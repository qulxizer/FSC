from camera.utils import Utils
from camera.stereo import Camera
import cv2 as cv

utils = Utils()

cam = Camera(
    capture=cv.VideoCapture(2),
    calibration_result=None, # type: ignore
    name="Left Camera",
    cv_index=0,
    framerate=30
)
utils.captureImageToDirectory( 
    camera=cam,
    directory="dataset/our_dataset/calibration/left/",
    key=ord('s')
)
