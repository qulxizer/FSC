from camera.utils import Utils
from camera.stereo import Camera
import cv2 as cv

utils = Utils()

Right = Camera(
    capture=cv.VideoCapture(0),
    name="Right Camera",
    cv_index=0,
    framerate=30
)
utils.captureImageToDirectory(
    camera=Right,
    directory="dataset/our_dataset/calibration/right_camera/",
    key=ord('s')
)