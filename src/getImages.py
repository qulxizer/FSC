from camera.utils import Utils
from camera.stereo import Camera
import cv2 as cv

utils = Utils()

Lcam = Camera(
    capture=cv.VideoCapture(0),
    calibration_result=None, # type: ignore
    name="Right Camera",
    cv_index=0,
    framerate=30
)

Rcam = Camera(
    capture=cv.VideoCapture(2),
    calibration_result=None, # type: ignore
    name="Left Camera",
    cv_index=2,
    framerate=30
)
utils.captureStereoImage( 
    Lcamera=Lcam,
    Rcamera=Rcam,
    Ldirectory="dataset/our_dataset/calibration2/Tleft/",
    Rdirectory="dataset/our_dataset/calibration2/Tright/",
    key=ord('s')
)
