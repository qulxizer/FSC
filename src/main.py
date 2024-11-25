import time
from model import Camera, StereoCalibrationParams
from camera.stereo import StereoCamera
from camera.utils import Utils
from camera.video_stream import VideoStream
from detector.Detector import Detector
import numpy as np
import cv2 as cv

utils = Utils()


left_cam = Camera(
    cv.VideoCapture(0),
    utils.loadCalibrationResultFrom("dataset/opencv_sample/left/calibration.npz"),
    "Left Camera",
    cv_index=0,
    framerate=30
)


right_cam = Camera(
    cv.VideoCapture(2),
    utils.loadCalibrationResultFrom("dataset/opencv_sample/right/calibration.npz"),
    "Right Camera",
    cv_index=2,
    framerate=30
)


stereo_camera = StereoCamera(
    left_camera=left_cam, 
    right_camera=right_cam,
    params=StereoCalibrationParams(
        focal_length=None,
        baseline=87,
        block_size=4,
        num_disparities=16 * 2,
        min_disparity=8 ,
        disp12MaxDiff=10,
        uniquenessRatio=15,
        speckle_window_size=50,
        speckleRange=1,
    ),
)

Limg = cv.imread("/home/qulx/Dev/FSC/dataset/opencv_sample/left/left.png")
Rimg = cv.imread("/home/qulx/Dev/FSC/dataset/opencv_sample/right/right.png")

# res, Limg = left_cam.capture.read()
# res, Rimg = right_cam.capture.read()
cv.imshow("Left Image", Limg)
time.sleep(100)

# cv.imshow("Right Image", Rimg)
disparity = stereo_camera.Test(Limg, Rimg)
cv.imshow("Disparity", cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)) # type: ignore
time.sleep(100)
# detector = Detector("/home/qulx/Dev/FSC/models/best.pt")
# img = cv.imread("/home/qulx/Dev/FSC/tmp/20241124_205817.jpg")
# img = cv.resize(img, (500,1000))
# detector.detect(img)
# detector.benchmark()