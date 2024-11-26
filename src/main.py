import time
from model import Camera, DepthEstimationParams
from camera.stereo import StereoCamera
from camera.utils import Utils
from camera.video_stream import VideoStream
from detector.Detector import Detector
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

utils = Utils()


left_cam = Camera(
    cv.VideoCapture(0),
    utils.loadCalibrationResultFrom("dataset/opencv_sample/left/calibration.npz"),
    "Left Camera",
    cv_index=0,
    framerate=30,
)


right_cam = Camera(
    cv.VideoCapture(2),
    utils.loadCalibrationResultFrom("dataset/opencv_sample/right/calibration.npz"),
    "Right Camera",
    cv_index=2,
    framerate=30,
)


stereo_camera = StereoCamera(
    left_camera=left_cam,
    right_camera=right_cam,
    params=DepthEstimationParams(
        baseline=87,
        block_size=7,
        num_disparities=16 * 4,
        min_disparity=-1,
        disp12MaxDiff=10,
        uniquenessRatio=15,
        speckle_window_size=50,
        speckleRange=1,
    ),
    results=utils.loadStereoCalibrationResultFrom("/home/qulx/Dev/FSC/dataset/opencv_sample/stereoCalibraton.npz")
)

Limg = cv.imread("/home/qulx/Dev/FSC/dataset/our_dataset/calibration2/Tleft/1732513638.6453598.png")
Rimg = cv.imread("/home/qulx/Dev/FSC/dataset/our_dataset/calibration2/Tright/1732513638.6639345.png")

while True:
    ret, Lframe = left_cam.capture.read()
    ret, Rframe = left_cam.capture.read()
    disparity = stereo_camera.Test(Lframe, Rframe)
    cv.imshow(
        "Disparity", cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX) # type: ignore
    )
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()