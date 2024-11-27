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

datasetDirectory = "/home/qulx/Dev/FSC/dataset/our_dataset/calibration3"


left_cam = Camera(
    cv.VideoCapture(0),
    utils.loadCalibrationResultFrom(f"{datasetDirectory}/left/calibration.npz"),
    "Left Camera",
    cv_index=0,
    framerate=30,
)


right_cam = Camera(
    cv.VideoCapture(2),
    utils.loadCalibrationResultFrom(f"{datasetDirectory}/right/calibration.npz"),
    "Right Camera",
    cv_index=2,
    framerate=30,
)


stereo_camera = StereoCamera(
    left_camera=left_cam,
    right_camera=right_cam,
    params=DepthEstimationParams(
        baseline=67,
        block_size=2,
        num_disparities=16 * 6,
        min_disparity=0,
        disp12MaxDiff=10,
        uniquenessRatio=15,
        speckle_window_size=10,
        speckleRange=1,
    ),
    results=utils.loadStereoCalibrationResultFrom(f"{datasetDirectory}/stereoCalibration.npz")
)

Limg = cv.imread("/home/qulx/Dev/FSC/dataset/our_dataset/calibration3/left/1732698042.099758.png")
Rimg = cv.imread("/home/qulx/Dev/FSC/dataset/our_dataset/calibration3/right/1732698042.1186182.png")

stereo_camera.displayDisparityMap(Limg, Rimg, "Disparity Map")

    
