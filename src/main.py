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

datasetDirectory = "/home/raspberry/FSC/dataset/our_dataset/calibration2"


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
        block_size=4,
        num_disparities=96,
        min_disparity=0,
        disp12MaxDiff=10,
        uniquenessRatio=0,
        speckle_window_size=10,
        speckleRange=1,
    ),
    results=utils.loadStereoCalibrationResultFrom(f"{datasetDirectory}/stereoCalibration.npz")
)

Limg = cv.imread("/home/raspberry/FSC/dataset/our_dataset/calibration2/Tleft/1732513638.6453598.png")
Rimg = cv.imread("/home/raspberry/FSC/dataset/our_dataset/calibration2/Tright/1732513638.6639345.png")

while True:    
    ret, Lframe = left_cam.capture.read()
    ret, Rframe = right_cam.capture.read()
    map = stereo_camera.Test(Lframe, Rframe)
    cv.imshow("Map", map)
    # cv.imshow("Right Image", Rimg)


    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    # plt.imshow(depth_map, "grey")
    # plt.show()


cv.destroyAllWindows()
# cv.waitKey()
    
