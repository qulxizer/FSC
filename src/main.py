from model import Camera, StereoCalibrationParams
from camera.stereo import StereoCamera
from camera.utils import Utils
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
        baseline=10,
        block_size=4,
        num_disparities=16 * 7,
        min_disparity=0,
        disp12MaxDiff=10,
        uniquenessRatio=15,
        speckle_window_size=50,
        speckleRange=1,
    ),
)

Limg = cv.imread("/home/qulx/Dev/FSC/dataset/opencv_sample/left/left.png")
Rimg = cv.imread("/home/qulx/Dev/FSC/dataset/opencv_sample/right/right.png")

stereo_camera.Test(Limg, Rimg)
