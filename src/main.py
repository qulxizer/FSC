from model import Camera, StereoCalibrationParams
from camera.stereo import StereoCamera
from camera.utils import Utils
import cv2 as cv

utils = Utils()


left_cam = Camera(
    cv.VideoCapture(2),
    utils.loadCalibrationResultFrom("dataset/opencv_sample/left/calibration.npz"),
    "Left Camera",
    cv_index=0,
    framerate=30
)

right_cam = Camera(
    cv.VideoCapture(0),
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
        baseline=67,
        block_size=17,
        num_disparities=16 * 6,
        min_disparity=0,
    ),
)

# stereo_camera.Test()
