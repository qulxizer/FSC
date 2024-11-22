# import threading
import time
import cv2 as cv
import numpy as np

from .utils import Utils
from .video_stream import VideoStream
from .depth_estimation import DepthEstimation
from model import Camera, StereoCalibrationParams

class StereoCamera():
    """docstring for StereoCamera."""
    def __init__(self,
                left_camera:Camera,
                right_camera:Camera,
                params: StereoCalibrationParams,
                ):
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.stereo_calibration_params = params
        if params.focal_length == None:
            # Getting left_camera focal length cause usually the left camera used as the refrence frame
            f_x = left_camera.calibration_result.CameraMatrix[0, 0]  # Focal length along the x-axis
            f_y = left_camera.calibration_result.CameraMatrix[1, 1]  # Focal length along the y-axis
            params.focal_length = (f_x + f_y) / 2



    def Test(self):
        depth = DepthEstimation(
            params=self.stereo_calibration_params
        )
        utils = Utils()

        # ObjectPoints should be the same from the right and left image opencv only one
        ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right ,R ,T, E, F =cv.stereoCalibrate(
            objectPoints=self.left_camera.calibration_result.ObjectPoints, # type: ignore
            imagePoints1=self.left_camera.calibration_result.ImagePoints, # type: ignore
            imagePoints2=self.right_camera.calibration_result.ImagePoints, # type: ignore
            cameraMatrix1=self.left_camera.calibration_result.CameraMatrix,
            cameraMatrix2=self.right_camera.calibration_result.CameraMatrix,
            distCoeffs1=self.left_camera.calibration_result.Distortion,
            distCoeffs2=self.right_camera.calibration_result.Distortion,
            imageSize=(640,480),
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=cv.CALIB_FIX_INTRINSIC
        ) # type: ignore

        Limg = cv.imread("dataset/opencv_sample/left/left.png")
        Rimg = cv.imread("dataset/opencv_sample/right/right.png")

        h, w, _ = Limg.shape
        rectified_left = utils.unDistortImage(Limg, self.left_camera.calibration_result, w,h)
        rectified_right = utils.unDistortImage(Rimg, self.right_camera.calibration_result, w,h)

        cv.imwrite("tmp/rectified_left.png", rectified_left)
        cv.imwrite("tmp/rectified_right.png", rectified_right)


        # Create a StereoSGBM matcher
        stereo = cv.StereoSGBM.create(
            minDisparity=0,
            numDisparities=16*2,  # Must be divisible by 16
            blockSize=15,
            P1=8 * 3 * 5**2,  # P1 and P2 are tuning parameters
            P2=32 * 3 * 5**2,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            disp12MaxDiff=1
        )
        left_gray = cv.cvtColor(rectified_left, cv.COLOR_BGR2GRAY)
        right_gray = cv.cvtColor(rectified_right, cv.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        # disparity_vis = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX) # type: ignore
        disparity[disparity == 0] = 1e-5
        depth = (self.stereo_calibration_params.focal_length * self.stereo_calibration_params.baseline) / disparity
        depth_vis = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)

        cv.imshow("Depth Map", depth_vis)
        cv.waitKey(0)
        cv.destroyAllWindows()

        

