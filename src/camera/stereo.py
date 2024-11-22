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
            self.stereo_calibration_params.focal_length = (f_x + f_y) / 2

        



    def Test(self, Limg:cv.typing.MatLike, Rimg:cv.typing.MatLike):
        h, w, c = Limg.shape        
        depth = DepthEstimation(
            params=self.stereo_calibration_params
        )

        stereo_calibration_result = depth.stereoCalibrate(
                                self.left_camera.calibration_result,
                                self.right_camera.calibration_result)


        stereo_rectification_result = depth.stereoRectify(
            self.left_camera.calibration_result,
            self.right_camera.calibration_result,
            stereo_calibration_result.R,
            stereo_calibration_result.T,
            w,
            h)

        undistorted_Limg, undistorted_Rimg = depth.stereoUnDistort(
                            Limg,self.left_camera.calibration_result,
                            Rimg,self.right_camera.calibration_result)
        
        # undistorted_Limg = cv.GaussianBlur(undistorted_Limg, (10,10), 5)
        # undistorted_Rimg = cv.GaussianBlur(undistorted_Rimg, (10,10), 5)

        
        cv.imwrite("tmp/undistorted_Limg.png", undistorted_Limg)
        cv.imwrite("tmp/undistorted_Rimg.png", undistorted_Rimg)


        disparity = depth.generateDisparity(undistorted_Limg, undistorted_Rimg, False)
        cv.imwrite("tmp/disparity.png", disparity)

        
        depth_map = cv.reprojectImageTo3D(disparity, stereo_rectification_result.Q)  # Z channel is depth
        cv.imwrite("tmp/depth_map.png", depth_map)

        print(depth.getDistance(disparity, (350,250)))


        

