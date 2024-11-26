# import threading
import time
import cv2 as cv
import numpy as np

from .utils import Utils
from .video_stream import VideoStream
from .depth_estimation import DepthEstimation
from model import Camera, StereoCalibrationResults, DepthEstimationParams

class StereoCamera():
    """docstring for StereoCamera."""
    def __init__(self,
                left_camera:Camera,
                right_camera:Camera,
                results: StereoCalibrationResults,
                params:DepthEstimationParams
                ):
        
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.stereo_calibration_params = params
        self.stereo_calibration_results = results
        if params.focal_length == 0:
            # Getting left_camera focal length cause usually the left camera used as the refrence frame
            f_x = left_camera.calibration_result.CameraMatrix[0, 0]  # Focal length along the x-axis
            f_y = left_camera.calibration_result.CameraMatrix[1, 1]  # Focal length along the y-axis
            self.stereo_calibration_params.focal_length = (f_x + f_y) / 2




    def Test(self, Limg:cv.typing.MatLike, Rimg:cv.typing.MatLike):
        h, w, c = Limg.shape  
        depth = DepthEstimation(
            params=self.stereo_calibration_params,
            results=self.stereo_calibration_results
        )


        stereo_rectification_result = depth.stereoRectify(
            self.left_camera.calibration_result.CameraMatrix,
            self.left_camera.calibration_result.Distortion,
            self.right_camera.calibration_result.CameraMatrix,
            self.right_camera.calibration_result.Distortion,
            self.stereo_calibration_results.R,
            self.stereo_calibration_results.T,
            w,
            h)

        undistorted_Limg, undistorted_Rimg = depth.stereoUnDistort(
            Limg,
            self.left_camera.calibration_result.CameraMatrix,
            self.left_camera.calibration_result.Distortion,
            stereo_rectification_result.R1,
            stereo_rectification_result.P1,
            
            Rimg,
            self.right_camera.calibration_result.CameraMatrix,
            self.right_camera.calibration_result.Distortion,
            stereo_rectification_result.R2,
            stereo_rectification_result.P2,

        )
        
        # undistorted_Limg = cv.GaussianBlur(undistorted_Limg, (10,10), 5)
        # undistorted_Rimg = cv.GaussianBlur(undistorted_Rimg, (10,10), 5)
        Ldisparity, Lmatcher = depth.generateDisparity(Limg, Rimg)

        right_matcher = cv.ximgproc.createRightMatcher(Lmatcher);
        left_disp = Lmatcher.compute(Limg, Rimg);
        right_disp = right_matcher.compute(Rimg,Limg);


        sigma = 1.5
        lmbda = 8000.0
        wls_filter = cv.ximgproc.createDisparityWLSFilter(Lmatcher);
        wls_filter.setLambda(lmbda);
        wls_filter.setSigmaColor(sigma);
        filtered_disp = wls_filter.filter(left_disp, Limg, disparity_map_right=right_disp);

        print(depth.getDistance(filtered_disp, (430,400)))
        # color_depth_map = cv.applyColorMap(filtered_disp, cv.COLORMAP_JET)
        # disparity_normalized = cv.normalize(Ldisparity, None, 0, 255, cv.NORM_MINMAX) # type: ignore
        return filtered_disp


        

