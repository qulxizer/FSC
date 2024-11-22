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
        depth = DepthEstimation(
            params=self.stereo_calibration_params
        )

        stereo_calibration_result = depth.stereoCalibrate(
                                self.left_camera.calibration_result,
                                self.right_camera.calibration_result)

        h, w, c = Limg.shape        

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
        
        disparity = depth.generateDisparity(undistorted_Limg, undistorted_Rimg)

        depth_map = cv.reprojectImageTo3D(disparity, stereo_rectification_result.Q)[:, :, 2]  # Z channel is depth

        # # Compute disparity
        # disparity = stereo.compute(left_gray, right_gray).astype(np.float32)
        # disparity_vis = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX) # type: ignore
        # disparity[disparity == 0] = 1e-5
        # depth = (self.stereo_calibration_params.focal_length * self.stereo_calibration_params.baseline) / disparity # type: ignore
        # depth_vis = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX) # type: ignore

        # cv.imwrite("tmp/disparity.png", disparity_vis)
        # cv.imwrite("tmp/depth_map.png", depth_vis)

        
        # x, y = 150, 100
        # # Get disparity value at the specific pixel
        # d = disparity[y, x]

        # # Avoid division by zero
        # if d > 0:
            
        #     Z = (self.stereo_calibration_params.focal_length * self.stereo_calibration_params.baseline) / d # type: ignore
        #     print(f"Depth (Z) at pixel ({x}, {y}): {Z:.2f} mm")
        # else:
        #     print(f"Disparity at pixel ({x}, {y}) is zero; depth is undefined.")


        
        # cv.imshow("Depth Map", depth_vis) # type: ignore
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        

