# import threading
import time
import cv2 as cv
from matplotlib import pyplot as plt
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
        
        # Set the focal length if not provided
        self._set_focal_length()
        
        # Initialize depth estimation
        self.depth_estimator = DepthEstimation(
            params=self.stereo_calibration_params,
            results=self.stereo_calibration_results
        )

    def _set_focal_length(self):
        """Set the focal length if it is not already provided."""
        if self.stereo_calibration_params.focal_length == 0:
            f_x = self.left_camera.calibration_result.CameraMatrix[0, 0]  # Focal length along the x-axis
            f_y = self.left_camera.calibration_result.CameraMatrix[1, 1]  # Focal length along the y-axis
            self.stereo_calibration_params.focal_length = (f_x + f_y) / 2



    def Test(self, Limg:cv.typing.MatLike, Rimg:cv.typing.MatLike):
        h, w, c = Limg.shape  



        stereo_rectification_result = self.depth_estimator.stereoRectify(
            self.left_camera.calibration_result.CameraMatrix,
            self.left_camera.calibration_result.Distortion,
            self.right_camera.calibration_result.CameraMatrix,
            self.right_camera.calibration_result.Distortion,
            self.stereo_calibration_results.R,
            self.stereo_calibration_results.T,
            w,
            h)

        Limg, Rimg = self.depth_estimator.stereoUnDistort(
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
        cv.imwrite("tmp/undist_limg.png", Limg)
        cv.imwrite("tmp/undist_rimg.png", Rimg)

        # undistorted_Limg = cv.GaussianBlur(undistorted_Limg, (10,10), 10)
        # undistorted_Rimg = cv.GaussianBlur(undistorted_Rimg, (10,10), 10)
        Ldisparity, Lmatcher = self.depth_estimator.generateDisparity(Limg, Rimg)

        right_matcher = cv.ximgproc.createRightMatcher(Lmatcher)
        Rdisparity = right_matcher.compute(Rimg,Limg)


        sigma = 4
        lmbda = 8000.0
        wls_filter = cv.ximgproc.createDisparityWLSFilter(Lmatcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        filtered_disp = wls_filter.filter(Ldisparity, Limg, disparity_map_right=Rdisparity)

        disparity_normalized = cv.normalize(filtered_disp, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1) # type: ignore
        # color_depth_map = cv.applyColorMap(disparity_normalized, cv.COLORMAP_JET)
        return disparity_normalized
    

    def displayRealtimeDisparity(self, Lcam:Camera, Rcam:Camera, title:str):
        fig, ax = plt.subplots()

        ret, Lframe = Lcam.capture.read()
        ret, Rframe = Rcam.capture.read()

        # Initial disparity map (placeholder, will update in the loop)
        disparity = self.Test(Lframe, Rframe)
        im = ax.imshow(disparity, cmap='jet')  # Use 'jet' for better color visualization
        fig.colorbar(im, ax=ax)  # Add a colorbar to show scale
        ax.set_title(title)

        # Define the onclick event to get coordinates
        def onclick(event):
            distance = self.depth_estimator.getDistance(disparity, (event.x, event.y))
            print(f"Distance in ({event.x}, {event.y}) {round(distance)}mm")
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.ion()  # Enable interactive mode
        plt.show()

        while True:
            ret, Lframe = Lcam.capture.read()
            ret, Rframe = Rcam.capture.read()
            disparity = self.Test(Lframe, Rframe)

            # Update the disparity map in the plot
            im.set_data(disparity)
            fig.canvas.draw_idle()

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        plt.ioff()
        plt.show()
            

    def displayDisparityMap(self, Limg:cv.typing.MatLike, Rimg:cv.typing.MatLike, title:str):
        fig, ax = plt.subplots()
        disparity = self.Test(Limg, Rimg)

        # Display disparity as an image
        im = ax.imshow(disparity, cmap='jet')  # Use 'jet' for better color visualization
        fig.colorbar(im, ax=ax)  # Add a colorbar to show scale

        def onclick(event):
            distance = self.depth_estimator.getDistance(disparity, (event.x, event.y))
            print(f"Distance in ({event.x}, {event.y}) {round(distance)}mm")

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()


        

