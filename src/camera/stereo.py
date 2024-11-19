# import threading
import time
import cv2 as cv
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
        self.left_camera.calibration_result.ObjectPoints
        # ObjectPoints should be the same from the right and left image opencv only one
        cv.stereoCalibrate(
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

        Lcam = VideoStream(self.left_camera)
        Rcam = VideoStream(self.right_camera)

        if not Lcam.ret:
            print("Error: Left camera failed to initialize.")
        if not Rcam.ret:
            print("Error: Right camera failed to initialize.")
        
        Lframe = Lcam.read()
        Rframe = Rcam.read()

        # cv.imshow("Before Left", Lframe)
        # cv.imshow("Before Right", Rframe)

        
        # cv.imshow("After Left", left_undistorted)
        # cv.imshow("After Right", right_undistorted)

            # time.sleep(100)
            # cv.destroyAllWindows()
        try:
            while True:
                left_undistorted = cv.undistort(Lframe,
                                self.left_camera.calibration_result.CameraMatrix,
                                self.left_camera.calibration_result.Distortion)
                right_undistorted = cv.undistort(Rframe,
                                                self.left_camera.calibration_result.CameraMatrix,
                                                self.left_camera.calibration_result.Distortion)
                # Read the right camera frame
                Rframe = Rcam.read()
                if Rframe is not None and Rframe.size > 0:
                    pass
                    # cv.imshow(self.right_camera.name, Rframe)
                else:
                    print("Empty or invalid right frame!")

                # Read the left camera frame
                Lframe = Lcam.read()
                if Lframe is not None and Lframe.size > 0:
                    pass
                    # cv.imshow(self.left_camera.name, Lframe)
                else:
                    print("Empty or invalid left frame!")

                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

                disparity = depth.depthMap(Lframe   , Rframe)
                distance = depth.getDistance(disparity, 320, 240)
                if distance != None:
                    print(distance)

                Lframe = cv.GaussianBlur(Lframe, (5,5), 1)
                Rframe = cv.GaussianBlur(Rframe, (5,5), 1)
                norm_image = cv.normalize(disparity, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
                cv.imshow("depth_map", norm_image)

        finally:
            # Stop both cameras and clean up
            Lcam.stop()
            Rcam.stop()
            cv.destroyAllWindows()
