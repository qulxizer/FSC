import cv2 as cv
from model import StereoCalibrationParams , Camera, CalibrationResult, StereoCalibrationResults, StereoRectificationResult
from .utils import Utils
import numpy as np

utils = Utils()


class DepthEstimation(object):
    """docstring for DepthEstimation."""

    def __init__(self, params:StereoCalibrationParams):
        self.baseline = params.baseline
        self.focal_length = params.focal_length
        self.block_size = params.block_size
        self.disp12MaxDiff = params.disp12MaxDiff
        self.minDisparity = params.min_disparity
        self.numDisparities = params.num_disparities
        self.speckleRange = params.speckleRange
        self.disparity_range = params.disparity_range
        self.uniquenessRatio = params.uniquenessRatio
        self.speckleWindowSize = params.speckle_window_size

    def stereoCalibrate(self, Lcam_calibration_result:CalibrationResult, Rcam_calibration_result:CalibrationResult) -> StereoCalibrationResults:

        # ObjectPoints should be the same from the right and left image opencv only one
        ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right ,R ,T, E, F = cv.stereoCalibrate(
            objectPoints=Lcam_calibration_result.ObjectPoints, # type: ignore
            imagePoints1=Lcam_calibration_result.ImagePoints, # type: ignore
            imagePoints2=Rcam_calibration_result.ImagePoints, # type: ignore
            cameraMatrix1=Lcam_calibration_result.CameraMatrix,
            cameraMatrix2=Rcam_calibration_result.CameraMatrix,
            distCoeffs1=Lcam_calibration_result.Distortion,
            distCoeffs2=Rcam_calibration_result.Distortion,
            imageSize=(640,480),
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=cv.CALIB_FIX_INTRINSIC
        ) # type: ignore
        return StereoCalibrationResults(
            ret=ret,
            camera_matrix_left=camera_matrix_left,
            dist_coeffs_left=dist_coeffs_left,
            camera_matrix_right=camera_matrix_right,
            dist_coeffs_right=dist_coeffs_right,
            R=R,
            T=T,
            E=E,
            F=F
        ) # type: ignore
    
    def stereoUnDistort(self, Limg:cv.typing.MatLike, Limg_calib:CalibrationResult,
                        Rimg:cv.typing.MatLike, Rimg_calib:CalibrationResult ) -> (cv.typing.MatLike, cv.typing.MatLike): # type: ignore
        if Limg.shape != Rimg.shape:
            ValueError("Left and Right images should be the same size")
    
        h, w, _ = Limg.shape
        undistorted_Limg = utils.unDistortImage(Limg,Limg_calib, w=w, h=h)
        undistorted_Rimg = utils.unDistortImage(Limg,Rimg_calib, w=w, h=h)

        return undistorted_Limg,  undistorted_Rimg

        
    def stereoRectify(self, calib_left:CalibrationResult, calib_right: CalibrationResult, R,T, w:int,h:int) -> StereoRectificationResult:
        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
            calib_left.CameraMatrix,
            calib_left.Distortion,
            calib_right.CameraMatrix,
            calib_right.Distortion,
            (w,h),
            R,
            T,
            flags=cv.CALIB_ZERO_DISPARITY,
            alpha=0
        )
        return StereoRectificationResult(
            R1,
            R2,
            P1,
            P2,
            Q
        )

    def generateDisparity(self,
                imgL:cv.typing.MatLike,
                imgR:cv.typing.MatLike,
                normalize=False) -> cv.typing.MatLike:
        """ This method takes two images and generate the disparity
        using cv.StereoSGBM. 
        """

        imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)


        stereo = cv.StereoSGBM.create(
            minDisparity=self.minDisparity,
            numDisparities=self.numDisparities,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size**2,  # Smoothness for small changes  
            P2=32 * 3 * self.block_size**2, # Keeps edges sharp  
            )
        disparity = stereo.compute(imgL,imgR)
        if normalize:
            return cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8) # type: ignore
        return disparity
    
    def getDistance(self, disparity: cv.typing.MatLike, coordinates: tuple[int, int]) -> float:
        x, y = coordinates
        if self.focal_length is None or self.baseline is None:
            raise ValueError("Focal length or baseline are None")
        disparity_value = disparity[y, x]  # Use (y, x) for OpenCV image indexing
        if disparity_value == 0:
            raise ValueError(f"Disparity at {coordinates} is zero; depth is undefined.")
        return (self.focal_length * self.baseline) / disparity_value

    
    def generateDepthMap(self, arg):
        pass

