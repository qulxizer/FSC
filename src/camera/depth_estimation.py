import cv2 as cv
from model import DepthEstimationParams , Camera, CalibrationResult, StereoCalibrationResults, StereoRectificationResult
from .utils import Utils
import numpy as np

utils = Utils()


class DepthEstimation(object):
    """docstring for DepthEstimation."""

    def __init__(self, results:StereoCalibrationResults, params:DepthEstimationParams):
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
        self.stereoCalibrationResult = results

    def stereoUnDistort(self, Limg:cv.typing.MatLike,K1,D1,R1,P1,
                        Rimg:cv.typing.MatLike,K2,D2,R2,P2) -> tuple[cv.typing.MatLike, cv.typing.MatLike]:

        if Limg.shape != Rimg.shape:
            ValueError("Left and Right images should be the same size")
    
        h, w, _ = Limg.shape
        undistorted_Limg = utils.unDistortImage(
            Limg,
            K1,
            D1,
            R1,
            P1,
            w,
            h
        )
        undistorted_Rimg = utils.unDistortImage(
            Rimg,
            K2,
            D2,
            R2,
            P2,
            w,
            h
        )
        return undistorted_Limg, undistorted_Rimg

        
    def stereoRectify(self, K1, D1, K2, D2, R, T, w,h) -> StereoRectificationResult:
        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
            K1,
            D1,
            K2,
            D2,
            (w,h),
            R,
            T
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
                imgR:cv.typing.MatLike,) -> tuple[cv.typing.MatLike, cv.StereoSGBM]:
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
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
            )
        disparity = stereo.compute(imgL,imgR)
        return disparity, stereo
    
    def getDistance(self, disparity: cv.typing.MatLike, coordinates: tuple[int, int]) -> float:
        x, y = coordinates
        if self.focal_length is None or self.baseline is None:
            print("Focal length or baseline are None")
        disparity_value = disparity[y, x]  # Use (y, x) for OpenCV image indexing
        if disparity_value == 0:
            print(f"Disparity at {coordinates} is zero; depth is undefined.")
        return (self.focal_length * self.baseline) / disparity_value # type: ignore

    
    def generateDepthMap(self, arg):
        pass

