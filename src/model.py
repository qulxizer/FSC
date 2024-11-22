from dataclasses import dataclass
import cv2 as cv
import numpy as np

@dataclass
class CalibrationResult(object):
    """Docstring for CalibrationResult."""
    Distortion: np.ndarray
    CameraMatrix: np.ndarray
    ObjectPoints: np.ndarray
    ImagePoints: np.ndarray

from dataclasses import dataclass


@dataclass
class StereoCalibrationResults(object):
    """Docstring for StereoCalibrationResults."""
    ret: float
    camera_matrix_left: cv.typing.MatLike
    dist_coeffs_left: cv.typing.MatLike
    camera_matrix_right: cv.typing.MatLike
    dist_coeffs_right: cv.typing.MatLike
    R: cv.typing.MatLike
    T:cv.typing.MatLike
    E:cv.typing.MatLike
    F:cv.typing.MatLike

from dataclasses import dataclass


@dataclass
class StereoRectificationResult(object):
    """Docstring for StereoRectigicationResult."""
    R1: cv.typing.MatLike
    R2: cv.typing.MatLike
    P1: cv.typing.MatLike
    P2: cv.typing.MatLike
    Q: cv.typing.MatLike

    
    
@dataclass
class StereoCalibrationParams:
    """Stereo vision parameters for calibration and depth estimation.
        leave focal_length empty if you want to automaticly generate it.
    """
    
    # Camera calibration parameters
    focal_length: None                      # Default focal length (in pixels)
    baseline: float                         # Default baseline (distance between cameras in mm)
    
    # Stereo matching parameters
    min_disparity: int = 0                    # Default min disparity (in pixels)
    num_disparities: int = 16                 # Default number of disparities (must be a multiple of 16)
    block_size: int = 15                     # Default block size for block matching
    disparity_range: int = 128               # Default max disparity range (in pixels)
    
    # parameters for fine-tuning stereo matching
    uniquenessRatio: int = 10                # Default uniqueness ratio
    speckle_window_size: int = 100             # Default window size for speckle filtering
    speckleRange: int = 32                   # Default maximum allowed disparity variation in speckle filtering
    disp12MaxDiff: int = 1          
    

@dataclass
class Camera:
    """Camera dataclass contains general information about
    the camera framerate cv_index. """
    capture: cv.VideoCapture 
    calibration_result: CalibrationResult
    name: str
    cv_index: int
    framerate: int





