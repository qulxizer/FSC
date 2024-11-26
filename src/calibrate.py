import sys
from camera.utils import Utils, Format
from camera.stereo import StereoCamera
from model import CalibrationResult, StereoCalibrationResults
import numpy as np
from pathlib import Path
import cv2 as cv


utils = Utils()


left_camera_directory = sys.argv[1]


def stereoCalibrate(Lcam_calibration_result:CalibrationResult, Rcam_calibration_result:CalibrationResult) -> StereoCalibrationResults:

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
            left_camera_matrix=camera_matrix_left,
            left_dist_coeffs=dist_coeffs_left,
            right_camera_matrix=camera_matrix_right,
            right_dist_coeffs=dist_coeffs_right,
            R=R,
            T=T,
            E=E,
            F=F
        ) # type: ignore
    

# Y, X
res_left = utils.calibrateCamera(6,8, left_camera_directory, Format.PNG)

right_camera_directory = sys.argv[2]

# Y, X
res_right = utils.calibrateCamera(6,8, right_camera_directory, Format.PNG)

if res_left == None:
    print("Right Result is empty")
if res_right == None:
    print("Left Result is empty")

stereo_calibration_location = sys.argv[3]



# Ensure the lengths of valid images are the same
if res_left != None and res_right != None:


    min_length = min(len(res_left.ObjectPoints), len(res_right.ObjectPoints))

    # Truncate the results to the smaller size
    res_left = utils.truncateCalibrationResult(res_left, min_length)
    res_right = utils.truncateCalibrationResult(res_right, min_length)

    # Save the calibration results
    stereoRes = stereoCalibrate(res_left, res_right)

    utils.saveCalibrationResult(res_left, left_camera_directory + "calibration")
    utils.saveCalibrationResult(res_right, right_camera_directory + "calibration")
    utils.saveStereoCalibrationResult(stereoRes, stereo_calibration_location + "stereoCalibraton")
