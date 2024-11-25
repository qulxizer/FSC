import sys
from camera.utils import Utils, Format
import numpy as np


utils = Utils()

left_camera_directory = sys.argv[1]

# Y, X
res_left = utils.calibrateCamera(6,8, left_camera_directory, Format.PNG)

right_camera_directory = sys.argv[2]

# Y, X
res_right = utils.calibrateCamera(6,8, right_camera_directory, Format.PNG)

if res_left == None:
    print("Right Result is empty")
if res_right == None:
    print("Left Result is empty")

# Ensure the lengths of valid images are the same
if res_left is not None and res_right is not None:
    min_length = min(len(res_left.ObjectPoints), len(res_right.ObjectPoints))

    # Truncate the results to the smaller size
    res_left = utils.truncateCalibrationResult(res_left, min_length)
    res_right = utils.truncateCalibrationResult(res_right, min_length)

    # Save the calibration results
    utils.saveCalibrationResult(res_left, left_camera_directory + "calibration")
    utils.saveCalibrationResult(res_right, right_camera_directory + "calibration")
