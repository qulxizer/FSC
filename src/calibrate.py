import sys
from camera.utils import Utils
import numpy as np


utils = Utils()

left_camera_directory = sys.argv[1]
res = utils.calibrateCamera(8,6, left_camera_directory)
if res != None:
    utils.saveCalibrationResultJson(res, left_camera_directory+"calibration.json")

right_camera_directory = sys.argv[2]
res = utils.calibrateCamera(8,6, right_camera_directory)
if res != None:
    utils.saveCalibrationResultJson(res, right_camera_directory+"calibration.json")