import numpy as np
from src.camera.depth_estimation import DepthEstimation
from matplotlib import pyplot as plt
import cv2 as cv

imgL = cv.imread("dataset/middlebury_edu/backpack/left.png", cv.IMREAD_GRAYSCALE)
imgR = cv.imread("dataset/middlebury_edu/backpack/right.png", cv.IMREAD_GRAYSCALE)
depth_estimator = DepthEstimation(
                                numDisparities=(229 - 28) // 16 * 16,
                                minDisparity=28,
                                block_size=11,
                                baseline=174.945,
                                focal_length=7190.247,
                                )
def test_estimation():
    disparity = depth_estimator.depthMap(imgL=imgL, imgR=imgR)
    plt.imshow(disparity, "grey") # type: ignore
    plt.show() # type: ignore
    while True:
        if plt.waitforbuttonpress():  # Wait for a key or mouse click
                plt.close()
                break

def test_get_distance():
    x, y = 1000, 1000
    disparity = depth_estimator.depthMap(imgL=imgL, imgR=imgR)
    estimated_distance = depth_estimator.getDistance(disparity, x=x,y=y)
    print(f"estimated distance: {estimated_distance/1000}m" )

    ground_truth_disparity = cv.imread("dataset/middlebury_edu/backpack/ground_truth.pgm", cv.IMREAD_UNCHANGED)
    disparity_value = ground_truth_disparity[y, x]
    distance = (depth_estimator.focal_length * depth_estimator.baseline) / disparity_value
    print(f"distance: {distance/1000}m" )

