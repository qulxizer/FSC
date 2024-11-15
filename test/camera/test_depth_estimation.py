from src.camera.depth_estimation import DepthEstimation
from matplotlib import pyplot as plt
import cv2 as cv

imgL = cv.imread("dataset/middlebury_edu/tsukuba/left1.ppm", cv.IMREAD_GRAYSCALE)
imgR = cv.imread("dataset/middlebury_edu/tsukuba/right1.ppm", cv.IMREAD_GRAYSCALE)
depth_estimator = DepthEstimation(numDisparities=16 * 6,
                                  minDisparity=0,
                                block_size=7,
                                baseline=193.001,
                                focal_length=3997.684,
                                )
def test_estimation():
    disparity = depth_estimator.estimate(imgL=imgL, imgR=imgR)
    plt.imshow(disparity, "grey") # type: ignore
    plt.show() # type: ignore
    while True:
        if plt.waitforbuttonpress():  # Wait for a key or mouse click
                plt.close()
                break

def test_get_distance():
    disparity_map = depth_estimator.estimate(imgL=imgL, imgR=imgR)
    distance = depth_estimator.get_distance(disparity_map, 175, 150)
    print(f"distance {distance/1000} in meter")

