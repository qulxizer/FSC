import cv2 as cv
from model import StereoCalibrationParams 

class DepthEstimation(object):
    """docstring for DepthEstimation."""
    # def __init__(self, numDisparities:int, block_size:int, 
    #             minDisparity:int, baseline:float,
    #             focal_length:float):
    #     self.minDisparities = minDisparity
    #     self.numDisparities = numDisparities
    #     self.block_size = block_size
    #     self.baseline = baseline
    #     self.focal_length = focal_length

    def __init__(self, params:StereoCalibrationParams):
        self.baseline = params.baseline
        self.focal_length = params.focal_length
        self.block_size = params.block_size
        self.disp12MaxDiff = params.disp12MaxDiff
        self.minDisparity = params.minDisparity
        self.numDisparities = params.numDisparities
        self.speckleRange = params.speckleRange
        self.disparity_range = params.disparity_range
        self.uniquenessRatio = params.uniquenessRatio
        self.speckleWindowSize = params.speckleWindowSize

    def depthMap(self,imgL:cv.typing.MatLike,
                imgR:cv.typing.MatLike) -> cv.typing.MatLike:
        """ This method takes two images and generate the depth map
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


        return disparity
    
    def getDistance(self, disparity:cv.typing.MatLike, x:int, y:int):
        """ Get the distance from the disparity at pixel (x, y). """
        d = disparity[y, x]  # Get the disparity value at (x, y)
        
        if d == 0:
            return float('inf')  # If disparity is 0, the object is too far or no match

        # Convert disparity to distance formula: Z = B*f / disparity
        if self.focal_length != None:
            distance = (self.focal_length * self.baseline) / d
            return distance
        else:
            print("focal lenght is none!")
