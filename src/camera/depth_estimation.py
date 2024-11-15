import cv2 as cv

class DepthEstimation(object):
    """docstring for DepthEstimation."""
    def __init__(self, numDisparities:int, blockSize:int, 
                minDisparities:int, baseline:float,
                focal_length:float):
        self.minDisparities = minDisparities
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        self.baseline = baseline
        self.focal_length = focal_length

    def estimate(self,imgL:cv.typing.MatLike,
                imgR:cv.typing.MatLike) -> cv.typing.MatLike:
        stereo = cv.StereoSGBM.create(
            minDisparity=self.minDisparities,
            numDisparities=self.numDisparities,
            blockSize=self.blockSize)
        disparity = stereo.compute(imgL,imgR)
        return disparity
    
    def get_distance(self, disparity:cv.typing.MatLike, x:int, y:int):
        """ Get the distance from the disparity at pixel (x, y) """
        d = disparity[y, x]  # Get the disparity value at (x, y)
        
        if d == 0:
            return float('inf')  # If disparity is 0, the object is too far or no match

        # Convert disparity to distance
        distance = (self.focal_length * self.baseline) / d
        return distance
