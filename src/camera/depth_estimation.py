import cv2 as cv

class DepthEstimation(object):
    """docstring for DepthEstimation."""
    def __init__(self, numDisparities:int, block_size:int, 
                minDisparity:int, baseline:float,
                focal_length:float):
        self.minDisparities = minDisparity
        self.numDisparities = numDisparities
        self.block_size = block_size
        self.baseline = baseline
        self.focal_length = focal_length

    def depthMap(self,imgL:cv.typing.MatLike,
                imgR:cv.typing.MatLike) -> cv.typing.MatLike:
        """ This method takes two images and generate the depth map
        using cv.StereoSGBM. 
        """
        stereo = cv.StereoSGBM.create(
            minDisparity=self.minDisparities,
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
        distance = (self.focal_length * self.baseline) / d
        return distance
