import cv2 as cv

class Utils(object):
    """docstring for Utils."""
    def __init__(self):
        pass

    
    def ListPorts(self, num:int):
        """
        This utility method will try to connect check each camera index
        until it reach the provided number and it will print working
        and unworking ports with the resolution.
        """

        for i in range(0,num): # if there are more than 5 non working ports stop the testing. 
            camera = cv.VideoCapture(i)
            if not camera.isOpened():
                print("Port %s is not working." %i)
            else:
                is_reading, _ = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    print("Port %s is working and reads images (%s x %s)" %(i,h,w))
                else:
                    print("Port %s for camera ( %s x %s) is present but does not reads." %(i,h,w))
