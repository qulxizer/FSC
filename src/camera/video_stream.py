import cv2 as cv
from threading import Thread
from .stereo import Camera



class VideoStream:
    """
    This class captures video frames in a seperate thread from the main.
    It helps make video faster.
    """
    def __init__(self, camera:Camera):
        self.camera = camera
        self.ret, self.frame = self.camera.capture.read()
        self.stopped = False
        Thread(target=self.update, args=()).start()

    def start(self):
        """
        Runs the video in a new thread.
        :return: This object (so you can chain methods).
        """
        return self

    def update(self):
        """
        Keeps reading frames from the video.
        """
        while not self.stopped:
            self.ret, self.frame = self.camera.capture.read()

    def read(self):
        """
        Gives the current video frame.
        """
        return self.frame

    def stop(self):
        """
        Stops the video and closes the stream.
        """
        self.stopped = True
        self.camera.capture.release()