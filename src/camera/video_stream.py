import cv2 as cv
from threading import Thread
from model import Camera

class VideoStream:
    """
    This class captures video frames in a separate thread from the main.
    It helps make video capture faster.
    """
    def __init__(self, camera: Camera):
        self.camera = camera
        self.ret, self.frame = self.camera.capture.read()
        if not self.ret:
            print(f"Failed to initialize camera {self.camera.name}")
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def start(self):
        """
        Starts the video capturing in a separate thread.
        """
        return self

    def update(self):
        """
        Keeps reading frames from the video.
        This runs in a separate thread to keep the main thread free for other tasks.
        """
        while not self.stopped:
            self.ret, self.frame = self.camera.capture.read()
            if not self.ret:
                print(f"Failed to read frame from {self.camera.name}")
                break  # If frame capture fails, stop the loop

    def read(self):
        """
        Returns the latest frame captured by the video stream.
        """
        return self.frame

    def stop(self):
        """
        Stops the video capture and releases the camera.
        """
        self.stopped = True
        self.thread.join()  # Wait for the thread to finish
        self.camera.capture.release()