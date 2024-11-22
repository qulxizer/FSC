import glob
import cv2 as cv
import numpy as np
import json
from time import time
from model import Camera, CalibrationResult
from .video_stream import VideoStream 
from enum import Enum

class Format(Enum):
    JPG = "jpg"
    PNG = "png"     

class Utils(object):
    """Useful utilites."""

    def loadCalibrationResultFrom(self, filename:str) -> CalibrationResult:
        data = np.load(filename)
        return CalibrationResult(
            data["Distortion"],
            data["CameraMatrix"],
            data["ObjectPoints"],
            data["ImagePoints"]
        )

    def saveCalibrationResult(self, result: CalibrationResult, filename: str):
        # Convert the calibration result back into a dictionary
        np.savez_compressed(filename,
                            Distortion=result.Distortion,
                            CameraMatrix=result.CameraMatrix,
                            ImagePoints=result.ImagePoints,
                            ObjectPoints=result.ObjectPoints,
                            )


    def listPorts(self, num:int):
        """
        This utility method will try to connect check each camera index
        until it reach the provided number and it will print working
        and unworking ports with the resolution.
        """

        for i in range(0,num): 
            camera = cv.VideoCapture(i)
            if not camera.isOpened():
                print("Port %s is not working." %i)
            else:
                is_reading, _ = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    print(f"Port {i} is working and reads images ({h} x {w})")
                else:
                    print(f"Port {i} for camera ({h} x {w}) is present but does not reads.")

    def truncateCalibrationResult(self, calibration_result, min_length):
        return CalibrationResult(
            Distortion=calibration_result.Distortion,
            CameraMatrix=calibration_result.CameraMatrix,
            ObjectPoints=calibration_result.ObjectPoints[:min_length],
            ImagePoints=calibration_result.ImagePoints[:min_length],
        )



    def unDistortImage(self, img:cv.typing.MatLike, calibResult:CalibrationResult, w:int,h:int) -> cv.typing.MatLike:

        # # Refining the camera matrix using parameters obtained by calibration
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(calibResult.CameraMatrix, calibResult.Distortion, (w,h), 1, (w,h))
        
        mapx,mapy=cv.initUndistortRectifyMap(
            calibResult.CameraMatrix,
            calibResult.Distortion,
            None, # type: ignore
            newcameramtx,
            (w,h),
            5
            ) # type: ignore
        dst = cv.remap(img,mapx,mapy,cv.INTER_LINEAR)
        return dst


    def captureImageToDirectory(self, camera:Camera,directory:str, key:int):
        """
        This utility method will capture image to the provided directory
        when the key is pressed the key requires the unicode representation
        of the character, to quit press 'q'
        """
        vs = VideoStream(camera=camera)
        try:
            while True:
                frame = vs.read()
                cv.imshow(camera.name, frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

                # check if capture key is pressed it capture image
                if cv.waitKey(1) & 0xFF == key:
                    filename = f"{directory}{str(time())}.png"
                    ret = cv.imwrite(filename, frame)
                    if not ret:
                        print("Failed to write frame.")
                        return 
                    print(f"{filename}, save successfuly")
        finally:
            vs.stop()
            cv.destroyAllWindows()

    def calibrateCamera(self, num_columns:int ,num_rows:int, directory:str, image_type:Format=Format.PNG):
        """
        This utility method will pull image to the provided directory and
        calibrate the images based on the providednum of columns and rows
        """
        # Termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((num_columns*num_rows,3), np.float32)
        objp[:,:2] = np.mgrid[0:num_rows,0:num_columns].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # Load all the image files
        images = glob.glob(f"{directory}*.{image_type.value}")

        # Iterate through each image
        grey = None
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (num_rows,num_columns), None)
        
            if not ret:
                print(f"Chessboard not found in image: {fname}")
                continue
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
        
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
        
                # Draw and display the corners
                cv.drawChessboardCorners(img, (num_rows,num_columns), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(50)
        
        cv.destroyAllWindows()

            
        # Perform camera calibration (None means it should calculate it)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objectPoints=objpoints, # type: ignore
            imagePoints=imgpoints, # type: ignore
            imageSize=gray.shape[::-1], 
            cameraMatrix=None, # type: ignore
            distCoeffs=None # type: ignore
        ) # type: ignore

        # Check calibration result
        if ret:
            print("Calibration successful.")
            print("Camera Matrix:\n", mtx)
            print("Distortion:\n", dist)
            return CalibrationResult(
                                    Distortion=dist,
                                    CameraMatrix=mtx,
                                    ObjectPoints=np.array(objpoints, dtype=np.float32),
                                    ImagePoints=np.array(imgpoints, dtype=np.float32),
                                    )


