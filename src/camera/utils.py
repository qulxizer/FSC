import glob
import cv2 as cv
import numpy as np
import json
from time import time
from model import Camera, CalibrationResult
from .video_stream import VideoStream 


class Utils(object):
    """Useful utilites."""

    def loadCalibrationResultFromJson(self, filename:str) -> CalibrationResult:
        with open(filename) as f:
            data = json.load(f)

        # Convert the ImagePoints and ObjectPoints to lists of points
        image_points = [np.array(points, np.float32).reshape(-1, 2) for points in data["ImagePoints"]]
        object_points = [np.array(points, np.float32).reshape(-1, 3) for points in data["ObjectPoints"]]

        return CalibrationResult(
            Distortion=np.array(data["Distortion"], np.float32),
            CameraMatrix=np.array(data["CameraMatrix"], np.float32),
            ImagePoints=image_points, # type: ignore
            ObjectPoints=object_points, # type: ignore
        )

    def saveCalibrationResultJson(self, result: CalibrationResult, filename: str):
        # Convert the calibration result back into a dictionary
        data = {
            "Distortion": result.Distortion.tolist(),
            "CameraMatrix": result.CameraMatrix.tolist(),
            "ImagePoints": result.ImagePoints.tolist(),  # 2D points as list of lists
            "ObjectPoints": result.ObjectPoints.tolist(),  # 3D points as list of lists
        }

        with open(filename, 'w') as file:
            json.dump(data, file)

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

    def calibrateCamera(self, num_columns:int ,num_rows:int, directory:str):
        """
        This utility method will pull image to the provided directory and
        calibrate the images based on the providednum of columns and rows
        """
        # Termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Create base object points for a single image
        objp = np.zeros((num_columns * num_rows, 3), np.float32)
        grid_x, grid_y = np.mgrid[0:num_columns, 0:num_rows]
        objp[:,:2] = np.vstack((grid_x.flatten(), grid_y.flatten())).T

        # Arrays to store object points and image points from all the images
        object_points  = []
        image_points = []

        # Load all the image files
        image_files = glob.glob(f"{directory}*.png")
        print(f"Looking for images in: {directory}")

        gray_image = None
        # Iterate through each image
        for image_file in image_files:
            # Read the image
            image = cv.imread(image_file)
            if image is None:
                print(f"Failed to load image: {image_file}")
                continue

            # Convert the image to grayscale for corner detection
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Find the chessboard corners in the image
            is_found, corners = cv.findChessboardCorners(gray_image, (num_columns, num_rows), None)

            # If corners are found, refine them and store them
            if is_found:
                object_points.append(objp.copy())  # Add a copy of objp to each successful detection
                refined_corners = cv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
                image_points.append(refined_corners)
                
                # Draw and display the corners
                cv.drawChessboardCorners(image, (num_columns, num_rows), refined_corners, is_found)
                cv.imshow('Chessboard Corners', image)
                cv.waitKey(500)

        cv.destroyAllWindows()

        if gray_image is not None and len(object_points) > 0:
            # Convert lists to numpy arrays as required by calibrateCamera
            object_points_array = np.array(object_points, dtype=np.float32)
            image_points_array = np.array(image_points, dtype=np.float32)
            
            # Perform camera calibration (None means it should calculate it)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                objectPoints=object_points_array, # type: ignore
                imagePoints=image_points_array, # type: ignore
                imageSize=gray_image.shape[::-1], 
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
                                        ObjectPoints=object_points_array,
                                        ImagePoints=image_points_array,
                                        )
            else:
                print("Calibration failed.")
        else:
            print("No valid images were found for calibration.")

    def calibrateStereo(self, Lcam:Camera, Rcam:Camera):
        pass
        # cv.stereoCalibrate()