# import threading
import time
import cv2 as cv
import numpy as np
from .video_stream import VideoStream
from .depth_estimation import DepthEstimation
from model import Camera, StereoCalibrationParams

class StereoCamera():
    """docstring for StereoCamera."""
    def __init__(self,
                left_camera:Camera,
                right_camera:Camera,
                params: StereoCalibrationParams,
                ):
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.stereo_calibration_params = params
        if params.focal_length == None:
            # Getting left_camera focal length cause usually the left camera used as the refrence frame
            f_x = left_camera.calibration_result.CameraMatrix[0, 0]  # Focal length along the x-axis
            f_y = left_camera.calibration_result.CameraMatrix[1, 1]  # Focal length along the y-axis
            params.focal_length = (f_x + f_y) / 2



    def Test(self):
        depth = DepthEstimation(
            params=self.stereo_calibration_params
        )
        self.left_camera.calibration_result.ObjectPoints
        # ObjectPoints should be the same from the right and left image opencv only one
        ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right ,R ,T, E, F =cv.stereoCalibrate(
            objectPoints=self.left_camera.calibration_result.ObjectPoints, # type: ignore
            imagePoints1=self.left_camera.calibration_result.ImagePoints, # type: ignore
            imagePoints2=self.right_camera.calibration_result.ImagePoints, # type: ignore
            cameraMatrix1=self.left_camera.calibration_result.CameraMatrix,
            cameraMatrix2=self.right_camera.calibration_result.CameraMatrix,
            distCoeffs1=self.left_camera.calibration_result.Distortion,
            distCoeffs2=self.right_camera.calibration_result.Distortion,
            imageSize=(640,480),
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=cv.CALIB_FIX_INTRINSIC
        ) # type: ignore

        Limg = cv.imread("dataset/opencv_sample/left/left01.jpg")
        Rimg = cv.imread("dataset/opencv_sample/right/right01.jpg")

        # Assume cameraMatrix and distCoeffs are already loaded
        h, w = Limg.shape[:2]
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
            camera_matrix_left, dist_coeffs_left, (w, h), 1, (w, h)
        )

        # dst = cv.undistort(Limg, camera_matrix_left, dist_coeffs_left, None, new_camera_matrix)
        Lmapx, Lmapy = cv.initUndistortRectifyMap(camera_matrix_left, dist_coeffs_left, None, new_camera_matrix, (w,h), 5)
        Ldst = cv.remap(Limg, Lmapx, Lmapy, cv.INTER_LINEAR)


        Rmapx, Rmapy = cv.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, None, new_camera_matrix, (w,h), 5)
        Rdst = cv.remap(Rimg, Rmapx, Rmapy, cv.INTER_LINEAR)


        for y in range(0, Rdst.shape[0], 50):
            cv.line(Rdst, (0, y), (Rdst.shape[1], y), (255, 0, 0), 1)
            cv.line(Rdst, (0, y), (Rdst.shape[1], y), (255, 0, 0), 1)

        cv.imshow("Left", Ldst)
        cv.imshow("Right", Rdst)
        cv.waitKey(10000000)
        cv.destroyAllWindows()
        # cv.waitKey(0)

        # R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(camera_matrix_left,
        #                                            dist_coeffs_left,
        #                                            camera_matrix_right,
        #                                            dist_coeffs_right,
        #                                            Limg.shape[:2], R, T)
        
        # map1_left, map2_left = cv.initUndistortRectifyMap(camera_matrix_left,
        #                                                   dist_coeffs_left,
        #                                                   R1,
        #                                                   P1,
        #                                                   Limg.shape[:2],
        #                                                   cv.CV_32F)
        
        # map1_right, map2_right = cv.initUndistortRectifyMap(camera_matrix_right,
        #                                                     dist_coeffs_right,
        #                                                     R2,
        #                                                     P2,
        #                                                     Limg.shape[:2],
        #                                                     cv.CV_32F)



        # stereo = cv.StereoSGBM_create(
        #     minDisparity=0,
        #     numDisparities=16*6,  # Must be divisible by 16
        #     blockSize=5,
        #     P1=8 * 3 * 5**2,  # 8 * channels * blockSize^2
        #     P2=32 * 3 * 5**2,  # 32 * channels * blockSize^2
        #     disp12MaxDiff=1,
        #     uniquenessRatio=15,
        #     speckleWindowSize=50,
        #     speckleRange=32,
        #     preFilterCap=63
        # )        
        # disparity = stereo.compute(Rdst, Rdst)
        # disparity_vis = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
        # cv.imshow('kiki', disparity_vis)
        # cv.waitKey(1000000)
        # cv.destroyAllWindows()
        # img = cv.imread('dataset/opencv_sample/left/left12.jpg')
        # h,  w = img.shape[:2]
        # Lnewcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix_left, dist_coeffs_left, (w,h), 1, (w,h))

        # # undistort
        # mapx, mapy = cv.initUndistortRectifyMap(camera_matrix_left, dist_coeffs_left, None, Lnewcameramtx, (w,h), 5)
        # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)        
        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        # cv.imshow("undist", dst)
        # cv.waitKey(100000000)
        # cv.destroyAllWindows()
            


        # # Display rectified images
        # cv.imshow("Rectified Left", Rdst)
        # cv.imshow("Rectified Right", Rdst)
        # time.sleep(100)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # stereo = cv.StereoSGBM_create(minDisparity=0, numDisparities=16*8, blockSize=5)

        # # Compute disparity
        # disparity = stereo.compute(Rdst, Rdst).astype(np.float32) / 16.0  # Scaling the disparity
        # print(f"Disparity range: min={np.min(disparity)}, max={np.max(disparity)}")

        ## Show the disparity map
        # disp_vis = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        # cv.imshow("Disp", disp_vis)
        # cv.waitKey(1000000)
        # cv.destroyAllWindows()

        # depth_map = cv.reprojectImageTo3D(disparity, Q)[:, :, 2]  # We are only interested in Z (depth)

        # # Check the depth range
        # print(f"Depth range: min={np.min(depth_map)}, max={np.max(depth_map)}")

        # # Handle invalid depth values (inf, NaN, or non-positive)
        # depth_map[np.isinf(depth_map)] = 0
        # depth_map[np.isnan(depth_map)] = 0
        # depth_map[depth_map <= 0] = 0  # Set non-positive depth values to 0
        
        # depth_map = depth_map * 100  # Adjust this scaling factor as necessary

        # # Normalize the depth map for visualization
        # depth_vis = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        # # Apply colormap for better visualization
        # colored_depth = cv.applyColorMap(depth_vis, cv.COLORMAP_JET)

        # # Show the depth map
        # cv.imshow("Depth Map", colored_depth)
        # cv.waitKey(0)
        # cv.destroyAllWindows()






        
