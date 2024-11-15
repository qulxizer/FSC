from camera.stereo import *
from camera.utils import Utils


# left_cam = Camera(
#     cv_index=0, framerate=30
# )
# right_cam = Camera(
#     cv_index=2, framerate=30
# )

# stereo_camera = StereoCamera(
#     left_camera=left_cam, 
#     right_camera=right_cam
# )

# stereo_camera.capture()

utils = Utils()
utils.ListPorts(6)