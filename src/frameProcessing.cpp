#include "calculation.hpp"
#include "ipc.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <opencv4/opencv2/opencv.hpp>
#include <unistd.h>

struct CallbackData {
  cv::Mat depthMat;
  libfreenect2::Freenect2Device
      *device; // Replace void* with the actual type of 'device'
};

// handler for opencv::setCursor
void mouseCallback(int event, int x, int y, int flags, void *userdata) {

  // Getting the depthMat from userdata by dereferencing and casting to the
  // desired type
  CallbackData *data = reinterpret_cast<CallbackData *>(userdata);
  cv::Mat depthMat = data->depthMat;
  libfreenect2::Freenect2Device *device =
      data->device; // Use 'device' as needed

  if (event == cv::EVENT_LBUTTONDOWN) {
    if (x >= 0 && x < depthMat.cols && y >= 0 && y < depthMat.rows) {
      // Read the depth value from the image at the clicked point
      float depthValue = depthMat.at<float>(y, x);
      if (depthValue > 0) {
        calculate3DCoordinates(x, y, depthValue,
                               getIntrinsicParameters(device));
      } else {
        std::cout << "No valid depth data at this point!" << std::endl;
      }
    }
  }
}

void previewFrame(cv::Mat *depthMat, cv::Mat *coloredMat,
                  libfreenect2::Freenect2Device *device) {
  cv::imshow("Depth Frame", *coloredMat);

  CallbackData data = {*depthMat, device};
  cv::setMouseCallback("Depth Frame", mouseCallback, &data);
  if ((char)cv::waitKey(1) == 'q') {
    exit(0);
  }
}

void processFrame(libfreenect2::Frame *depthFrame,
                  libfreenect2::Frame *colorFrame,
                  libfreenect2::Freenect2Device *device) {
  if (!depthFrame)
    return;

  // Create a registration object for undistortion
  libfreenect2::Registration registration(device->getIrCameraParams(),
                                          device->getColorCameraParams());

  libfreenect2::Frame unDistortedDepth(depthFrame->width, depthFrame->height,
                                       4);

  // Perform the undistortion
  registration.undistortDepth(depthFrame, &unDistortedDepth);

  cv::Mat unDistortedDepthMat(unDistortedDepth.height, unDistortedDepth.width,
                              CV_8UC4, unDistortedDepth.data);

  cv::Mat colorMat(colorFrame->height, colorFrame->width, CV_8UC4,
                   colorFrame->data);

  cv::resize(colorMat, colorMat,
             cv::Size(unDistortedDepthMat.cols, unDistortedDepthMat.rows));
  printf("cols: %i rows: %i\n", colorMat.cols, colorMat.rows);
  // Sending the frame to python script using ipc shared memory
  sendCvMatToSharedMemory(colorMat, "shared_image");

  // previewing Frame
  previewFrame(&unDistortedDepthMat, &colorMat, device);
}
