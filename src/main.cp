#include "frameProcessing.hpp"
#include <iostream>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>

void captureFrames(libfreenect2::SyncMultiFrameListener *listener,
                   libfreenect2::Freenect2Device *device) {
  libfreenect2::FrameMap frames;
  while (true) {
    if (!listener->waitForNewFrame(frames, 10 * 1000)) { // 10-second timeout
      std::cerr << "Timeout waiting for frames!" << std::endl;
      break;
    }

    // Get the depth frame and process it
    libfreenect2::Frame *depthFrame = frames[libfreenect2::Frame::Depth];
    processFrame(depthFrame, device);

    // Release the frames after processing
    listener->release(frames);
  }
}

int main() {
  // Initialize Freenect2
  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *device = nullptr;

  // Discover and open the first available device
  if (freenect2.enumerateDevices() == 0) {
    std::cerr << "No Kinect v2 devices found!" << std::endl;
    return -1;
  }

  std::string serial = freenect2.getDefaultDeviceSerialNumber();
  device = freenect2.openDevice(serial);

  if (!device) {
    std::cerr << "Failed to open device!" << std::endl;
    return -1;
  }

  // Setup listeners for color and depth frames
  libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color |
                                                libfreenect2::Frame::Depth);
  device->setColorFrameListener(&listener);
  device->setIrAndDepthFrameListener(&listener);

  // Start the device
  device->start();

  std::cout
      << "Kinect v2 started. Press Ctrl+C to quit, Or Press Q in the preview."
      << std::endl;

  // Start a thread to capture frames
  std::thread captureThread(captureFrames, &listener, device);

  // Main loop for processing frames (loop is kept to ensure continuous
  // processing)
  while (true) {
    // Just loop and allow the other thread to handle frame capturing and
    // processing
  }

  // Join the capture thread before exiting
  captureThread.join();

  // Stop the device
  device->stop();
  device->close();
  delete device;

  return 0;
}
