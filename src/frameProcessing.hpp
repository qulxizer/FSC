#ifndef FRAMEPROCESSING_H  // Header guards
#define FRAMEPROCESSING_H

#include <opencv2/opencv.hpp>
#include <libfreenect2/libfreenect2.hpp>

// Function declarations
void previewFrame(cv::Mat& depthMat, libfreenect2::Freenect2Device* device);  // Pass Mat by reference
void processFrame(libfreenect2::Frame* depthFrame, libfreenect2::Freenect2Device* device);

#endif  // FRAMEPROCESSING_H
