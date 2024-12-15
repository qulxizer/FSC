#include <opencv4/opencv2/opencv.hpp>
#include <stdlib.h>

#ifndef FRAMEPROCESSING_H // Header guards
#define FRAMEPROCESSING_H
void sendCvMatToSharedMemory(const cv::Mat &mat, const std::string &shmName);
#endif
