#include <cstddef>
#include <opencv4/opencv2/opencv.hpp>
#include <stdlib.h>
#include <utility>

#ifndef FRAMEPROCESSING_H // Header guards
#define FRAMEPROCESSING_H
void writeCvMatToSharedMemory(const cv::Mat &mat, const std::string &shmName);

void writeDataToSharedMemory(const void *data, const std::size_t *size,
                             const std::string &shmName);

std::pair<void *, std::size_t>
readDataFromSharedMemory(const std::string &shmName,
                         const std::chrono::milliseconds &timeout);
#endif
