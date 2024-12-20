#include <cstddef>
#include <opencv4/opencv2/opencv.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <stdlib.h>
#include <utility>

#ifndef FRAMEPROCESSING_H // Header guards
#define FRAMEPROCESSING_H
void writeDataToSharedMemory(const void *data, const std::size_t &size,
                             const std::string &shmName);

boost::interprocess::mapped_region
readDataFromSharedMemory(const std::string &shmName,
                         const std::chrono::milliseconds &timeout);
#endif
