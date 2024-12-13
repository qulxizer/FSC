#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cstring> // for memcpy
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

void sendCvMatToSharedMemory(const cv::Mat &mat, const std::string &shmName) {
  // Calculate the total size needed
  size_t dataSize = mat.total() * mat.elemSize();

  // Create shared memory
  boost::interprocess::shared_memory_object shm(
      boost::interprocess::create_only, shmName.c_str(),
      boost::interprocess::read_write);
  shm.truncate(dataSize);

  // Map shared memory
  boost::interprocess::mapped_region region(shm,
                                            boost::interprocess::read_write);

  // Copy data to shared memory
  std::memcpy(region.get_address(), mat.data, dataSize);
}

int main() {
  // Example: Read an image
  cv::Mat image = cv::imread("tmp/tomato.jpg", cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Failed to load image\n";
    return -1;
  }

  // Send the image to shared memory
  sendCvMatToSharedMemory(image, "shared_image");

  std::cout << "Image sent to shared memory.\n";
  return 0;
}
