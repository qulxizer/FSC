#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <opencv4/opencv2/opencv.hpp>

void writeCvMatToSharedMemory(const cv::Mat &coloredMat,
                              const std::string &shmName) {
  // Calculate the total size needed
  size_t dataSize = coloredMat.total() * coloredMat.elemSize();

  // Create shared memory
  boost::interprocess::shared_memory_object shm(
      boost::interprocess::open_or_create, shmName.c_str(),
      boost::interprocess::read_write);
  shm.truncate(dataSize);

  // Map shared memory
  boost::interprocess::mapped_region region(shm,
                                            boost::interprocess::read_write);

  // Copy data to shared memory
  std::memcpy(region.get_address(), coloredMat.data, dataSize);
  printf("frame has been sent to %s \n", shmName.c_str());
}

void writeDataToSharedMemory(const void *data, const std::size_t &size,
                             const std::string &shmName) {

  boost::interprocess::shared_memory_object shm(
      boost::interprocess::open_or_create, shmName.c_str(),
      boost::interprocess::read_write);

  shm.truncate(size);

  boost::interprocess::mapped_region region(shm,
                                            boost::interprocess::read_write);

  std::memcpy(region.get_address(), data, size);
  printf("data has been sent to %s \n", shmName.c_str());
}

std::pair<void *, std::size_t>
readDataFromSharedMemory(const std::string &shmName,
                         const std::chrono::milliseconds &timeout) {
  auto start = std::chrono::high_resolution_clock::now();
  while (true) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (timeout.count() < duration.count()) {
      printf("Timeout exceeded while reading %s \n", shmName.c_str());
      exit(1);
    }

    try {
      boost::interprocess::shared_memory_object shm(
          boost::interprocess::open_only, shmName.c_str(),
          boost::interprocess::read_only);

      boost::interprocess::mapped_region region(shm,
                                                boost::interprocess::read_only);
      return {region.get_address(), region.get_size()};

    } catch (boost::interprocess::interprocess_exception &e) {
      if (errno == ENOENT) {
        continue;
      } else {
        printf("Error opening shared memory: %s", e.what());
      }
    }
  }

  printf("failed to receive data at %s \n", shmName.c_str());
  return {nullptr, 0};
}
