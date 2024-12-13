cmake_minimum_required(VERSION 3.15)
project(FSC)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source files
file(GLOB SOURCES src/*.cpp)

# Exclude main.cpp if needed
# list(FILTER SOURCES EXCLUDE REGEX ".*main.cpp")

# Include directories
include_directories(src)

# Find required packages
find_package(OpenCV REQUIRED)

find_library(FREENECT2_LIBRARIES freenect2 REQUIRED)
find_path(FREENECT2_INCLUDE_DIR libfreenect2/libfreenect2.hpp REQUIRED)

include_directories(${FREENECT2_INCLUDE_DIR})

# Create the library or executable
add_executable(main src/ipc.cpp ${SOURCES})

# Link libraries
target_link_libraries(main ${OpenCV_LIBS} ${FREENECT2_LIBRARIES})
