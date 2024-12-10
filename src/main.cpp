#include <iostream>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>


float fx{};  // Focal length x
float fy{};  // Focal length y
float cx{};  // Principal point x
float cy{};  // Principal point y



/*
    Calculate the 3d coordinates for the provided u, x (x, y) points
    by using the camera intrinsic parameters.
*/
void calculate3DCoordinates(int u, int v, float depth) {
    
    // Depth (Z) is the depth value we got from the depth map
    float Z = depth;

    /*
        Calculating the X & Y, Want to read more?
        https://www.cs.cornell.edu/courses/cs664/2008sp/handouts/cs664-9-camera-geometry.pdf

        tl;dr?
        This formula is based on the pinhole camera model. It converts 2D image coordinates (u, v)
        to 3D world coordinates (X, Y, Z) using camera intrinsic parameters:
        - fx, fy: focal lengths.
        - cx, cy: principal point (image center).
        - Z: depth of the point.
    */
    float X = (u - cx) * Z / fx;
    float Y = (v - cy) * Z / fy;

    // Printing the x,y,z in cm by dividing the number by 10
    printf("X: %.3f cm, Y: %.3f cm, Z: %.3f cm. \n", X/10, Y/10, Z/10);
}

// handler for opencv::setCursor
void mouseCallback(int event, int x, int y, int flags, void *userdata)
{

    // Getting the depthMat from userdata by dereferencing and casting to the desired type
    cv::Mat depthMat = *((cv::Mat*)userdata);
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (x >= 0 && x < depthMat.cols && y >= 0 && y < depthMat.rows)
        {
            std::cout << "x:" << x << "y:" << y;

            // Read the depth value from the image at the clicked point
            float depthValue = depthMat.at<float>(y, x);
            if (depthValue > 0)
            {
                std::cout << "Distance at (" << x << ", " << y << "): " << depthValue/1000 << " meters" << std::endl;
                calculate3DCoordinates(x, y, depthValue);
            }
            else
            {
                std::cout << "No valid depth data at this point!" << std::endl;
            }
        }
    }
}

void processFrame(libfreenect2::Frame *depthFrame, libfreenect2::Freenect2Device *device)
{
    if (!depthFrame)
        return;

    // Use OpenCV Mat to process the depth frame
    cv::Mat depthMat(depthFrame->height, depthFrame->width, CV_32FC1, depthFrame->data);
    depthMat = depthMat;
    // Create a registration object for undistortion
    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame unDistortedDepth(depthFrame->width, depthFrame->height, 4);

    // Perform the undistortion
    registration.undistortDepth(depthFrame, &unDistortedDepth);

    // Now convert the undistorted depth data to OpenCV Mat
    cv::Mat undistortedMat(unDistortedDepth.height, unDistortedDepth.width, CV_32FC1, unDistortedDepth.data);

    // Display the frame
    cv::imshow("Depth Frame", undistortedMat);
    cv::setMouseCallback("Depth Frame", mouseCallback, &depthMat);
    if ((char)cv::waitKey(1) == 'q')
    {
        exit(0);
    }
}

void captureFrames(libfreenect2::SyncMultiFrameListener *listener, libfreenect2::Freenect2Device *device)
{
    libfreenect2::FrameMap frames;
    while (true)
    {
        if (!listener->waitForNewFrame(frames, 10 * 1000))
        { // 10-second timeout
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

int main()
{
    // Initialize Freenect2
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *device = nullptr;

    // Discover and open the first available device
    if (freenect2.enumerateDevices() == 0)
    {
        std::cerr << "No Kinect v2 devices found!" << std::endl;
        return -1;
    }

    std::string serial = freenect2.getDefaultDeviceSerialNumber();
    device = freenect2.openDevice(serial);

    if (!device)
    {
        std::cerr << "Failed to open device!" << std::endl;
        return -1;
    }

    // Setup listeners for color and depth frames
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);

    // Start the device
    device->start();

    std::cout << "Kinect v2 started. Press Ctrl+C to quit, Or Press Q in the preview." << std::endl;

    cx = device->getIrCameraParams().cx;
    cy = device->getIrCameraParams().cy;
    fx = device->getIrCameraParams().fx;
    fy = device->getIrCameraParams().fy;


    // Start a thread to capture frames
    std::thread captureThread(captureFrames, &listener, device);

    // Main loop for processing frames (loop is kept to ensure continuous processing)
    while (true)
    {
        // Just loop and allow the other thread to handle frame capturing and processing
    }

    // Join the capture thread before exiting
    captureThread.join();

    // Stop the device
    device->stop();
    device->close();
    delete device;

    return 0;
}
