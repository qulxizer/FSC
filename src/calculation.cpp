#include <iostream>
#include <libfreenect2/libfreenect2.hpp>

libfreenect2::Freenect2Device::IrCameraParams getIntrinsicParameters(libfreenect2::Freenect2Device* dev) {
    return dev->getIrCameraParams();
}

/*
    Calculate the 3d coordinates for the provided u, x (x, y) points
    by using the camera intrinsic parameters.
*/
void calculate3DCoordinates(int u, int v, float depth, libfreenect2::Freenect2Device::IrCameraParams params) {
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
    float X = (u - params.cx) * Z / params.fx;
    float Y = (v - params.cy) * Z / params.fy;

    // Printing the x,y,z in cm by dividing the number by 10
    printf("X: %.3f cm, Y: %.3f cm, Z: %.3f cm. \n", X/10, Y/10, Z/10);
}
