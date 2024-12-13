#ifndef CALCULATION_H // Header guards
#define CALCULATION_H
#include <libfreenect2/libfreenect2.hpp>

void calculate3DCoordinates(
    int u, int v, float depth,
    libfreenect2::Freenect2Device::IrCameraParams params);
libfreenect2::Freenect2Device::IrCameraParams
getIntrinsicParameters(libfreenect2::Freenect2Device *dev);

#endif
