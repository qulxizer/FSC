#ifndef CALCULATION_H  // Header guards
#define CALCULATION_H

void calculate3DCoordinates(int u, int v, float depth, libfreenect2::Freenect2Device::IrCameraParams params); 
libfreenect2::Freenect2Device::IrCameraParams getIntrinsicParameters(libfreenect2::Freenect2Device* dev);

#endif
