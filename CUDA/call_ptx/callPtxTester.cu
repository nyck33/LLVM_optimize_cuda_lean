#include <cuda.h>
#include <iostream>

int main() {
    CUmodule cuModule;
    CUfunction cuFunction;
    cuInit(0);
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);
    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuModuleLoad(&cuModule, "cudaComputePortfolioRisk.ptx");
    cuModuleGetFunction(&cuFunction, cuModule, "_Z14matrixMultiplyPdS_S_iii");

    // Set up kernel parameters and launch the kernel
    // ...

    cuCtxDestroy(cuContext);
    return 0;
}