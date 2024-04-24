/*
nyck33@lenovo-gtx1650:/mnt/d/LLVM/Lean/CUDA$ nvcc -o testNoCublasCovarMatrix noCublasDirver.cpp cudaComputePortfolioRisk.cu -lcudart
nyck33@lenovo-gtx1650:/mnt/d/LLVM/Lean/CUDA$ ./testNoCublasCovarMatrix      
GPU computation time: 1.01582 seconds.
CPU computation time: 8.46907 seconds.
Results are correct.
*/

#include <cuda.h>

#include <cuda_runtime.h>

// CUDA kernel to perform the matrix multiplication A * B
__global__ void matrixMultiply(double* A, double* B, double* C, int ARows, int ACols, int BCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ARows && col < BCols) {
        double sum = 0.0;
        for (int k = 0; k < ACols; ++k) {
            sum += A[row * ACols + k] * B[k * BCols + col];
        }
        C[row * BCols + col] = sum;
    }
}

// Host function to initialize memory, call the kernels, and clean up
extern "C" void computeCovarianceMatrix(double* S, double* R, double* Sigma, int sRows, int sCols) {
    double *d_S, *d_R, *d_T, *d_Sigma;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&d_S, sRows * sCols * sizeof(double));
    cudaMallocManaged(&d_R, sCols * sCols * sizeof(double));
    cudaMallocManaged(&d_T, sRows * sCols * sizeof(double)); // Intermediate result
    cudaMallocManaged(&d_Sigma, sRows * sRows * sizeof(double));

    // Copy data into managed memory
    cudaMemcpy(d_S, S, sRows * sCols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, sCols * sCols * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid1((sCols + 15) / 16, (sRows + 15) / 16);
    dim3 blocksPerGrid2((sRows + 15) / 16, (sRows + 15) / 16);

    // Perform S * R = T
    matrixMultiply<<<blocksPerGrid1, threadsPerBlock>>>(d_S, d_R, d_T, sRows, sCols, sCols);

    // Perform T * S^T = Sigma (assuming S is square and sCols == sRows)
    matrixMultiply<<<blocksPerGrid2, threadsPerBlock>>>(d_T, d_S, d_Sigma, sRows, sCols, sRows);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy the result matrix back to the host memory
    cudaMemcpy(Sigma, d_Sigma, sRows * sRows * sizeof(double), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_S);
    cudaFree(d_R);
    cudaFree(d_T);
    cudaFree(d_Sigma);
}
