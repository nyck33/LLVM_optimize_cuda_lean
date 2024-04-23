/*write cuda implementation of the following function
extern "C"
__global__ void cudaComputeCovarianceMatrix(double* s, double* r, double* result, int sRows, int sCols, int rCols) {
    // Use cuBLAS to perform the multiplication S * R * S
    // This is a simplified placeholder. Actual implementation will involve cuBLAS calls and handling the result matrix.
}
Sigma = S * R * S
*/
#include <cublas_v2.h> // Include the cuBLAS library for GPU-accelerated linear algebra operations

extern "C"
__global__ void cudaComputeCovarianceMatrix(double* s, double* r, double* result, int sRows, int sCols, int rCols) {
    // Calculate the row index of the element to be processed by this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of the element to be processed by this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within the bounds of the result matrix
    if(row < sRows && col < rCols) {
        double sum = 0; // Initialize the sum for this element
        // Iterate over the shared dimension to compute the dot product
        for(int k = 0; k < sCols; ++k) {
            sum += s[row * sCols + k] * r[k * rCols + col]; // Accumulate the product of corresponding elements
        }
        result[row * rCols + col] = sum; // Store the computed sum in the result matrix
    }
}

void computeCovarianceMatrix(double* s, double* r, double* result, int sRows, int sCols, int rCols, cudaStream_t stream = 0) {
    double* d_s; // Pointer for the device memory of matrix S
    double* d_r; // Pointer for the device memory of matrix R
    double* d_result; // Pointer for the device memory of the result matrix

    // Allocate device memory for matrices S, R, and the result
    cudaMalloc(&d_s, sRows * sCols * sizeof(double));
    cudaMalloc(&d_r, sCols * rCols * sizeof(double));
    cudaMalloc(&d_result, sRows * rCols * sizeof(double));

    // Copy matrices S and R from host to device memory
    cudaMemcpy(d_s, s, sRows * sCols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, sCols * rCols * sizeof(double), cudaMemcpyHostToDevice);

    // Define the block size and grid size for the kernel launch
    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 blocksPerGrid((rCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (sRows + threadsPerBlock.y - 1) / threadsPerBlock.y); // Calculate the number of blocks needed

    // Launch the CUDA kernel to compute the covariance matrix
    cudaComputeCovarianceMatrix<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_s, d_r, d_result, sRows, sCols, rCols);

    // Copy the result matrix from device to host memory
    cudaMemcpy(result, d_result, sRows * rCols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free the device memory allocated for matrices S, R, and the result
    cudaFree(d_s);
    cudaFree(d_r);
    cudaFree(d_result);
}

