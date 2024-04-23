//nvcc -arch=sm_75 --compiler-options '-fPIC' -shared -o libcovariance.so cublasComputePortfolioRisk.cu -lcublas
#include <cublas_v2.h>

extern "C" {
    void computeCovarianceMatrix(double* s, double* r, double* result, int sRows, int sCols, int rCols, cudaStream_t stream = 0) {
        double* d_s; // Pointer for the device memory of matrix S
        double* d_r; // Pointer for the device memory of matrix R
        double* d_result; // Pointer for the device memory of the result matrix
        double* d_temp; // Temporary device memory to hold intermediate results

        // Allocate device memory for matrices S, R, and the result
        cudaMalloc(&d_s, sRows * sCols * sizeof(double));
        cudaMalloc(&d_r, sCols * rCols * sizeof(double));
        cudaMalloc(&d_result, sRows * rCols * sizeof(double));
        cudaMalloc(&d_temp, sRows * rCols * sizeof(double)); // Memory for the intermediate result S * R

        // Copy matrices S and R from host to device memory
        cudaMemcpy(d_s, s, sRows * sCols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, r, sCols * rCols * sizeof(double), cudaMemcpyHostToDevice);

        // Create a handle for cuBLAS operations
        cublasHandle_t handle;
        cublasCreate(&handle);
        if (stream) {
            cublasSetStream(handle, stream);
        }

        // Define scalar values for the multiplication
        const double alpha = 1.0;
        const double beta = 0.0;


        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rCols, sRows, sCols, &alpha, d_r, rCols, d_s, sCols, &beta, d_temp, rCols);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, sRows, sRows, rCols, &alpha, d_temp, sRows, d_s, sCols, &beta, d_result, sRows);
        
        /*
        // Perform matrix multiplication result = d_temp * S^T
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    sRows, rCols, sCols, &alpha,
                    d_s, sRows, d_r, sCols, &beta,
                    d_temp, sRows);

        // Perform matrix multiplication result = d_temp * S^T
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    sRows, sRows, rCols, &alpha,
                    d_temp, sRows, d_s, sRows, &beta,
                    d_result, sRows);
        */
        // Copy the result matrix from device to host memory
        cudaMemcpy(result, d_result, sRows * sRows * sizeof(double), cudaMemcpyDeviceToHost);

        // Free the device memory allocated for matrices S, R, the result, and temporary storage
        cudaFree(d_s);
        cudaFree(d_r);
        cudaFree(d_result);
        cudaFree(d_temp);

        // Destroy the cuBLAS handle
        cublasDestroy(handle);
    }
}
