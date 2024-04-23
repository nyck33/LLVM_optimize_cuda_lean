//nvcc -o testCovarianceMatrix testCovarianceMatrix.cpp cublasComputePortfolioRisk.cu -lcublas

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>

// Declare the external function from the CUDA file
extern "C" void computeCovarianceMatrix(double* s, double* r, double* result, int sRows, int sCols, int rCols, cudaStream_t stream = 0);

int main() {
    // Define the size of the matrices
    int sRows = 1000, sCols = 1000, rCols = 1000;
    size_t numElementsS = sRows * sCols;
    size_t numElementsR = sCols * rCols;
    size_t numElementsResult = sRows * rCols;

    // Allocate host memory for matrices S, R, and the result
    double* s = new double[numElementsS];
    double* r = new double[numElementsR];
    double* result = new double[numElementsResult];

    // Initialize matrices S and R with random values
    for (size_t i = 0; i < numElementsS; ++i) {
        s[i] = rand() % 10;
    }
    for (size_t i = 0; i < numElementsR; ++i) {
        r[i] = rand() % 10;
    }

    // Timing code
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function
    computeCovarianceMatrix(s, r, result, sRows, sCols, rCols, 0);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Execution time: " << elapsed.count() << " seconds." << std::endl;

    // Free host memory
    delete[] s;
    delete[] r;
    delete[] result;

    return 0;
}
