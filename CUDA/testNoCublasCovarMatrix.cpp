#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>


// Forward declaration for the function defined in the CUDA file
extern "C" void computeCovarianceMatrix(double* s, double* r, double* result, int sRows, int sCols, int rCols, cudaStream_t stream = 0);

// CPU-based matrix multiplication for verification
void multiplyMatrices(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }
}

int main() {
    int sRows = 1000, sCols = 1000, rCols = 1000;
    std::vector<double> s(sRows * sCols);
    std::vector<double> r(sCols * rCols);
    std::vector<double> result(sRows * rCols);
    std::vector<double> resultCPU(sRows * rCols);
    std::vector<double> temp(sRows * rCols);

    // Initialize matrices with random data
    for (size_t i = 0; i < s.size(); ++i) s[i] = rand() % 10;
    for (size_t i = 0; i < r.size(); ++i) r[i] = rand() % 10;

    // Compute S * R * S^T on CPU
    auto startCPU = std::chrono::high_resolution_clock::now();
    multiplyMatrices(s, r, temp, sRows, sCols, rCols);
    multiplyMatrices(temp, s, resultCPU, sRows, rCols, sCols);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCPU = endCPU - startCPU;
    std::cout << "CPU computation time: " << elapsedCPU.count() << " seconds." << std::endl;

    // Compute S * R * S^T using CUDA on GPU
    auto startGPU = std::chrono::high_resolution_clock::now();
    computeCovarianceMatrix(s.data(), r.data(), result.data(), sRows, sCols, rCols, 0);
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedGPU = endGPU - startGPU;
    std::cout << "GPU computation time: " << elapsedGPU.count() << " seconds." << std::endl;

    // Compare results
    bool correct = true;
    for (size_t i = 0; i < result.size(); ++i) {
        if (fabs(result[i] - resultCPU[i]) > 1e-5) {
            correct = false;
            //print the result and resultCPU that triggers the error
            std::cout << "Result: " << result[i] << " CPU Result: " << resultCPU[i] << std::endl;
            //print the difference
            std::cout << "Difference: " << fabs(result[i] - resultCPU[i]) << std::endl;
            break;
        }
    }
    std::cout << "Results are " << (correct ? "correct." : "incorrect.") << std::endl;

    return 0;
}
