#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cublas_v2.h>

extern "C" void computeCovarianceMatrix(double* s, double* r, double* result, int sRows, int sCols, int rCols, cudaStream_t stream = 0);

// CPU-based matrix multiplication
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

// Function to print matrices for debugging
void printMatrix(const std::vector<double>& mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Define the size of the matrices
    int sRows = 1000, sCols = 1000, rCols = 1000;
    std::vector<double> s(sRows * sCols);
    std::vector<double> r(sCols * rCols);
    std::vector<double> result(sRows * rCols);
    std::vector<double> resultCPU(sRows * rCols);
    std::vector<double> temp(sRows * rCols);

    // Initialize matrices S and R with random values
    for (size_t i = 0; i < s.size(); ++i) s[i] = rand() % 10;
    for (size_t i = 0; i < r.size(); ++i) r[i] = rand() % 10;

    // Compute S * R * S^T on CPU for comparison
    auto startCPU = std::chrono::high_resolution_clock::now();
    multiplyMatrices(s, r, temp, sRows, sCols, rCols);  // S * R
    multiplyMatrices(temp, s, resultCPU, sRows, rCols, sCols);  // (S * R) * S^T
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCPU = endCPU - startCPU;
    std::cout << "CPU computation time: " << elapsedCPU.count() << " seconds." << std::endl;

    // Compute S * R * S^T using cuBLAS on GPU
    auto startGPU = std::chrono::high_resolution_clock::now();
    computeCovarianceMatrix(s.data(), r.data(), result.data(), sRows, sCols, rCols, 0);
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedGPU = endGPU - startGPU;
    std::cout << "GPU computation time: " << elapsedGPU.count() << " seconds." << std::endl;

    // Compare GPU and CPU results
    bool correct = true;
    for (size_t i = 0; i < result.size(); ++i) {
        if (std::fabs(result[i] - resultCPU[i]) > 1e-1) {
            correct = false;
            break;
        }
    }

    std::cout << "Results are " << (correct ? "correct." : "incorrect.") << std::endl;

    // Optionally print the resulting matrix
    // printMatrix(result, sRows, rCols);

    return 0;
}
