#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

extern "C" void computeCovarianceMatrix(double* s, double* r, double* result, int sRows, int sCols, int rCols);

// Function to perform matrix multiplication on CPU
void multiplyMatrices(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int M, int N, int P) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// Function to compute S * R * S^T on CPU
void computeCovarianceMatrixCPU(const std::vector<double>& s, const std::vector<double>& r, std::vector<double>& result, int sRows, int sCols, int rCols) {
    std::vector<double> temp(sRows * rCols);
    multiplyMatrices(s, r, temp, sRows, sCols, rCols);
    multiplyMatrices(temp, s, result, sRows, rCols, sCols);
}

int main() {
    int sRows = 1000, sCols = 1000, rCols = 1000;
    std::vector<double> s(sRows * sCols, 1.0); // Initialize matrices with example values
    std::vector<double> r(sCols * rCols, 1.0);
    std::vector<double> resultGPU(sRows * rCols, 0.0);
    std::vector<double> resultCPU(sRows * rCols, 0.0);

    // Compute on GPU
    auto startGPU = std::chrono::high_resolution_clock::now();
    computeCovarianceMatrix(s.data(), r.data(), resultGPU.data(), sRows, sCols, rCols);
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedGPU = endGPU - startGPU;
    std::cout << "GPU computation time: " << elapsedGPU.count() << " seconds." << std::endl;

    // Compute on CPU
    auto startCPU = std::chrono::high_resolution_clock::now();
    computeCovarianceMatrixCPU(s, r, resultCPU, sRows, sCols, rCols);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCPU = endCPU - startCPU;
    std::cout << "CPU computation time: " << elapsedCPU.count() << " seconds." << std::endl;

    // Compare results
    bool correct = true;
    for (size_t i = 0; i < resultGPU.size(); ++i) {
        if (fabs(resultGPU[i] - resultCPU[i]) > 1e-1) { // You might need to adjust this threshold depending on the precision
            correct = false;
            break;
        }
    }
    std::cout << "Results are " << (correct ? "correct." : "incorrect.") << std::endl;

    return 0;
}
