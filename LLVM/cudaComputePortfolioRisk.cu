/*
cgeist cudaComputePortfolioRisk.cu --raise-scf-to-affine -function=* -S --resource-dir=$LLVM_BUILD_DIR/lib/clang/18 --cuda-gpu-arch=sm_75
*/

//set env var
//export LLVM_BUILD_DIR=/mnt/d/LLVM/NewPolygeistDir/llvm-project/build

/* output LLVM
 clang++ -O3 --cuda-gpu-arch=sm_75 --cuda-path=/usr/local/cuda -S -emit-llvm -x cuda --cuda-device-only cudaComputePortfolioRisk.cu -o cudaComputePortfolioRisk.ll
clang++: warning: CUDA version 12.1 is only partially supported [-Wunknown-cuda-version]
*/

/*
nyck33@lenovo-gtx1650:/mnt/d/LLVM/Lean/CUDA$ nvcc -o testNoCublasCovarMatrix noCublasDirver.cpp cudaComputePortfolioRisk.cu -lcudart
nyck33@lenovo-gtx1650:/mnt/d/LLVM/Lean/CUDA$ ./testNoCublasCovarMatrix      
GPU computation time: 1.01582 seconds.
CPU computation time: 8.46907 seconds.
Results are correct.
*/

/*
To apply LLVM optimizations to CUDA code, there are several avenues to explore depending on what exactly you want to optimize and how you integrate CUDA with LLVM-based toolchains. Here are some methods and considerations for optimizing CUDA code using LLVM technologies:

### 1. **Using Clang and LLVM with CUDA**
LLVM's Clang compiler has support for CUDA, which allows it to compile CUDA code. By using Clang, you can leverage LLVM's optimization passes directly on CUDA code.

**Steps to compile CUDA with Clang:**
1. **Install Clang with CUDA Support**: Make sure your version of Clang supports CUDA. This typically requires Clang 3.8 or later.
2. **Compile the CUDA Code**:
   ```bash
   clang++ -O3 --cuda-gpu-arch=sm_75 -L/usr/local/cuda/lib64 -lcudart -lcuda -lstdc++ -include cuda_runtime.h -x cuda --cuda-path=/usr/local/cuda your_cuda_file.cu -o your_program
   ```
   - `-O3`: High level of optimization.
   - `--cuda-gpu-arch=sm_75`: Specify the GPU architecture (adjust based on your GPU).
   - `--cuda-path`: Path to your CUDA installation.
   - Adjust paths and flags based on your setup and requirements.

### 2. **LLVM Passes for Optimization**
After compiling the CUDA kernels into LLVM bitcode using Clang, you can apply LLVM's optimization passes:
- **Generate LLVM Bitcode**: You can modify your build process to emit LLVM bitcode.
- **Optimize Bitcode**: Use `opt` tool from LLVM to apply specific optimization passes.
   ```bash
   opt -O3 -S your_program.bc -o your_program_opt.bc
   ```

### 3. **Profile-Guided Optimizations (PGO)**
If you have access to typical datasets or workload characteristics, you can use profiling tools to guide further optimizations:
- **Generate Profile Data**: Run your application with instrumentation to collect profile data.
- **Recompile with Profile Data**: Use the profile data to guide the LLVM optimizations, which can tailor the optimizations to the specific behavior of your application.

### 4. **Auto-tuning with LLVM**
For specific kernels or algorithms, consider using auto-tuning frameworks that leverage LLVM to find the best-performing code version by iteratively compiling and testing different versions of the code with various LLVM optimization flags or parameters.

### 5. **Using libNVVM**
`libNVVM` is a library that provides an NVVM IR (based on LLVM IR) compilation path. You can use it to generate optimized PTX code from CUDA NVVM IR.
- This approach involves more detailed manipulation of IR and is used internally by tools like Numba.

### 6. **Kernel Fusion and Other Optimizations**
Consider higher-level optimizations like kernel fusion, which can be applied manually or using tools that support such transformations. These optimizations often reduce the overhead of launching multiple kernels and can significantly improve performance.

### Further Integration with LLVM
- If your project has complex needs, you might explore deeper integrations with LLVM, possibly writing custom LLVM passes that are specifically tailored to optimize your CUDA kernels based on observed computational patterns or specific algorithmic needs.

### Debugging and Validation
After applying optimizations, ensure thorough testing and validation to confirm that no optimizations have altered the correctness of your program. Use unit tests, integration tests, and validation against known results to ensure reliability.

This approach combines leveraging existing tools and potentially developing new tools or scripts to manage the optimization process efficiently.
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
