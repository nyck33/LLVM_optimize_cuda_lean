```bash
#output LLVM IR
clang++ -O3 --cuda-gpu-arch=sm_75 --cuda-path=/usr/local/cuda -S -emit-llvm -x cuda --cuda-device-only cudaComputePortfolioRisk.cu -o cudaComputePortfolioRisk.ll

# optiimize LLVM IR
opt -O3 -S cudaComputePortfolioRisk.ll -o cudaComputePortfolioRisk_optimized.ll

# output ptx
llc -march=nvptx64 -mcpu=sm_75 -filetype=asm cudaComputePortfolioRisk_optimized.ll -o cudaComputePortfolioRisk.ptx
```