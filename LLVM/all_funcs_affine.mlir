module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @_Z29__device_stub__matrixMultiplyPdS_S_iii(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg5 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = gpu.block_id  y
    %3 = arith.index_cast %2 : index to i32
    %4 = gpu.block_dim  y
    %5 = arith.index_cast %4 : index to i32
    %6 = arith.muli %3, %5 : i32
    %7 = gpu.thread_id  y
    %8 = arith.index_cast %7 : index to i32
    %9 = arith.addi %6, %8 : i32
    %10 = arith.muli %9, %arg4 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = arith.muli %9, %arg5 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = gpu.block_id  x
    %15 = arith.index_cast %14 : index to i32
    %16 = gpu.block_dim  x
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.muli %15, %17 : i32
    %19 = gpu.thread_id  x
    %20 = arith.index_cast %19 : index to i32
    %21 = arith.addi %18, %20 : i32
    %22 = arith.index_cast %21 : i32 to index
    %23 = arith.cmpi slt, %9, %arg3 : i32
    %24 = arith.cmpi slt, %21, %arg5 : i32
    %25 = arith.andi %23, %24 : i1
    scf.if %25 {
      %26 = affine.for %arg6 = 0 to %1 iter_args(%arg7 = %cst) -> (f64) {
        %27 = affine.load %arg0[%arg6 + symbol(%11)] : memref<?xf64>
        %28 = affine.load %arg1[%arg6 * symbol(%0) + symbol(%22)] : memref<?xf64>
        %29 = arith.mulf %27, %28 : f64
        %30 = arith.addf %arg7, %29 : f64
        affine.yield %30 : f64
      }
      affine.store %26, %arg2[symbol(%13) + symbol(%22)] : memref<?xf64>
    }
    return
  }
  func.func @computeCovarianceMatrix(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c8_i64 = arith.constant 8 : i64
    %c2_i32 = arith.constant 2 : i32
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() : memref<1xmemref<?xf64>>
    %alloca_0 = memref.alloca() : memref<1xmemref<?xf64>>
    %alloca_1 = memref.alloca() : memref<1xmemref<?xf64>>
    %alloca_2 = memref.alloca() : memref<1xmemref<?xf64>>
    %cast = memref.cast %alloca_2 : memref<1xmemref<?xf64>> to memref<?xmemref<?xf64>>
    %0 = arith.muli %arg3, %arg4 : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c8_i64 : i64
    %3 = call @_ZL17cudaMallocManagedIdE9cudaErrorPPT_mj(%cast, %2, %c1_i32) : (memref<?xmemref<?xf64>>, i64, i32) -> i32
    %cast_3 = memref.cast %alloca_1 : memref<1xmemref<?xf64>> to memref<?xmemref<?xf64>>
    %4 = arith.muli %arg4, %arg4 : i32
    %5 = arith.extsi %4 : i32 to i64
    %6 = arith.muli %5, %c8_i64 : i64
    %7 = call @_ZL17cudaMallocManagedIdE9cudaErrorPPT_mj(%cast_3, %6, %c1_i32) : (memref<?xmemref<?xf64>>, i64, i32) -> i32
    %cast_4 = memref.cast %alloca_0 : memref<1xmemref<?xf64>> to memref<?xmemref<?xf64>>
    %8 = call @_ZL17cudaMallocManagedIdE9cudaErrorPPT_mj(%cast_4, %2, %c1_i32) : (memref<?xmemref<?xf64>>, i64, i32) -> i32
    %cast_5 = memref.cast %alloca : memref<1xmemref<?xf64>> to memref<?xmemref<?xf64>>
    %9 = arith.muli %arg3, %arg3 : i32
    %10 = arith.extsi %9 : i32 to i64
    %11 = arith.muli %10, %c8_i64 : i64
    %12 = call @_ZL17cudaMallocManagedIdE9cudaErrorPPT_mj(%cast_5, %11, %c1_i32) : (memref<?xmemref<?xf64>>, i64, i32) -> i32
    %13 = affine.load %alloca_2[0] : memref<1xmemref<?xf64>>
    %14 = "polygeist.memref2pointer"(%13) : (memref<?xf64>) -> !llvm.ptr
    %15 = "polygeist.pointer2memref"(%14) : (!llvm.ptr) -> memref<?xi8>
    %16 = "polygeist.memref2pointer"(%arg0) : (memref<?xf64>) -> !llvm.ptr
    %17 = "polygeist.pointer2memref"(%16) : (!llvm.ptr) -> memref<?xi8>
    %18 = call @cudaMemcpy(%15, %17, %2, %c1_i32) : (memref<?xi8>, memref<?xi8>, i64, i32) -> i32
    %19 = affine.load %alloca_1[0] : memref<1xmemref<?xf64>>
    %20 = "polygeist.memref2pointer"(%19) : (memref<?xf64>) -> !llvm.ptr
    %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
    %22 = "polygeist.memref2pointer"(%arg1) : (memref<?xf64>) -> !llvm.ptr
    %23 = "polygeist.pointer2memref"(%22) : (!llvm.ptr) -> memref<?xi8>
    %24 = call @cudaMemcpy(%21, %23, %6, %c1_i32) : (memref<?xi8>, memref<?xi8>, i64, i32) -> i32
    %25 = arith.addi %arg4, %c15_i32 : i32
    %26 = arith.divsi %25, %c16_i32 : i32
    %27 = arith.addi %arg3, %c15_i32 : i32
    %28 = arith.divsi %27, %c16_i32 : i32
    %29 = affine.load %alloca_2[0] : memref<1xmemref<?xf64>>
    %30 = affine.load %alloca_1[0] : memref<1xmemref<?xf64>>
    %31 = affine.load %alloca_0[0] : memref<1xmemref<?xf64>>
    %32 = arith.index_cast %26 : i32 to index
    %33 = arith.index_cast %28 : i32 to index
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %32, %arg12 = %33, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c16, %arg15 = %c16, %arg16 = %c1) {
      func.call @_Z29__device_stub__matrixMultiplyPdS_S_iii(%29, %30, %31, %arg3, %arg4, %arg4) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32) -> ()
      gpu.terminator
    }
    %34 = affine.load %alloca_0[0] : memref<1xmemref<?xf64>>
    %35 = affine.load %alloca_2[0] : memref<1xmemref<?xf64>>
    %36 = affine.load %alloca[0] : memref<1xmemref<?xf64>>
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %33, %arg12 = %33, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c16, %arg15 = %c16, %arg16 = %c1) {
      func.call @_Z29__device_stub__matrixMultiplyPdS_S_iii(%34, %35, %36, %arg3, %arg4, %arg3) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32) -> ()
      gpu.terminator
    }
    %37 = call @cudaDeviceSynchronize() : () -> i32
    %38 = "polygeist.memref2pointer"(%arg2) : (memref<?xf64>) -> !llvm.ptr
    %39 = "polygeist.pointer2memref"(%38) : (!llvm.ptr) -> memref<?xi8>
    %40 = affine.load %alloca[0] : memref<1xmemref<?xf64>>
    %41 = "polygeist.memref2pointer"(%40) : (memref<?xf64>) -> !llvm.ptr
    %42 = "polygeist.pointer2memref"(%41) : (!llvm.ptr) -> memref<?xi8>
    %43 = call @cudaMemcpy(%39, %42, %11, %c2_i32) : (memref<?xi8>, memref<?xi8>, i64, i32) -> i32
    %44 = affine.load %alloca_2[0] : memref<1xmemref<?xf64>>
    %45 = "polygeist.memref2pointer"(%44) : (memref<?xf64>) -> !llvm.ptr
    %46 = "polygeist.pointer2memref"(%45) : (!llvm.ptr) -> memref<?xi8>
    %47 = call @cudaFree(%46) : (memref<?xi8>) -> i32
    %48 = affine.load %alloca_1[0] : memref<1xmemref<?xf64>>
    %49 = "polygeist.memref2pointer"(%48) : (memref<?xf64>) -> !llvm.ptr
    %50 = "polygeist.pointer2memref"(%49) : (!llvm.ptr) -> memref<?xi8>
    %51 = call @cudaFree(%50) : (memref<?xi8>) -> i32
    %52 = affine.load %alloca_0[0] : memref<1xmemref<?xf64>>
    %53 = "polygeist.memref2pointer"(%52) : (memref<?xf64>) -> !llvm.ptr
    %54 = "polygeist.pointer2memref"(%53) : (!llvm.ptr) -> memref<?xi8>
    %55 = call @cudaFree(%54) : (memref<?xi8>) -> i32
    %56 = affine.load %alloca[0] : memref<1xmemref<?xf64>>
    %57 = "polygeist.memref2pointer"(%56) : (memref<?xf64>) -> !llvm.ptr
    %58 = "polygeist.pointer2memref"(%57) : (!llvm.ptr) -> memref<?xi8>
    %59 = call @cudaFree(%58) : (memref<?xi8>) -> i32
    return
  }
  func.func private @_ZL17cudaMallocManagedIdE9cudaErrorPPT_mj(%arg0: memref<?xmemref<?xf64>>, %arg1: i64, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xmemref<?xf64>>) -> !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
    %2 = call @cudaMallocManaged(%1, %arg1, %arg2) : (memref<?xmemref<?xi8>>, i64, i32) -> i32
    return %2 : i32
  }
  func.func private @cudaMemcpy(memref<?xi8>, memref<?xi8>, i64, i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @cudaDeviceSynchronize() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @cudaFree(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @cudaMallocManaged(memref<?xmemref<?xi8>>, i64, i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}
