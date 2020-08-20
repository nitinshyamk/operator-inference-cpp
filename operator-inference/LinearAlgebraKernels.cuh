#ifndef OPINF_LINALG_KERNELS
#define OPINF_LINALG_KERNELS

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "utilities.h"
#include <float.h>

__global__ void transpose_kernel(double* src, double* dest, size_t src_M, size_t src_N);

__global__ void find_column_maxes_kernel(double* src, double* dest, size_t src_M, size_t src_N);

__global__ void column_normalize_kernel(double* matrix, double* scaling, size_t mat_M, size_t mat_N);

__global__ void set_ones_kernel(double* src, size_t N);

__device__ size_t lookup_ind(size_t* src, size_t sz, size_t lookup_ind);

__global__ void get_matrix_squared_kernel(double* matrix, size_t M, size_t N, double* matrix_squared, size_t* lookup);

#endif OPINF_LINALG_KERNELS