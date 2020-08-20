#pragma once
#ifndef OPINF_CUDAGPUMATRIX_H
#define OPINF_CUDAGPUMATRIX_H

#include <cuda_runtime_api.h>
#include <memory>
#include "CudaMatrix.h"
#include "utilities.h"

class cuda_gpu_matrix
{
public:
	cuda_gpu_matrix(size_t m, size_t n) : M(m), N(n), gpuPtr(allocate_on_device<double>(m * n * sizeof(double))) 
	{
		cudaMemset(gpuPtr.get(), 0, sizeof(double) * m * n);
	}

	const size_t M;
	const size_t N;
	std::shared_ptr<double> gpuPtr;
};

#endif OPINF_CUDAGPUMATRIX_H