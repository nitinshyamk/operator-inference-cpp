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
	cuda_gpu_matrix(size_t m, size_t n) : M(m), N(n)
	{
		this->allocateMemory(m, n);
	}

	const size_t M;
	const size_t N;(
	std::shared_ptr<double> gpuPtr;

private:

	void allocateMemory(size_t m, size_t n)
	{
		double* gpuPtrLocal;
		checkCudaError<cuda_memory_error>(
			cudaMalloc(reinterpret_cast<void**>(&gpuPtrLocal), m * n * sizeof(double))
		);
		gpuPtr = std::shared_ptr<double>(gpuPtrLocal, [](double* ptr) { cudaFree(ptr); });
	}

	friend class cuda_host_matrix;


};

#endif OPINF_CUDAGPUMATRIX_H