#pragma once
#ifndef OPINF_CUDAMATRIX_H
#define OPINF_CUDAMATRIX_H

#include <memory>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdexcept>
#include <exception>
#include <tuple>

#include "CudaGpuMatrix.h"
#include "CudaVector.h"
#include "CudaLibraries.h"
#include "utilities.h"



class cuda_host_matrix
{
	using const_proxy_t = matrix_bracket_proxy<const cuda_host_matrix, const double>;
	using matrix_proxy_t = matrix_bracket_proxy<cuda_host_matrix, double>;
	friend class cuda_gpu_matrix;

public:
	enum MatrixType
	{
		CM_SPARSE,
		CM_DENSE
	};

	cuda_host_matrix(int m, int n, MatrixType matrixType = CM_DENSE) : M(m), N(n), matrixType(matrixType)
	{
		// currently matrixType doesn't do anything, but can be used later to inform the relevant storage choices
		// CM_DENSE matrix storage
		long sz = m * n;
		this->data = std::shared_ptr<double>(new double[sz], std::default_delete<double[]>());
		memset(data.get(), 0, sizeof(double) * sz);
	}

	cuda_host_matrix(int m, int n, MatrixType matrixType, double* data) : cuda_host_matrix(m, n, matrixType)
	{
		this->data = std::shared_ptr<double>(data);
	}

	void copyToGpuMemory(cuda_gpu_matrix& gpuMatrix) const
	{
		checkCudaOperationStatus<cuda_memory_error>(
			cublasSetMatrix(this->M, this->N, sizeof(double), (this->data).get(), this->M, gpuMatrix.gpuPtr.get(), this->M)
		);
	}

	void copyFromGpuMemory(const cuda_gpu_matrix& gpuMatrix)
	{
		checkCudaOperationStatus<cuda_memory_error>(
			cublasGetMatrix(this->M, this->N, sizeof(double), gpuMatrix.gpuPtr.get(), this->M, (this->data).get(), this->M)
		);
	}

	void print()
	{
		print_mat(data.get(), M, N);
	}

	// Basic operators
	double& operator()(int row, int col)
	{
		// column major order  is key //
		return (this->data).get()[columnMajorZeroIndex(row, col, this->M, this->N)];
	}

	matrix_proxy_t operator[](size_t row)
	{
		return matrix_proxy_t(*this, row);
	}

	std::pair<size_t, size_t> getMatrixSize() const
	{
		return std::make_pair(this->M, this->N);
	}
	std::shared_ptr<double> data;

private:
	MatrixType matrixType;

	size_t M;
	size_t N;
};

#endif OPINF_CUDAMATRIX_H

