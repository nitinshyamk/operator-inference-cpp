#pragma once
#ifndef OPINF_LIBRARYLOADER_H
#define OPINF_LIBRARYLOADER_H

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>

#include "utilities.h"


class library_load_error : public std::runtime_error
{
public:
	library_load_error() : runtime_error("CUDA library load failure") {}

	virtual const char* what() const throw()
	{
		return "CUDA library load failure";
	}
};

/// <summary>
/// Responsible for loading required runtime libraries including:
///	 cuBLAS
///	 cuSPARSE
///  cuSOLVER
/// </summary>
class cuda_libraries
{
	using status = cublasStatus_t;
public:

	cuda_libraries()
	{
		checkCudaStatus<library_load_error>(cublasCreate_v2(&blas_handle));
		checkCudaStatus<library_load_error>(cublasLtCreate(&blaslt_handle));
		checkCudaStatus<library_load_error>(cusparseCreate(&sparse_handle));

		std::cout << "Loaded CUDA libraries successfully." << std::endl;
	}

	~cuda_libraries()
	{
		cublasDestroy_v2(blas_handle);
		cublasLtDestroy(blaslt_handle);
		cusparseDestroy(sparse_handle);
		std::cout << "Closed connections to CUDA libraries." << std::endl;
	}

	cuda_libraries(const cuda_libraries& cpy) = delete;

	const cublasHandle_t get_blas_handle() const
	{
		return blas_handle;
	}

	const cublasLtHandle_t get_blaslt_handle() const
	{
		return blaslt_handle;
	}

	const cusparseHandle_t get_sparse_handle() const
	{
		return sparse_handle;
	}

private:
	cublasHandle_t blas_handle;
	cublasLtHandle_t blaslt_handle;
	cusparseHandle_t sparse_handle;
};

#endif OPINF_LIBRARYLOADER_H

