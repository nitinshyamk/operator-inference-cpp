#pragma once
#ifndef OPINF_UTILITIES_H
#define OPINF_UTILITIES_H

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iostream>

template <typename exception_t>
bool checkCudaOperationStatus(cublasStatus_t code)
{
	if (code != CUBLAS_STATUS_SUCCESS)
		throw exception_t();
	return true;
};

template <typename exception_t>
bool checkCudaOperationStatus(cusparseStatus_t code)
{
	if (code != CUSPARSE_STATUS_SUCCESS)
		throw exception_t();
	return true;
};

template <typename exception_t>
bool checkCudaError(cudaError_t err)
{
	if (err != cudaSuccess)
		throw exception_t();
	return true;
}

inline __device__ __host__ size_t columnMajorZeroIndex(size_t row, size_t col, size_t M, size_t N)
{
	return col * N + row;
};

void print_mat(const double* arr, int M, int N)
{
	using std::cout;
	using std::endl;
	if (M * N > 1000)
	{
		cout << "Matrix is too large to print." << endl;
		return;
	}

	cout << "Printing matrix of dimensions " << M << " x " << N << endl;
	for (int r = 0; r < M; ++r)
	{
		for (int c = 0; c < N; ++c)
		{
			cout << arr[columnMajorZeroIndex(r, c, M, N)] << '\t';
		}
		cout << endl;
	}
}

template<typename Matrix, typename Result>
class matrix_bracket_proxy
{
public:
	matrix_bracket_proxy(Matrix& A, size_t row) : A(A), row(row) {}
	Result& operator[](size_t col) { return A(row, col); }

private:
	Matrix& A;
	size_t row;
};


class cuda_memory_error : public std::runtime_error
{
public:
	cuda_memory_error() : runtime_error("CUDA memory allocation or transfer failure") {}

	virtual const char* what() const throw()
	{
		return "CUDA memory allocation or transfer failure";
	}
};

#endif OPINF_UTILITIES_H