#pragma once
#ifndef OPINF_LINEARALGEBRA_H
#define OPINF_LINEARALGEBRA_H

#include <cublasLt.h>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "CudaLibraries.h"
#include "CudaGpuMatrix.h"

class incompatible_dimensions_error : public std::runtime_error
{

public:
	incompatible_dimensions_error(int Am, int An, int Bm, int Bn) : 
		runtime_error(helper_format(Am, An, Bm, Bn)) {}

	virtual const char* what() const throw()
	{
		return "CUDA library load failure";
	}

	std::string helper_format(int Am, int An, int Bm, int Bn)
	{
		std::string phrase = "Incompatible matrix dimensions error. Cannot multiply matrices of dimensions ";
		phrase = phrase + std::to_string(Am) + " x " + std::to_string(An) + " and " + std::to_string(Bm) + " x " + std::to_string(Bn);
		return phrase;
	}

};

class cublas_matrix_description
{

};

class linear_algebra
{
public:
	linear_algebra(const cuda_libraries& libraries) : cudalibraries(libraries) {}

	/// <summary>
	/// Computes and returns C where C = (A  or transpose(A) ) * (B or transpose(B))
	/// </summary>
	/// <param name="A"></param>
	/// <param name="transposeA"></param>
	/// <param name="B"></param>
	/// <param name="transposeB"></param>
	/// <param name="BColumnSubset"></param>
	/// <param name="ans"></param>
	void Multiply(
		const cuda_gpu_matrix& A,
		bool transposeA,
		const cuda_gpu_matrix& B,
		bool transposeB,
		std::pair<size_t, size_t> BColumnSubset,
		const cuda_gpu_matrix& ans)
	{
		int A_M = transposeA ? A.N : A.M;
		int A_N = transposeA ? A.M : A.N;
		int B_M = transposeB ? B.N : B.M;
		int B_N = transposeB ? B.M : B.N;

		if (A_N != B_M)
			throw incompatible_dimensions_error(A_M, A_N, B_M, B_N);

		cuda_gpu_matrix C(A_M, B_N);

		cublasLtMatmulDesc_t operation_description = NULL;
		cublasLtMatrixLayout_t Adescription, Bdescription, Cdescription;
		cublasLtMatmulPreference_t preference = NULL;
		cublasLtMatmulHeuristicResult_t heuristicResult = {};
		int returnedResults = 0;

		cublasLtMatmulAlgoGetHeuristic(
			cudalibraries.get_blaslt_handle(),
			operation_description,
			Adescription,
			Bdescription,
			Cdescription,
			Cdescription,
			preference,
			1,
			&heuristicResult,
			&returnedResults);

		cublasLtMatmul(
			cudalibraries.get_blaslt_handle(),
			operation_description,
			NULL, // alpha
			A.gpuPtr.get(),
			Adescription,
			B.gpuPtr.get(),
			Bdescription,
			NULL, //beta
			C.gpuPtr.get(),
			Cdescription,
			C.gpuPtr.get(),
			Cdescription,
			&heuristicResult.algo,
			NULL, //workspace ptr
			0, //workspace size
			0); // Cudastream
		//todo

	}

private:
	cuda_libraries cudalibraries;
};
#endif OPINF_LINEARALGEBRA_H
