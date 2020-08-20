#include "LinearAlgebra.h"



cuda_gpu_matrix
linear_algebra::multiply(
	const cuda_gpu_matrix& A,
	bool transposeA,
	const cuda_gpu_matrix& B,
	bool transposeB) const
{
	int A_M = transposeA ? A.N : A.M;
	int A_N = transposeA ? A.M : A.N;
	int B_M = transposeB ? B.N : B.M;
	int B_N = transposeB ? B.M : B.N;

	if (A_N != B_M)
		throw incompatible_dimensions_error(A_M, A_N, B_M, B_N);

	cuda_gpu_matrix C(A_M, B_N);

	cublasLtMatmulDesc_t op_description = NULL;
	checkCudaStatus<cublas_matrix_operation_error>(
		cublasLtMatmulDescCreate(&op_description, CUBLAS_COMPUTE_64F, CUDA_R_64F));
	cublasOperation_t shouldTransposeValue = CUBLAS_OP_T;

	auto setTransposeAttribute = [&op_description, &shouldTransposeValue](cublasLtMatmulDescAttributes_t attr) -> void
	{
		checkCudaStatus<cublas_matrix_operation_error>(
			cublasLtMatmulDescSetAttribute(
				op_description,
				attr,
				&shouldTransposeValue,
				sizeof(shouldTransposeValue)));
	};

	if (transposeA)
		setTransposeAttribute(CUBLASLT_MATMUL_DESC_TRANSA);

	if (transposeB)
		setTransposeAttribute(CUBLASLT_MATMUL_DESC_TRANSB);

	// create descriptions
	cublasLtMatrixLayout_t Adescription, Bdescription, Cdescription;
	checkCudaStatus<cublas_matrix_operation_error>(
		cublasLtMatrixLayoutCreate(&Adescription, CUDA_R_64F, A_M, A_N, A_M));
	checkCudaStatus<cublas_matrix_operation_error>(
		cublasLtMatrixLayoutCreate(&Bdescription, CUDA_R_64F, B_M, B_N, B_M));
	checkCudaStatus<cublas_matrix_operation_error>(
		cublasLtMatrixLayoutCreate(&Cdescription, CUDA_R_64F, C.M, C.N, C.M));

	size_t workspace_size = 1 << 16;
	std::shared_ptr<void> workspace = allocate_on_device<void>(workspace_size);

	cublasLtMatmulPreference_t preference = NULL;
	checkCudaStatus<cublas_matrix_operation_error>(
		cublasLtMatmulPreferenceCreate(&preference));
	checkCudaStatus<cublas_matrix_operation_error>(
		cublasLtMatmulPreferenceSetAttribute(
			preference,
			CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
			&workspace_size,
			sizeof(workspace_size)));

	cublasLtMatmulHeuristicResult_t heuristicResult = {};
	int returnedResults = 0;
	cublasLtMatmulAlgoGetHeuristic(
		cudalibraries.get_blaslt_handle(),
		op_description,
		Adescription,
		Bdescription,
		Cdescription,
		Cdescription,
		preference,
		1,
		&heuristicResult,
		&returnedResults);

	double alpha = 1, beta = 0;
	cublasLtMatmul(
		cudalibraries.get_blaslt_handle(),
		op_description,
		&alpha,
		A.gpuPtr.get(),
		Adescription,
		B.gpuPtr.get(),
		Bdescription,
		&beta,
		// currently C must equal D, so we just pass it in twice
		C.gpuPtr.get(),
		Cdescription,
		C.gpuPtr.get(),
		Cdescription,
		&heuristicResult.algo,
		workspace.get(),
		workspace_size,
		0); // default cudastream

	checkCudaError<cublas_matrix_operation_error>(cudaDeviceSynchronize());
	return C;
};


cuda_gpu_matrix
linear_algebra::concatenate(
		const cuda_gpu_matrix& A,
		const cuda_gpu_matrix& B,
		bool shouldConcatenateVertically) const
{
	if (shouldConcatenateVertically && A.N != B.N || !shouldConcatenateVertically && A.M != B.M)
		throw std::invalid_argument("Invalid dimensions for specified concatenation operation");
	
	double* Asrc = A.gpuPtr.get();
	double* Bsrc = B.gpuPtr.get();
	if (!shouldConcatenateVertically)
	{
		cuda_gpu_matrix ans(A.M, A.N + B.N);
		checkCudaError<cuda_memory_error>(cudaMemcpy(
			ans.gpuPtr.get(),
			Asrc,
			A.M * A.N * sizeof((*Asrc)),
			cudaMemcpyDeviceToDevice));

		checkCudaError<cuda_memory_error>(cudaMemcpy(
			ans.gpuPtr.get() + A.M * A.N,
			Bsrc,
			B.M * B.N * sizeof((*Bsrc)),
			cudaMemcpyDeviceToDevice));
		return ans;
	}

	cuda_gpu_matrix ans(A.M + B.M, A.N);
	for (size_t col = 0; col < A.N; ++col)
	{
		checkCudaError<cuda_memory_error>(cudaMemcpy(
			ans.gpuPtr.get() + columnMajorZeroIndex(0, col, ans.M, ans.N),
			Asrc + columnMajorZeroIndex(0, col, A.M, A.N),
			A.M * sizeof((*Asrc)),
			cudaMemcpyDeviceToDevice));

		checkCudaError<cuda_memory_error>(cudaMemcpy(
			ans.gpuPtr.get() + columnMajorZeroIndex(A.M, col, ans.M, ans.N),
			Bsrc + columnMajorZeroIndex(0, col, B.M, B.N),
			B.M * sizeof((*Bsrc)),
			cudaMemcpyDeviceToDevice));
	}
	return ans;
}

svd
linear_algebra::svd_decomposition(const cuda_gpu_matrix& A) const
{
	return svd{ A, A, A };
}

cuda_gpu_matrix
linear_algebra::pinv(const svd& decomposition) const
{
	return decomposition.S;
}

cuda_gpu_matrix
linear_algebra::subset(
	const cuda_gpu_matrix& A,
	std::pair<size_t, size_t> rowrange,
	std::pair<size_t, size_t> colrange) const
{
	if (rowrange.second > A.M - 1 || rowrange.first < 0 || colrange.second > A.N - 1 || colrange.first < 0)
		throw std::invalid_argument("Invalid range parameters for taking a subset");
	if (rowrange.second < rowrange.first || colrange.second < colrange.first)
		throw std::invalid_argument("Invalid range arguments");

	size_t B_M = rowrange.second - rowrange.first + 1;
	size_t B_N = colrange.second - colrange.first + 1;
	
	cuda_gpu_matrix ans(B_M, B_N);

	double* src = A.gpuPtr.get(), *dest = ans.gpuPtr.get();


	// if the row range is the same, a single contiguous block
	// otherwise we need to copy in chunks

	if (B_M = A.M)
	{
		size_t start_dest = columnMajorZeroIndex(0, colrange.first, A.M, A.N);
		checkCudaError<cuda_memory_error>(
			cudaMemcpy(dest, src + start_dest, A.M * B_N * sizeof(*(src)), cudaMemcpyDeviceToDevice));
		return ans;
	}

	for (size_t start_col = 0; start_col < B_N; ++start_col)
	{
		double* start_dest = dest + columnMajorZeroIndex(rowrange.first, start_col + colrange.first, A.M, A.N);
		double* start_src = src + columnMajorZeroIndex(0, start_col, ans.M, ans.N);
		checkCudaError<cuda_memory_error>(
			cudaMemcpy(start_dest, start_src, ans.M * sizeof(*(src)), cudaMemcpyDeviceToDevice));
	}
	return ans;
}

cuda_gpu_matrix 
linear_algebra::transpose(const cuda_gpu_matrix& A, bool forceDeepCopy) const
{
	cuda_gpu_matrix ans(A.N, A.M);

	if (!forceDeepCopy && A.M == 1 || A.N == 1)
	{
		ans.gpuPtr = A.gpuPtr;
		return ans;
	}

	size_t block1Dim = 1 << 5;
	dim3 gridDim((A.M + block1Dim - 1) / block1Dim, (A.N + block1Dim - 1) / block1Dim);
	dim3 blockDim(block1Dim, block1Dim);

	transpose_kernel KERNEL_ARGS2(gridDim, blockDim) (A.gpuPtr.get(), ans.gpuPtr.get(), A.M, A.N);

	checkCudaError<cublas_matrix_operation_error>(cudaGetLastError());
	checkCudaError<cublas_matrix_operation_error>(cudaDeviceSynchronize());
	return ans;
}

cuda_gpu_matrix
linear_algebra::find_column_maxes(const cuda_gpu_matrix& A) const
{
	cuda_gpu_matrix scaling(1, A.N);
	size_t blockDim = 1 << 5;
	size_t gridDim = (A.N + blockDim - 1) / blockDim;

	find_column_maxes_kernel KERNEL_ARGS2(gridDim, blockDim) (A.gpuPtr.get(), scaling.gpuPtr.get(), A.M, A.N);

	checkCudaError<cublas_matrix_operation_error>(cudaGetLastError());
	checkCudaError<cublas_matrix_operation_error>(cudaDeviceSynchronize());
	return scaling;
}

void 
linear_algebra::column_normalize(cuda_gpu_matrix& A, const cuda_gpu_matrix& scaling) const
{
	if (A.N != scaling.N || scaling.M != 1)
		throw std::invalid_argument("incompatible dimensions for applying column normalization operations");

	column_normalize_kernel KERNEL_ARGS2(gridDim, blockDim) (A.gpuPtr.get(), scaling.gpuPtr.get(), A.M, A.N);
	checkCudaError<cublas_matrix_operation_error>(cudaGetLastError());
	checkCudaError<cublas_matrix_operation_error>(cudaDeviceSynchronize());
}

cuda_gpu_vector
linear_algebra::get_ones(size_t n) const
{
	cuda_gpu_vector ans(n);
	size_t blockDim = 1 << 5;
	size_t gridDim = (n + blockDim - 1) / blockDim;
	set_ones_kernel KERNEL_ARGS2(gridDim, blockDim) (ans.gpuPtr.get(), n);
	checkCudaError<cublas_matrix_operation_error>(cudaGetLastError());
	checkCudaError<cublas_matrix_operation_error>(cudaDeviceSynchronize());
	return ans;
}

cuda_gpu_matrix 
linear_algebra::get_matrix_squared(const cuda_gpu_matrix& A) const
{
	size_t width = A.N * (A.N + 1) / 2;
	cuda_gpu_matrix ans(A.M, width);

	std::shared_ptr<size_t> lookup_tbl(new size_t[A.N], std::default_delete<size_t[]>());
	for (size_t i = 1; i <= A.N; ++i)
	{
		lookup_tbl.get()[i - 1] = (i * i + i) / 2 - 1;
	}

	size_t block1Dim = 1 << 5;
	dim3 gridDim((A.M + block1Dim - 1) / block1Dim, (width + block1Dim - 1) / block1Dim);
	dim3 blockDim(block1Dim, block1Dim);

	get_matrix_squared_kernel KERNEL_ARGS2(gridDim, blockDim) (A.gpuPtr.get(), A.M, A.N, ans.gpuPtr.get(), lookup_tbl.get());

	checkCudaError<cublas_matrix_operation_error>(cudaGetLastError());
	checkCudaError<cublas_matrix_operation_error>(cudaDeviceSynchronize());

	return ans;
}

cuda_gpu_matrix
linear_algebra::tikhonov(const cuda_gpu_matrix& A, const cuda_gpu_matrix& b, double k) const
{
	cuda_gpu_matrix zeromat(A.N, b.N);

	cuda_host_matrix identity_host(A.N, A.N);
	for (size_t i = 0; i < A.N; ++i)
	{
		identity_host[i][i] = sqrt(k);
	}
	cuda_gpu_matrix identity(A.N, A.N);
	identity_host.copyToGpuMemory(identity);

	auto Aplus = concatenate(A, identity, true);
	auto bplus = concatenate(b, zeromat, true);

	auto pinv_Aplus = pinv(Aplus);
	auto solution = multiply(Aplus, false, bplus, false);
	return solution;
}

