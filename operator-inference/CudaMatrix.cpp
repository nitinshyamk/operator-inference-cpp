#include "CudaMatrix.h"


cuda_host_matrix::cuda_host_matrix(size_t m, size_t n, MatrixType matrixType) : M(m), N(n), matrixType(matrixType)
{
	// currently matrixType doesn't do anything, but can be used later to inform the relevant storage choices
	// CM_DENSE matrix storage
	size_t sz = m * n;
	this->data = std::shared_ptr<double>(new double[sz], std::default_delete<double[]>());
	memset(data.get(), 0, sizeof(double) * sz);
};

cuda_host_matrix::cuda_host_matrix(size_t m, size_t n, MatrixType matrixType, double* data) : cuda_host_matrix(m, n, matrixType)
{
	this->data = std::shared_ptr<double>(data);
};

void cuda_host_matrix::copyToGpuMemory(const cuda_gpu_matrix& gpuMatrix) const
{
	checkCudaStatus<cuda_memory_error>(
		cublasSetMatrix(
			this->M,
			this->N,
			sizeof(double),
			(this->data).get(),
			this->M,
			gpuMatrix.gpuPtr.get(),
			this->M));
};

void cuda_host_matrix::copyFromGpuMemory(const cuda_gpu_matrix& gpuMatrix)
{
	checkCudaStatus<cuda_memory_error>(
		cublasGetMatrix(
			this->M,
			this->N,
			sizeof(double),
			gpuMatrix.gpuPtr.get(),
			this->M,
			(this->data).get(),
			this->M));
};

void cuda_host_matrix::print()
{
	print_mat(data.get(), M, N);
};

// Basic operators
double& cuda_host_matrix::operator()(int row, int col)
{
	// column major order  is key //
	return (this->data).get()[columnMajorZeroIndex(row, col, this->M, this->N)];
};

cuda_host_matrix::matrix_proxy_t cuda_host_matrix::operator[](size_t row)
{
	return cuda_host_matrix::matrix_proxy_t(*this, row);
};

std::pair<size_t, size_t> cuda_host_matrix::getMatrixSize() const
{
	return std::make_pair(this->M, this->N);
};