#include "CudaGpuVector.h"

cuda_gpu_vector::cuda_gpu_vector(size_t n) : cuda_gpu_matrix(n, 1) {};

double& cuda_gpu_vector::operator[](size_t i)
{
	return (cuda_gpu_matrix::gpuPtr).get()[i];
};