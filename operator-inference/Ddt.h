#pragma once
#ifndef OPINF_DDTOP_H
#define OPINF_DDTOP_H

#include <exception>

#include "CudaGpuMatrix.h"
#include "utilities.h"

// Set of supported finite difference schemes 
enum FDSchemeEnum
{

	/// <summary>
	/// Differencing mechanism for a discrete dynamical system (simply take 
	/// </summary>
	Discrete,

	/// <summary>
	/// Forward differencing (forward Euler integration)
	/// </summary>
	ForwardEuler,

	/// <summary>
	/// Backward differencing (backward Euler integration)
	/// </summary>
	BackwardEuler,

	/// <summary>
	/// 2nd order central differencing
	/// </summary>
	CentralDifference2,

	/// <summary>
	/// 2nd order forward differencing (explicit)
	/// </summary>
	ForwardDifference2,

	/// <summary>
	/// 2nd order backward differencing (implicit)
	/// </summary>
	BackwardDifference2,

	/// <summary>
	/// 4th order central differencing (current implementation 5 point stencil https://en.wikipedia.org/wiki/Five-point_stencil)
	/// </summary>
	CentralDifference4,

	/// <summary>
	/// 4th order backward differencing
	/// </summary>
	BackwardDifference4,

	/// <summary>
	/// 4th order forward differencing
	/// </summary>
	ForwardDifference4
};

class ddt_kernel_error : public std::runtime_error
{
public:
	ddt_kernel_error() : runtime_error("Discrete difference (temporal) ddt operation failured during kernel invocation or synchronization") {}

	virtual const char* what() const throw()
	{
		return "Discrete difference (temporal) ddt operation failured during kernel invocation or synchronization";
	}
};

template <FDSchemeEnum FDScheme>
__global__ void ddtKernel(
	double dt,
	size_t M,
	size_t N,
	const double* gpuMatrixSource,
	size_t startCol,
	size_t endCol,
	double* ddtResult)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;

	auto index = [M, N](size_t r, size_t c) { return columnMajorZeroIndex(r, c, M, N); };

	if (r < M && c + startCol <= endCol)
	{
		int ansind = index(r, c);
		int ind = index(r, c + startCol);
		int indP1 = index(r, c + startCol + 1);
		int indM1 = index(r, c + startCol - 1);
		int indP2 = index(r, c + startCol + 2);
		int indM2 = index(r, c + startCol - 2);
		int indP3 = index(r, c + startCol + 3);
		int indP4 = index(r, c + startCol + 4);
		int indM3 = index(r, c + startCol - 3);
		int indM4 = index(r, c + startCol - 4);

		const double* src = gpuMatrixSource;

		switch (FDScheme) {
		case Discrete:
			ddtResult[ansind] = src[indP1] / dt;
			return;
		case ForwardEuler:
			ddtResult[ansind] = (src[indP1] - src[ind]) / dt;
			return;
		case BackwardEuler:
			ddtResult[ansind] = (src[ind] - src[indM1]) / dt;
			return;
		case CentralDifference2:
			ddtResult[ansind] = (src[indP1] - src[indM1]) / (2 * dt);
			return;
		case ForwardDifference2:
			ddtResult[ansind] = (-3 * src[ind] + 4 * src[indP1] - src[indP2]) / (2 * dt);
			return;
		case BackwardDifference2:
			ddtResult[ansind] = (3 * src[ind] - 4 * src[indM1] + src[indM2]) / (2 * dt);
			return;
		case CentralDifference4:
			ddtResult[ansind] = (src[indM2] / 12.0 - 2.0 * src[indM2] / 3.0 + 2.0 * src[indP1] / 3.0 - src[indP2] / 12) / dt;
			return;
		case BackwardDifference4:
			ddtResult[ansind] = (25.0 * src[ind] / 12.0 - 4 * src[indM1] + 3 * src[indM2] - 4.0 * src[indM3] / 3.0 + src[indM4] / 4.0) / dt;
			return;
		case ForwardDifference4:
			ddtResult[ansind] = (-25.0 * src[ind] / 12.0 + 4 * src[indP1] - 3 * src[indP2] + 4.0 * src[indP3] / 3.0 - src[indP4] / 4.0) / dt;
			return;
		default:
			// cannot enter a default case with an unsupported enum scheme
			//assert(false);
			return;
		}
	}
};

template <FDSchemeEnum FDScheme>
class Ddt
{
public:
	// this works for now, but eventually needs to be accessible from GPU memory;
	Ddt(double dt) : dt(dt) {}

	/// <summary>
	/// Gets the index range that ddt will compute derivatives (finite differences) for, using 0-indexing.
	/// </summary>
	/// <param name="gpuMatrixSource">assumes that gpuMatrixSource has dimensions N \times K</param>
	/// <returns></returns>
	std::pair<size_t, size_t> getIndices(const cuda_gpu_matrix& gpuMatrixSource)
	{
		auto K = gpuMatrixSource.N;
		switch (FDScheme)
		{
		case Discrete:
			return std::make_pair(0, K - 2);
		case ForwardEuler:
			return std::make_pair(0, K - 2);
		case BackwardEuler:
			return std::make_pair(1, K - 1);
		case CentralDifference2:
			return std::make_pair(1, K - 2);
		case ForwardDifference2:
			return std::make_pair(0, K - 3);
		case BackwardDifference2:
			return std::make_pair(2, K - 1);
		case CentralDifference4:
			return std::make_pair(2, K - 3);
		case BackwardDifference4:
			return std::make_pair(4, K - 1);
		case ForwardDifference4:
			return std::make_pair(0, K - 5);
		default:
			throw std::invalid_argument("Unsupported finite difference scheme (template parameter FDScheme)");
			break;
		}
	}


	/// <summary>
	/// Performs specified ddt operation on a matrix in GPU memory
	/// </summary>
	/// <param name="gpuMatrix"> Assumed to be in format N x K where K is the time dimension </param>
	cuda_gpu_matrix operator()(const cuda_gpu_matrix& gpuMatrixSource)
	{
		size_t startind, endind, M;

		M = gpuMatrixSource.M;
		std::tie(startind, endind) = this->getIndices(gpuMatrixSource);
		cuda_gpu_matrix ans(M, endind - startind + 1);
		// invoke kernel on matrices
		
		size_t block1Dim = 1 << 5;
		dim3 gridDim((M + block1Dim - 1)/ block1Dim, (endind - startind + block1Dim)/ block1Dim);
		dim3 blockDim(block1Dim, block1Dim);

		ddtKernel<FDScheme><<<gridDim, blockDim>>>(
			this->dt,
			gpuMatrixSource.M,
			gpuMatrixSource.N,
			gpuMatrixSource.gpuPtr.get(),
			startind,
			endind,
			ans.gpuPtr.get());

		checkCudaError<ddt_kernel_error>(cudaGetLastError());
		checkCudaError<ddt_kernel_error>(cudaDeviceSynchronize());

		return ans;
	}

private:
	double dt;
};

#endif OPINF_DDTOP_H

