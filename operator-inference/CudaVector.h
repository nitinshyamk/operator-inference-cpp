#pragma once
#ifndef OPINF_CUDAVECTOR_H
#define OPINF_CUDAVECTOR_H

#include <memory>
#include <cublasLt.h>

#include "CudaMatrix.h"
#include "CudaLibraries.h"

class cuda_vector
{
	cuda_vector(size_t n) : N(n), shouldTranspose(false)
	{
		this->data = std::shared_ptr<double>(new double[N], std::default_delete<double[]>());
		memset(data.get(), 0, sizeof(double) * N);
	}

	double& operator[](size_t i)
	{
		return (data.get())[i];
	}

	bool checkVectorSizesMatch(const cuda_vector& vecb) const
	{
		if (this->N != vecb.N)
		{
			throw std::invalid_argument("Vector dimension mismatch for addition");
		}
	}

private:
	size_t N;
	std::shared_ptr<double> data;
	bool shouldTranspose;

};

#endif OPINF_CUDAVECTOR_H

