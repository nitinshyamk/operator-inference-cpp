#pragma once
#ifndef OPINF_OPERATORINFERENCE_H
#define OPINF_OPERATORINFERENCE_H

#include <deque>
#include <tuple>
#include "CudaLibraries.h"
#include "CudaGpuMatrix.h"
#include "CudaMatrix.h"
#include "CudaGpuVector.h"
#include "Ddt.h"
#include "LinearAlgebra.h"

struct operators
{
	cuda_gpu_matrix A;
	cuda_gpu_matrix B;
	cuda_gpu_matrix C;
	cuda_gpu_matrix F;
	cuda_gpu_matrix H;
	cuda_gpu_matrix N;
};

struct model_form
{
	bool Linear;
	bool Quadratic;
	bool Bilinear;
	bool Input;
	bool Constant;
};

template<FDSchemeEnum fdScheme, bool shouldScaleData>
class OperatorInference
{

	OperatorInference(const linear_algebra& linalg) : linalg(linalg) {}

	/// <summary>
	/// Returns operators from operator inference procedure
	/// </summary>
	/// <param name="modelform"></param>
	/// <param name="X"></param>
	/// <param name="U"></param>
	/// <param name="Vr"></param>
	/// <param name="dt"></param>
	/// <param name="lambda"></param>
	/// <param name="scaleData"></param>
	/// <returns></returns>
	operators infer(
		model_form& modelform,
		const cuda_gpu_matrix& X,
		const cuda_gpu_matrix& U,
		const cuda_gpu_matrix& Vr,
		double dt,
		double lambda)
	{
		Ddt<fdScheme> ddt(dt);

		auto Xdot = ddt(X);
		auto Xdot_ind = ddt.getIndices(X);
		auto rhs = linalg.multiply(Xdot, /* transpose Xdot */ true, Vr, false);

		// Create data matrices with reduced model basis
		cuda_gpu_matrix D;
		size_t L_sz, Q_sz, B_sz, C_sz;
		std::tie(D, L_sz, Q_sz, B_sz, C_sz) = get_data_matrices(modelform, X, U, Vr, Xdot_ind);

		// Find (possibly scaled) tikhonov solution
		cuda_gpu_matrix tik_sol = find_scaled_tikhonov_solution(D);

		// Retrieve operators
		return extract_operators(modelform, tik_sol, L_sz, Q_sz, B_sz, C_sz);
	}

	cuda_gpu_matrix find_scaled_tikhonov_solution(const cuda_gpu_matrix& D)
	{
		if (shouldScaleData)
		{
			scaling = linalg.find_column_maxes(D);
			D = linalg.column_normalize(D, scaling);
		}

		cuda_gpu_matrix tik_sol = linalg.transpose(linalg.tikhonov(Dscl, rhs, lambda));

		if (shouldScaleData)
		{
			return linalg.column_normalize(tik_sol, scaling);
		}

		return tik_sol;
	}

	operators extract_operators(
		const model_form& model_form,
		const cuda_gpu_matrix& tikhonov_solution, 
		size_t L_sz, 
		size_t Q_sz, 
		size_t B_sz, 
		size_t C_sz)
	{
		// Extract operators
		operators ops;
		auto row_range = std::make_pair(0, tik_sol.M - 1);
		auto extract_operator = [&tik_sol, &row_range](size_t start_col, size_t end_col) -> cuda_gpu_matrix
		{
			return linalg.subset(tik_sol, row_range, std::make_pair(start_col, end_col));
		};

		// check whether extraction is even needed
		if (model_form.Linear)
		{
			ops.A = extract_operator(0, L_sz - 1);
		}
		size_t start_ind = Lz;
		if (model_form.Quadratic)
		{
			ops.F = extract_operator(start_ind, start_ind + Q_sz - 1);
			//ops.H
		}
		start_ind += Q_sz;
		if (model_form.Bilinear)
		{
			ops.N = extract_operator(start_ind, start_ind + B_sz - 1);
		}
		start_ind += B_sz;
		if (model_form.Input)
		{
			ops.B = extract_operator(start_ind, start_ind + U.N - 1);
		}
		start_ind += U.N;
		if (model_form.Constant)
		{
			ops.C = extract_operator(start_ind, start_ind);
		}
		return ops;
	}

	/// <summary>
	/// Returns the data matrices and relevant indices in the form
	///	(D, L_sz, Q_sz, B_sz, C_sz) where the prefixes in front of _sz refer to the corresponding model term
	///	for which the values should index into.
	/// </summary>
	/// <param name="modelform"></param>
	/// <param name="X"></param>
	/// <param name="U"></param>
	/// <param name="Vr"></param>
	/// <param name="indexRange"></param>
	/// <returns></returns>
	std::tuple<cuda_gpu_matrix, size_t, size_t, size_t, size_t> get_data_matrices(
		model_form& modelform,
		const cuda_gpu_matrix& X,
		const cuda_gpu_matrix& U,
		const cuda_gpu_matrix& Vr,
		std::pair<size_t, size_t> indexRange)
	{
		size_t r = Vr.N;
		size_t K = indexRange.second - indexRange.first + 1;

		auto Xsubset = linalg.subset(X, std::make_pair(0, X.M - 1), indexRange);
		auto XhatT = linalg.transpose(linalg.multiply(Vr, true, Xsubset, false));

		std::deque<cuda_gpu_matrix> data_matrices;
		size_t L_sz, Q_sz, C_sz;

		// Learn I nput term
		size_t I_sz = modelform.Input ? K : 0;
		if (modelform.Input)
		{
			data_matrices.push_front(linalg.subset(U, std::make_pair(0, U.M - 1), indexRange));
		}

		// Learn B ilinear term
		size_t B_sz = modelform.Bilinear ? U.M;
		if (modelform.Bilinear)
		{
			throw std::invalid_argument("Not implemented: unclear whether Matlab reference implementation is functional");
		}
	
		// Learn Q uadratic term
		size_t Q_sz = modelform.Quadratic ? r * (r + 1) / 2 : 0;
		if (modelform.Quadratic)
		{
			data_matrices.push_front(linalg.get_matrix_squared(XhatT));
		}
		
		// Learn L inear term
		size_t L_sz = modelform.Linear ? r : 0;
		if (modelform.Linear)
		{
			data_matrices.push_front(XhatT);
		}

		// Learn C onstant term
		size_t C_sz = modelform.Constant ? 1 : 0;
		if (modelform.Constant)
		{
			data_matrices.push_back(linalg.get_ones(K));
		}

		if (data_matrices.empty())
		{
			throw std::invalid_argument("No estimated terms in model");
		}

		auto D = combine_all(data_matrices);

		return std::make_tuple(D, L_sz, Q_sz, B_sz, C_sz);
	}

	cuda_gpu_matrix combine_all(std::deque<cuda_gpu_matrix> matrices)
	{
		throw std::invalid_argument("not implemented");
	}


private:
	const linear_algebra& linalg;
};

#endif OPINF_OPERATORINFERENCE_H
