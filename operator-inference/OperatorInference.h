#pragma once
#ifndef OPINF_OPERATORINFERENCE_H
#define OPINF_OPERATORINFERENCE_H

/// <summary>
/// Core operator inference
/// </summary>
class OperatorInference
{

};

//int main() {
//	// load libraries
//	// get trajectories from somewhere? 
//	// X - N x K state matrix
//	// U - K x m input data matrix (over time)
//	// Vr - N x r reduced basis 
//
//	// additional params, compiler defined?
//	// 
//	// learning reduced model - whether 'LQR/I/CLB' etc
//	// 
//
//	//known parameters/sizes:
//	K, r (Vr),
//
//	// CUSTOM - GPU KERNEL
//	[Xdot, index_vec] = ddt(X, ...)
//	// index_vec is the set of derivatives for which we just computed the derivative
//
//	// LIKELY AS BLAS - SPARSE
//	rhs = transpose(Xdot) * Vr;
//
//	// now we get the approximated Xhat
//	// likely as BLAS/SPARSE
//	Xhat = transpose(Vr) * colsubset(X, index_vec);
//	
//	IF (learnQuadratic)
//	// NO BLAS/SPARSE op - custom build as GPU kernel
//		Xhatsq = get_x_squared(transpose(X))
//	ELSE (learnQuadratic)
//
//	IF (constant)
//		C = all ones of appropriate dimension
//
//	IF (learn bilinear)
//		XU = 
//
//	// now we get the Kroenecker product (like) matrix
//
//
//
//	// if we need I 
//
//
//
//	// exact set of operations in cuBlAS, cuSparse
//	//		
//}
#endif OPINF_OPERATORINFERENCE_H
