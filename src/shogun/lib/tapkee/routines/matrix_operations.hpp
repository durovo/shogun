/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 *
 */

#ifndef TAPKEE_MATRIX_OPS_H_
#define TAPKEE_MATRIX_OPS_H_

#include <tapkee_defines.hpp>

namespace tapkee
{
namespace tapkee_internal
{

//! Matrix-matrix operation used to 
//! compute smallest eigenvalues and 
//! associated eigenvectors of a sparse matrix
//! Essentially solves linear system 
//! with provided right-hand side part.
//!
struct SparseInverseMatrixOperation
{
	SparseInverseMatrixOperation(const SparseWeightMatrix& matrix) : solver()
	{
		solver.compute(matrix);
	}
	/** Solves linear system with provided right-hand size
	 */
	inline DenseMatrix operator()(const DenseMatrix& operatee)
	{
		return solver.solve(operatee);
	}
	SparseSolver solver;
	static const char* ARPACK_CODE;
	static const bool largest;
};
const char* SparseInverseMatrixOperation::ARPACK_CODE = "SM";
const bool SparseInverseMatrixOperation::largest = false;

//! Matrix-matrix operation used to 
//! compute smallest eigenvalues and 
//! associated eigenvectors of a dense matrix
//! Essentially solves linear system 
//! with provided right-hand side part.
//!
struct DenseInverseMatrixOperation
{
	DenseInverseMatrixOperation(const DenseMatrix& matrix) : solver()
	{
		solver.compute(matrix);
	}
	/** Solves linear system with provided right-hand size
	 */
	inline DenseMatrix operator()(const DenseMatrix& operatee)
	{
		return solver.solve(operatee);
	}
	DenseSolver solver;
	static const char* ARPACK_CODE;
	static const bool largest;
};
const char* DenseInverseMatrixOperation::ARPACK_CODE = "SM";
const bool DenseInverseMatrixOperation::largest = false;

//! Matrix-matrix operation used to
//! compute largest eigenvalues and
//! associated eigenvectors. Essentially
//! computes matrix product with 
//! provided right-hand side part.
//!
struct DenseMatrixOperation
{
	DenseMatrixOperation(const DenseMatrix& matrix) : _matrix(matrix)
	{
	}
	//! Computes matrix product of the matrix and provided right-hand 
	//! side matrix
	//! 
	//! @param rhs right-hand size matrix
	//!
	inline DenseMatrix operator()(const DenseMatrix& rhs)
	{
		return _matrix.selfadjointView<Eigen::Upper>()*rhs;
	}
	const DenseMatrix& _matrix;
	static const char* ARPACK_CODE;
	static const bool largest;
};
const char* DenseMatrixOperation::ARPACK_CODE = "LM";
const bool DenseMatrixOperation::largest = true;

//! Matrix-matrix operation used to
//! compute largest eigenvalues and
//! associated eigenvectors of X*X^T like
//! matrix implicitly. Essentially
//! computes matrix product with provided
//! right-hand side part *twice*.
//!
struct DenseImplicitSquareMatrixOperation
{
	DenseImplicitSquareMatrixOperation(const DenseMatrix& matrix) : _matrix(matrix)
	{
	}
	//! Computes matrix product of the matrix and provided right-hand 
	//! side matrix twice
	//! 
	//! @param rhs right-hand side matrix
	//!
	inline DenseMatrix operator()(const DenseMatrix& rhs)
	{
		return _matrix.selfadjointView<Eigen::Upper>()*(_matrix.selfadjointView<Eigen::Upper>()*rhs);
	}
	const DenseMatrix& _matrix;
	static const char* ARPACK_CODE;
	static const bool largest;
};
const char* DenseImplicitSquareMatrixOperation::ARPACK_CODE = "LM";
const bool DenseImplicitSquareMatrixOperation::largest = true;

#ifdef TAPKEE_GPU
struct GPUDenseImplicitSquareMatrixOperation
{
	GPUDenseImplicitSquareMatrixOperation(const DenseMatrix& matrix)
	{
		timed_context c("Storing matrices");
		mat = viennacl::matrix<ScalarType>(matrix.cols(),matrix.rows());
		vec = viennacl::matrix<ScalarType>(matrix.cols(),1);
		res = viennacl::matrix<ScalarType>(matrix.cols(),1);
		viennacl::copy(matrix,mat);
	}
	//! Computes matrix product of the matrix and provided right-hand 
	//! side matrix twice
	//! 
	//! @param rhs right-hand side matrix
	//!
	inline DenseMatrix operator()(const DenseMatrix& rhs)
	{
		timed_context c("Computing product");
		viennacl::copy(rhs,vec);
		res = viennacl::linalg::prod(mat, vec);
		vec = res;
		res = viennacl::linalg::prod(mat, vec);
		DenseMatrix result(rhs);
		viennacl::copy(res,result);
		return result;
	}
	viennacl::matrix<ScalarType> mat;
	viennacl::matrix<ScalarType> vec;
	viennacl::matrix<ScalarType> res;
	static const char* ARPACK_CODE;
	static bool largest;
};
const char* GPUDenseImplicitSquareMatrixOperation::ARPACK_CODE = "LM";
const bool GPUDenseImplicitSquareMatrixOperation::largest = true;

struct GPUDenseMatrixOperation
{
	GPUDenseMatrixOperation(const DenseMatrix& matrix)
	{
		mat = viennacl::matrix<ScalarType>(matrix.cols(),matrix.rows());
		vec = viennacl::matrix<ScalarType>(matrix.cols(),1);
		res = viennacl::matrix<ScalarType>(matrix.cols(),1);
		viennacl::copy(matrix,mat);
	}
	//! Computes matrix product of the matrix and provided right-hand 
	//! side matrix twice
	//! 
	//! @param rhs right-hand side matrix
	//!
	inline DenseMatrix operator()(const DenseMatrix& rhs)
	{
		viennacl::copy(rhs,vec);
		res = viennacl::linalg::prod(mat, vec);
		DenseMatrix result(rhs);
		viennacl::copy(res,result);
		return result;
	}
	viennacl::matrix<ScalarType> mat;
	viennacl::matrix<ScalarType> vec;
	viennacl::matrix<ScalarType> res;
	static const char* ARPACK_CODE;
	static bool largest;
};
const char* GPUDenseMatrixOperation::ARPACK_CODE = "LM";
const bool GPUDenseMatrixOperation::largest = true;
#endif

}
}

#endif