
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

#ifdef ANNOTATE
extern "C" {
#include <annotate.h>
}
#endif

#ifdef HOV
#include "hov.h"
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include <cassert>
#endif

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

#if defined(ANNOTATE) && defined(KERNEL_SYMGS)
    roi_begin_();
#ifdef SYNC_ON_ROI
    annotate_synchronize_(1);
#endif
#endif

#ifdef HOV
  assert(x.localLength==A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

  assert(A.optimizationData != 0);
  HovData * hov_data = (HovData*)A.optimizationData;
  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;
  const double * const rv = r.values;
  double * const xv = x.values;

  for (local_int_t i=0; i< nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];
    hov_pair_t * pair = &hov_data->spmv_pairs[i]; // Reuse SpMV pairs

    for (int j=0; j< currentNumberOfNonzeros; j++) {
      hov_result_f64_u32_t res;
      hov_gather_f64_u32(&res, pair, j);
      sum -= currentValues[j] * res.data_val;
    }
    sum += xv[i]*currentDiagonal;
    xv[i] = sum/currentDiagonal;
  }

  for (local_int_t i=nrow-1; i>=0; i--) {
    const double * const currentValues = A.matrixValues[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];
    hov_pair_t * pair = &hov_data->spmv_pairs[i];

    for (int j = 0; j< currentNumberOfNonzeros; j++) {
      hov_result_f64_u32_t res;
      hov_gather_f64_u32(&res, pair, j);
      sum -= currentValues[j]*res.data_val;
    }
    sum += xv[i]*currentDiagonal;
    xv[i] = sum/currentDiagonal;
  }

  int ret = 0;
#else
  int ret = ComputeSYMGS_ref(A, r, x);
#endif

#if defined(ANNOTATE) && defined(KERNEL_SYMGS)
    roi_end_();
#ifdef SYNC_ON_ROI
    annotate_synchronize_(2);
#endif
#endif

  return ret;
}
