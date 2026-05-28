
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

#ifdef ANNOTATE
#include <annotate.h>
#endif

#ifdef HOV
#include "hov.h"
#endif

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

#if defined(ANNOTATE) && defined(ROI_SPMVM)
    roi_begin_();
#ifdef SYNC_ON_ROI
    annotate_synchronize_(1);
#endif
#endif

#ifdef HOV
  assert(A.optimizationData != 0);
  HovData * hov_data = (HovData*)A.optimizationData;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const int cur_nnz = A.nonzerosInRow[i];
    hov_pair_t * pair = &hov_data->spmv_pairs[i];

    for (int j=0; j< cur_nnz; j++) {
      hov_result_f64_u32_t res;
      hov_gather_f64_u32(&res, pair, j);
      sum += cur_vals[j] * res.data_val;
    }
    yv[i] = sum;
  }
  A.isSpmvOptimized = true;
  int ret = 0;
#else
  A.isSpmvOptimized = false;
  int ret = ComputeSPMV_ref(A, x, y);
#endif

#if defined(ANNOTATE) && defined(ROI_SPMVM)
    roi_end_();
#ifdef SYNC_ON_ROI
    annotate_synchronize_(2);
#endif
#endif

  return ret;
}
