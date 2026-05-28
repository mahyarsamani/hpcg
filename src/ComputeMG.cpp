
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"

#ifdef ANNOTATE
#include <annotate.h>
#endif

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

#if defined(ANNOTATE) && defined(ROI_MG)
    roi_begin_();
#ifdef SYNC_ON_ROI
    annotate_synchronize_(1);
#endif
#endif

  A.isMgOptimized = false;
  int ret = ComputeMG_ref(A, r, x);

#if defined(ANNOTATE) && defined(ROI_MG)
    roi_end_();
#ifdef SYNC_ON_ROI
    annotate_synchronize_(2);
#endif
#endif

  return ret;
}
