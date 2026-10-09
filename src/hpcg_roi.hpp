// One region of interest (ROI) per run.
//
// Every Compute* kernel is called many times: the reference timing,
// CG_ref, TestCG (on a matrix whose diagonal is scaled by 1e6, first
// without preconditioning), the optimization CG and the timed CG sets.
// A marker placed unconditionally in a kernel would therefore emit many
// roi_begin_/roi_end_ pairs, the first of them in TestCG.
//
// Instead, main() arms the ROI right before the first timed CG call, and
// the kernel selected with KERNEL_* marks only its first call after that:
// on the real matrix, inside a preconditioned CG call. The begin disarms,
// so nested or later calls of the kernel do not mark again.
#ifndef HPCG_ROI_HPP
#define HPCG_ROI_HPP

#ifdef ANNOTATE
extern "C" {
#include <annotate.h>
}

// Set by main() before the first timed CG call (defined in main.cpp).
extern bool hpcg_roi_armed;

// Marks the ROI's begin if it is armed, and disarms it. Returns whether it
// marked, to be passed to HpcgRoiEnd.
inline bool HpcgRoiBegin() {
  if (!hpcg_roi_armed) return false;
  hpcg_roi_armed = false;
  roi_begin_();
#ifdef SYNC_ON_ROI
  annotate_synchronize_(1);
#endif
  return true;
}

// Marks the ROI's end if the matching HpcgRoiBegin marked its begin.
inline void HpcgRoiEnd(bool began) {
  if (!began) return;
  roi_end_();
#ifdef SYNC_ON_ROI
  annotate_synchronize_(2);
#endif
}
#endif // ANNOTATE

#endif // HPCG_ROI_HPP
