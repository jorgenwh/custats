#ifndef KERNELS_H_
#define KERNELS_H_

#include <inttypes.h>
#include <cuda_runtime.h>

#include "common.h"

namespace kernels {

void call_poisson_logpmf_kernel(const int *k, const double *r, double *out, const int size);

} // kernels

#endif // KERNELS_H_
