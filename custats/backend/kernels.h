#ifndef KERNELS_H_
#define KERNELS_H_

#include <inttypes.h>
#include <cuda_runtime.h>

#include "common.h"

namespace kernels {

void call_poisson_logpmf_kernel(const int *k, const double *r, double *out, const int size);
void call_poisson_logpmf_experimental_kernel(const unsigned int *observed_counts, const float *counts, 
    const unsigned int n_counts, const float base_lambda, const float error_rate, float *out);

} // kernels

#endif // KERNELS_H_
