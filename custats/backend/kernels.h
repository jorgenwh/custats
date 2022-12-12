#ifndef KERNELS_H_
#define KERNELS_H_

#include <inttypes.h>
#include <cuda_runtime.h>

#include "common.h"

namespace kernels {

void call_poisson_logpmf_kernel(const int *k, const double *r, double *out, const int size);
void call_poisson_logpmf_experimental_kernel(
    int *observed_counts, float *counts, float base_lambda, float error_rate, 
    float *out, int n_counts);

} // kernels

#endif // KERNELS_H_
