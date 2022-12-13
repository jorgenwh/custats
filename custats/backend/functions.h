#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"

void poisson_logpmf_hh2h(const int *k, const double *r, double *out, const int size);
void poisson_logpmf_dh2d(const int *k, const double *r, double *out, const int size);
void poisson_logpmf_hd2d(const int *k, const double *r, double *out, const int size);
void poisson_logpmf_dd2d(const int *k, const double *r, double *out, const int size);

void poisson_logpmf_experimental(
    unsigned int *observed_counts, float *counts, unsigned int n_counts,
    float base_lambda, float error_rate, float *out);

#endif // FUNCTIONS_H_
