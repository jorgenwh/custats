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
    int *observed_counts, std::vector<int> &observed_counts_shape, 
    float *counts, std::vector<int> &counts_shape,
    float base_lambda, float error_rate, 
    float *out, std::vector<int> &out_shape);

#endif // FUNCTIONS_H_
