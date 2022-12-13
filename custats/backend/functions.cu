#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"
#include "functions.h"

void poisson_logpmf_hh2h(const int *k, const double *r, double *out, const int size)
{
  int *k_d;
  double *r_d, *out_d;
  cuda_errchk(cudaMalloc(&k_d, size*sizeof(int)));
  cuda_errchk(cudaMalloc(&r_d, size*sizeof(double)));
  cuda_errchk(cudaMalloc(&out_d, size*sizeof(double)));
  cuda_errchk(cudaMemcpy(k_d, k, size*sizeof(int), cudaMemcpyHostToDevice));
  cuda_errchk(cudaMemcpy(r_d, r, size*sizeof(double), cudaMemcpyHostToDevice));

  kernels::call_poisson_logpmf_kernel(k_d, r_d, out_d, size);

  cuda_errchk(cudaMemcpy(out, out_d, size*sizeof(double), cudaMemcpyDeviceToHost));
  cuda_errchk(cudaFree(k_d));
  cuda_errchk(cudaFree(r_d));
  cuda_errchk(cudaFree(out_d));
}

void poisson_logpmf_dh2d(const int *k, const double *r, double *out, const int size)
{
  double *r_d;
  cuda_errchk(cudaMalloc(&r_d, size*sizeof(double)));
  cuda_errchk(cudaMemcpy(r_d, r, size*sizeof(double), cudaMemcpyHostToDevice));

  kernels::call_poisson_logpmf_kernel(k, r_d, out, size);

  cuda_errchk(cudaFree(r_d));
}

void poisson_logpmf_hd2d(const int *k, const double *r, double *out, const int size)
{
  int *k_d;
  cuda_errchk(cudaMalloc(&k_d, size*sizeof(int)));
  cuda_errchk(cudaMemcpy(k_d, k, size*sizeof(int), cudaMemcpyHostToDevice));

  kernels::call_poisson_logpmf_kernel(k_d, r, out, size);

  cuda_errchk(cudaFree(k_d));
}

void poisson_logpmf_dd2d(const int *k, const double *r, double *out, const int size)
{
  kernels::call_poisson_logpmf_kernel(k, r, out, size);
}

void poisson_logpmf_experimental(
    unsigned int *observed_counts, float *counts, unsigned int n_counts,
    float base_lambda, float error_rate, float *out)
{
  kernels::call_poisson_logpmf_experimental_kernel(
      observed_counts, counts, n_counts, base_lambda, error_rate, out);
}

