#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"
#include "functions.h"

void poisson_logpmf_hh2h(const int *k, const double *r, double *out, const int size)
{
  int *k_d;
  double *r_d, *out_d;
  cudaMalloc(&k_d, size*sizeof(int));
  cudaMalloc(&r_d, size*sizeof(double));
  cudaMalloc(&out_d, size*sizeof(double));
  cudaMemcpy(k_d, k, size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(r_d, r, size*sizeof(double), cudaMemcpyHostToDevice);

  kernels::call_poisson_logpmf_kernel(k_d, r_d, out_d, size);

  cudaMemcpy(out, out_d, size*sizeof(double), cudaMemcpyDeviceToHost);
}

void poisson_logpmf_dh2d(const int *k, const double *r, double *out, const int size)
{
  double *r_d;
  cudaMalloc(&r_d, size*sizeof(double));
  cudaMemcpy(r_d, r, size*sizeof(double), cudaMemcpyHostToDevice);

  kernels::call_poisson_logpmf_kernel(k, r_d, out, size);
}

void poisson_logpmf_hd2d(const int *k, const double *r, double *out, const int size)
{
  int *k_d;
  cudaMalloc(&k_d, size*sizeof(int));
  cudaMemcpy(k_d, k, size*sizeof(int), cudaMemcpyHostToDevice);

  kernels::call_poisson_logpmf_kernel(k_d, r, out, size);
}

void poisson_logpmf_dd2d(const int *k, const double *r, double *out, const int size)
{
  kernels::call_poisson_logpmf_kernel(k, r, out, size);
}
