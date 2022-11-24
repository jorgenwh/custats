#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"
#include "functions.h"

void poisson_logpmf_hh2h(const int *k, const float *r, float *out, const int size)
{
  int *k_d;
  float *r_d, *out_d;
  cuda_errchk(cudaMalloc(&k_d, size*sizeof(int)));
  cuda_errchk(cudaMalloc(&r_d, size*sizeof(float)));
  cuda_errchk(cudaMalloc(&out_d, size*sizeof(float)));
  cuda_errchk(cudaMemcpy(k_d, k, size*sizeof(int), cudaMemcpyHostToDevice));
  cuda_errchk(cudaMemcpy(r_d, r, size*sizeof(float), cudaMemcpyHostToDevice));

  kernels::call_poisson_logpmf_kernel(k_d, r_d, out_d, size);

  cuda_errchk(cudaMemcpy(out, out_d, size*sizeof(float), cudaMemcpyDeviceToHost));
}

void poisson_logpmf_dh2d(const int *k, const float *r, float *out, const int size)
{
  float *r_d;
  cuda_errchk(cudaMalloc(&r_d, size*sizeof(float)));
  cuda_errchk(cudaMemcpy(r_d, r, size*sizeof(float), cudaMemcpyHostToDevice));

  kernels::call_poisson_logpmf_kernel(k, r_d, out, size);
}

void poisson_logpmf_hd2d(const int *k, const float *r, float *out, const int size)
{
  int *k_d;
  cuda_errchk(cudaMalloc(&k_d, size*sizeof(int)));
  cuda_errchk(cudaMemcpy(k_d, k, size*sizeof(int), cudaMemcpyHostToDevice));

  kernels::call_poisson_logpmf_kernel(k_d, r, out, size);
}

void poisson_logpmf_dd2d(const int *k, const float *r, float *out, const int size)
{
  kernels::call_poisson_logpmf_kernel(k, r, out, size);
}
