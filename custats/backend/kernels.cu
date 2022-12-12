#include <iostream>
#include <inttypes.h>
#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"

namespace kernels {

__global__ static void poisson_logpmf_kernel(
    const int *k, const double *r, double *out, const int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
  {
    out[i] = k[i] * logf(r[i]) - r[i] - lgammaf(k[i] + 1);
  }
}

void call_poisson_logpmf_kernel(const int *k, const double *r, double *out, const int size)
{
  poisson_logpmf_kernel<<<SDIV(size, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE>>>(
      k, r, out, size);
}

__global__ static void poisson_logpmf_experimental_kernel(
    int *observed_counts, float *counts, 
    float base_lambda, float error_rate,
    float *out, int n_counts,
    float *sums, float *frequencies, float *poisson_lambda, float *prob)
{
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int i_offset = gtid * 5;

  float sum = 0.0;
#pragma unroll
  for (int i = 0; i < 5; i++)
  {
    sum += counts[i_offset + i];
  }
  sums[gtid] = sum;

#pragma unroll
  for (int i = 0; i < 5; i++)
  {
    frequencies[i_offset + i] = logf(counts[i_offset + i] / sums[gtid]);
  }

#pragma unroll
  if (gtid < 5)
  {
      poisson_lambda[gtid] = (gtid + error_rate) * base_lambda;
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 5; i++)
  {
    prob[i_offset + i] = ((observed_counts[gtid] * logf(poisson_lambda[i])) - poisson_lambda[i] - lgammaf(observed_counts[gtid] + 1)) + frequencies[i_offset + i];
  }

#pragma unroll
  for (int i = 0; i < 5; i++)
  {
    out[gtid] += expf(prob[i_offset + i]);
  }
  out[gtid] = logf(out[gtid]);
}

void call_poisson_logpmf_experimental_kernel(
    int *observed_counts, float *counts, float base_lambda, float error_rate, 
    float *out, int n_counts)
{
  float *sums;
  cuda_errchk(cudaMalloc(&sums, n_counts*sizeof(float)));
  float *frequencies;
  cuda_errchk(cudaMalloc(&frequencies, n_counts*5*sizeof(float)));
  float *poisson_lambda;
  cuda_errchk(cudaMalloc(&poisson_lambda, 5*sizeof(float)));
  float *prob;
  cuda_errchk(cudaMalloc(&prob, n_counts*5*sizeof(float)));

  poisson_logpmf_experimental_kernel<<<SDIV(n_counts, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE>>>(
      observed_counts, counts, base_lambda, error_rate, out, n_counts,
      sums, frequencies, poisson_lambda, prob);

  cuda_errchk(cudaFree(sums));
  cuda_errchk(cudaFree(frequencies));
  cuda_errchk(cudaFree(poisson_lambda));
  cuda_errchk(cudaFree(prob));
}

} // kernels
