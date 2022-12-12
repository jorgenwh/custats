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
    float *out, int n_counts)
{
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int i_offset = gtid * 5;

  __shared__ float poisson_lambda[5];
  if (gtid < 5)
  {
    poisson_lambda[gtid] = (gtid + error_rate) * base_lambda;
  }
  __syncthreads();

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 5; i++)
  {
    sum += counts[i_offset + i];
  }

  float output = 0.0f;
#pragma unroll
  for (int i = 0; i < 5; i++)
  {
    output += expf(((observed_counts[gtid] * logf(poisson_lambda[i])) - poisson_lambda[i] - lgammaf(observed_counts[gtid] + 1)) + logf(counts[i_offset + i] / sum));
  }

  out[gtid] = logf(output);
}

void call_poisson_logpmf_experimental_kernel(
    int *observed_counts, float *counts, float base_lambda, float error_rate, 
    float *out, int n_counts)
{
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      poisson_logpmf_experimental_kernel, 0, 0));

  poisson_logpmf_experimental_kernel<<<SDIV(n_counts, thread_block_size), thread_block_size>>>(
      observed_counts, counts, base_lambda, error_rate, out, n_counts);
}

} // kernels
