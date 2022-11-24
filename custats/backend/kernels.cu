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

} // kernels
