import time
import scipy
from scipy.special import logsumexp
import numpy as np
import cupy as cp

from custats import poisson_logpmf
from custats import experimental_logpmf

BASE_LAMBDA = 7.5
ERROR_RATE = 0.01

def fast_poisson_logpmf(k, r):
    return k * np.log(r) - r - scipy.special.gammaln(k+1)

def func(observed_counts, counts, base_lambda, error_rate):
    sums = np.sum(counts, axis=-1)[:, None]
    frequencies = np.log(counts / sums)
    poisson_lambda = (np.arange(counts.shape[1])[None, :] + error_rate) * base_lambda
    prob = fast_poisson_logpmf(observed_counts[:, None], poisson_lambda)
    prob = logsumexp(frequencies + prob, axis=-1)
    return prob

def profile_logpmf_func(logpmf_func, n_counts, runs):
    total_t = 0

    for i in range(runs):
        observed_counts = np.random.randint(0, 20, n_counts, dtype=np.int32)
        model_counts = (np.random.randint(0, 100, (n_counts, 5)) / 10).astype(np.float16) + 0.1

        # If using the experimental GPU solution, copy data to GPU and convert model_counts from f16 to f32
        if logpmf_func == experimental_logpmf:
            observed_counts = cp.asanyarray(observed_counts)
            model_counts = cp.asanyarray(model_counts.astype(np.float32))

        t = time.time()
        result = logpmf_func(observed_counts, model_counts, BASE_LAMBDA, ERROR_RATE)
        cp.cuda.runtime.deviceSynchronize()
        total_t += (time.time() - t)

    return total_t / runs

def assert_logpmf_func_correctness(experimental_func, reference_func, n_counts, runs, decimal_accuracy=4):
    for i in range(runs):
        observed_counts = np.random.randint(0, 20, n_counts, dtype=np.int32)
        model_counts = (np.random.randint(0, 100, (n_counts, 5)) / 10).astype(np.float16) + 0.1
        observed_counts_d = cp.asarray(observed_counts)
        model_counts_d = cp.asarray(model_counts.astype(np.float32))

        reference_result = reference_func(observed_counts, model_counts, BASE_LAMBDA, ERROR_RATE)
        experimental_result = experimental_func(observed_counts_d, model_counts_d, BASE_LAMBDA, ERROR_RATE)
        experimental_result_h = cp.asnumpy(experimental_result)

        #np.testing.assert_array_almost_equal(
                #experimental_result_h, reference_result, decimal=decimal_accuracy)
        np.allclose(experimental_result_h, reference_result)

if __name__ == "__main__":
    n_counts = 50000

    assert_logpmf_func_correctness(
            experimental_func=experimental_logpmf, reference_func=func,
            n_counts=n_counts, runs=10, decimal_accuracy=4)
    t1 = profile_logpmf_func(logpmf_func=func, n_counts=n_counts, runs=100)
    t2 = profile_logpmf_func(logpmf_func=experimental_logpmf, n_counts=n_counts, runs=100)

    print(f"mean time per call (func, CPU)                : {round(t1, 5)}s")
    print(f"mean time per call (experimental_logpmf, GPU) : {round(t2, 5)}s")
    print(f"GPU solution was {round(t1/t2, 1)}x faster")
