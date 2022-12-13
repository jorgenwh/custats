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
    total_ns = 0

    print(f"Profiling {logpmf_func}")
    for i in range(runs):
        observed_counts = np.random.randint(0, 20, n_counts, dtype=np.int32)
        model_counts = (np.random.randint(0, 100, (n_counts, 5)) / 10).astype(np.float16) + 0.1

        # If using the experimental GPU solution, copy data to GPU and convert model_counts from f16 to f32
        if logpmf_func == experimental_logpmf:
            observed_counts = cp.asanyarray(observed_counts)
            model_counts = cp.asanyarray(model_counts.astype(np.float32))

        t = time.time_ns()
        result = logpmf_func(observed_counts, model_counts, BASE_LAMBDA, ERROR_RATE)
        cp.cuda.runtime.deviceSynchronize()
        total_ns += (time.time_ns() - t)

        print(f"{i}/{runs}", end="\r")
    print(f"{runs}/{runs}")

    return total_ns / runs

def assert_logpmf_func_correctness(experimental_func, reference_func, n_counts, runs, threshold=0.01):
    print("Asserting correctness")
    for i in range(runs):
        observed_counts = np.random.randint(0, 20, n_counts, dtype=np.int32)
        model_counts = (np.random.randint(0, 100, (n_counts, 5)) / 10).astype(np.float16) + 0.1
        observed_counts_d = cp.asarray(observed_counts)
        model_counts_d = cp.asarray(model_counts.astype(np.float32))

        reference_result = reference_func(observed_counts, model_counts, BASE_LAMBDA, ERROR_RATE)
        experimental_result = experimental_func(observed_counts_d, model_counts_d, BASE_LAMBDA, ERROR_RATE)
        experimental_result_h = cp.asnumpy(experimental_result)

        assert np.isfinite(experimental_result_h).all()
        assert experimental_result_h.shape == reference_result.shape
        assert np.allclose(experimental_result_h, reference_result, rtol=threshold)

        print(f"{i}/{runs}", end="\r")
    print(f"{runs}/{runs}")

if __name__ == "__main__":
    n_counts = 50000 
    runs = 100
    tolerance_threshold = 0.003

    assert_correctness = True
    profile = True

    if assert_correctness:
        assert_logpmf_func_correctness(
                experimental_func=experimental_logpmf, reference_func=func,
                n_counts=n_counts, runs=runs, threshold=tolerance_threshold)

    if profile:
        ns_call_h = profile_logpmf_func(logpmf_func=func, n_counts=n_counts, runs=runs)
        ns_call_d = profile_logpmf_func(logpmf_func=experimental_logpmf, n_counts=n_counts, runs=runs)
        ms_call_h = ns_call_h / 1e6
        ms_call_d = ns_call_d / 1e6
        s_call_h = ns_call_h / 1e9
        s_call_d = ns_call_d / 1e9

        print(f"mean time per call (func, CPU)                : {round(ms_call_h, 5)} ms")
        print(f"mean time per call (experimental_logpmf, GPU) : {round(ms_call_d, 5)} ms")
        print(f"GPU solution was {round(ns_call_h/ns_call_d, 1)}x faster")
