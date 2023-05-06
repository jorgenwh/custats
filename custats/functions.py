import numpy as __np
import cupy as __cp
import scipy as __scipy
import cupyx as __cupyx

from custats_backend import poisson_logpmf_hh2h as __poisson_logpmf_hh2h
from custats_backend import poisson_logpmf_dh2d as __poisson_logpmf_dh2d
from custats_backend import poisson_logpmf_hd2d as __poisson_logpmf_hd2d
from custats_backend import poisson_logpmf_dd2d as __poisson_logpmf_dd2d
from custats_backend import _logpmf as __backend_logpmf

def numpy_logpmf(k, r):
    """NumPy-based Poisson Log Probability Mass Function

    Input:
        k: (numpy.ndarray) Number of observed events.
        r: (numpy.ndarray) Poisson rate parameter.
    Output:
        (numpy.ndarray) Log probability mass function.
    """
    return k * __np.log(r) - r - __scipy.special.gammaln(k+1)

def cupy_logpmf(k, r):
    """GPU Accelerated Poisson Log Probability Mass Function using CuPy as a drop-in replacement for NumPy

    Input:
        k: (cupy.ndarray) Number of observed events.
        r: (cupy.ndarray) Poisson rate parameter.
    Output:
        (cupy.ndarray) Log probability mass function.
    """
    return k * __cp.log(r) - r - __cupyx.scipy.special.gammaln(k+1)

def logpmf(k, r):
    """
    GPU Accelerated Poisson Log Probability Mass Function.

    Input:
        k: (numpy.ndarray, cupy.ndarray) Number of observed events.
        r: (numpy.ndarray, cupy.ndarray) Poisson rate parameter.
    Output:
        (numpy.ndarray, cupy.ndarray) Log probability mass function.
    """
    assert k.dtype == __np.int32, "k.dtype must be np.int32"
    assert r.dtype == __np.float64, "r.dtype must be np.float64"
    assert k.shape == r.shape, "k and r must have equal shapes"
    assert isinstance(k, (__np.ndarray, __cp.ndarray)) and isinstance(r, (__np.ndarray, __cp.ndarray))

    if isinstance(k, __np.ndarray) and isinstance(r, __np.ndarray):
        return __poisson_logpmf_hh2h(k, r)
    if isinstance(k, __cp.ndarray) and isinstance(r, __np.ndarray):
        out = __cp.zeros_like(k, dtype=__np.float64)
        __poisson_logpmf_dh2d(k.data.ptr, r, out.data.ptr)
        return out
    if isinstance(k, __np.ndarray) and isinstance(r, __cp.ndarray):
        out = __cp.zeros_like(r, dtype=__np.float64)
        __poisson_logpmf_hd2d(k, r.data.ptr, out.data.ptr)
        return out
    if isinstance(k, __cp.ndarray) and isinstance(r, __cp.ndarray):
        out = __cp.zeros_like(k, dtype=__np.float64)
        __poisson_logpmf_dd2d(k.data.ptr, r.data.ptr, out.data.ptr, out.size)
        return out

    return NotImplemented

def experimental_logpmf(observed_counts, counts, base_lambda, error_rate):
    """Experimental CUDA implementation of the outer function using LOGPMF in KAGE."""
    assert counts.shape[1] == 15, f"Number of model counts must be 15. Encountered {counts.shape[1]}"
    out = __cp.zeros_like(observed_counts, dtype=__np.float32)
    n_counts = counts.shape[0]
    __backend_logpmf(
        observed_counts.data.ptr, counts.data.ptr, n_counts, base_lambda, error_rate, out.data.ptr)
    return out
