import numpy as __np
import cupy as __cp

from custats_backend import poisson_logpmf_hh2h as __poisson_logpmf_hh2h
from custats_backend import poisson_logpmf_dh2d as __poisson_logpmf_dh2d
from custats_backend import poisson_logpmf_hd2d as __poisson_logpmf_hd2d
from custats_backend import poisson_logpmf_dd2d as __poisson_logpmf_dd2d

def poisson_logpmf(k, r):
    assert k.dtype == __np.int32, "k.dtype must be np.int32"
    assert r.dtype == __np.float32, "r.dtype must be np.float32"
    assert k.shape == r.shape, "k and r must have equal shapes"
    assert isinstance(k, (__np.ndarray, __cp.ndarray)) and isinstance(r, (__np.ndarray, __cp.ndarray)), "k and r must be numpy or cupy arrays"

    if isinstance(k, __np.ndarray) and isinstance(r, __np.ndarray):
        return __poisson_logpmf_hh2h(k, r)
    if isinstance(k, __cp.ndarray) and isinstance(r, __np.ndarray):
        out = __cp.zeros_like(k, dtype=__np.float32)
        __poisson_logpmf_dh2d(k.data.ptr, r, out.data.ptr)
        return out
    if isinstance(k, __np.ndarray) and isinstance(r, __cp.ndarray):
        out = __cp.zeros_like(r, dtype=__np.float32)
        __poisson_logpmf_hd2d(k, r.data.ptr, out.data.ptr)
        return out
    if isinstance(k, __cp.ndarray) and isinstance(r, __cp.ndarray):
        out = __cp.zeros_like(k, dtype=__np.float32)
        __poisson_logpmf_dd2d(k.data.ptr, r.data.ptr, out.data.ptr, out.size)
        return out

    return NotImplemented
