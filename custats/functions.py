import numpy as np
import cupy as cp

from custats_backend import poisson_logpmf_hh2h
from custats_backend import poisson_logpmf_dh2d
from custats_backend import poisson_logpmf_hd2d
from custats_backend import poisson_logpmf_dd2d

def poisson_logpmf(k, r):
    if isinstance(k, np.ndarray) and isinstance(r, np.ndarray):
        pass
    if isinstance(k, cp.ndarray) and isinstance(r, np.ndarray):
        pass
    if isinstance(k, np.ndarray) and isinstance(r, cp.ndarray):
        pass
    if isinstance(k, cp.ndarray) and isinstance(r, cp.ndarray):
        pass

    return NotImplemented
