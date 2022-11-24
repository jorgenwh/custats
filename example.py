import numpy as np
import cupy as cp
import scipy

from custats import poisson_logpmf


# Reference implementation used to check correctness
def ref(k, r):
    return k * np.log(r) - r - scipy.special.gammaln(k+1)


# len(k) == len(r) == size
size = 40_000_000

# k, r and reference output initialized in the host RAM
k = np.random.randint(low=1, high=100, size=size, dtype=np.int32)
r = np.random.uniform(low=1, high=100, size=size)
ref_out = ref(k, r)

# k, r and reference output copied to device RAM
k_d = cp.asarray(k)
r_d = cp.asarray(r)
ref_out_d = cp.asarray(ref_out)


# poisson_logpmf when both k and r are NumPy arrays. 
# Output will also be a NumPy array. hh2h (host,host 2 host)
out_hh2h = poisson_logpmf(k, r)

# poisson_logpmf when k is a CuPy array and r is a NumPy array.
# Output will be a CuPy array. dh2d (device,host 2 device)
out_dh2d = poisson_logpmf(k_d, r)

# poisson_logpmf when k is a NumPy array and r is a CuPy array.
# Output will be a CuPy array. hd2d (host,device 2 device)
out_hd2d = poisson_logpmf(k, r_d)

# poisson_logpmf when both k and r are CuPy arrays. 
# Output will also be a CuPy array. dd2d (device,device 2 device)
out_dd2d = poisson_logpmf(k_d, r_d)


# Asserts
assert isinstance(out_hh2h, np.ndarray)
assert isinstance(out_dh2d, cp.ndarray)
assert isinstance(out_hd2d, cp.ndarray)
assert isinstance(out_dd2d, cp.ndarray)

assert \
        k.shape == r.shape == k_d.shape == r_d.shape == \
        out_hh2h.shape == out_dh2d.shape == out_hd2d.shape == out_dd2d.shape

assert k.dtype == k_d.dtype == np.int32
assert r.dtype == r_d.dtype == np.float64
assert out_hh2h.dtype == out_dh2d.dtype == out_hd2d.dtype == out_dd2d.dtype == np.float64

np.testing.assert_almost_equal(out_hh2h, ref_out, decimal=4)
cp.testing.assert_array_almost_equal(out_dh2d, ref_out_d, decimal=4)
cp.testing.assert_array_almost_equal(out_hd2d, ref_out_d, decimal=4)
cp.testing.assert_array_almost_equal(out_dd2d, ref_out_d, decimal=4)
