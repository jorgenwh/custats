import time

import numpy as np
import cupy as cp
import scipy

from custats import poisson_logpmf

def numpy_poisson_logpmf(k, r):
    return k * np.log(r) - r - scipy.special.gammaln(k+1)

def time_call(k, r, func):
    t = time.time()
    out = func(k, r)
    cp.cuda.runtime.deviceSynchronize()
    return time.time() - t


size = 40_000_000
num_runs = 100

times = {
        "scipy": 0, 
        "numpy": 0, 
        "custats_hh2h": 0, 
        "custats_dh2d": 0, 
        "custats_hd2d": 0, 
        "custats_dd2d": 0
}

for i in range(1, num_runs+1):
    k = np.random.randint(low=1, high=100, size=size, dtype=np.int32)
    r = np.random.uniform(low=1, high=100, size=size).astype(np.float32)
    k_d = cp.asarray(k)
    r_d = cp.asarray(r)

    times["scipy"] += time_call(k, r, func=scipy.stats.poisson.logpmf)
    times["numpy"] += time_call(k, r, func=numpy_poisson_logpmf)
    times["custats_hh2h"] += time_call(k, r, func=poisson_logpmf)
    times["custats_dh2d"] += time_call(k_d, r, func=poisson_logpmf)
    times["custats_hd2d"] += time_call(k, r_d, func=poisson_logpmf)
    times["custats_dd2d"] += time_call(k_d, r_d, func=poisson_logpmf)

    print(f"run {i}/{num_runs}", end="\r")
print(f"run {i}/{num_runs}")

print("--- call time averaged over 100 calls ---")
for k in times:
    print(f"{k: <12} : {round(times[k]/num_runs, 4)} s")

