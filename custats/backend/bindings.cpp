#include <inttypes.h>
#include <string>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "functions.h"

namespace py = pybind11;

PYBIND11_MODULE(custats_backend, m) 
{
  m.doc() = "Documentation for the custats C and CUDA backend module";

  m.def("poisson_logpmf_hh2h", [](const py::array_t<int> &k, const py::array_t<double> &r)
  {
    py::buffer_info buf = k.request();
    const int size = k.size();
    const int *k_data = k.data();
    const double *r_data = r.data();

    auto out = py::array_t<double>(buf.size);
    double *out_data = out.mutable_data();

    poisson_logpmf_hh2h(k_data, r_data, out_data, size);
    return out;
  });

  m.def("poisson_logpmf_dh2d", [](long k_ptr, const py::array_t<double> &r, long out_ptr)
  {
    const int size = r.size();
    const int *k_data = reinterpret_cast<int *>(k_ptr);
    const double *r_data = r.data();
    double *out_data = reinterpret_cast<double *>(out_ptr);
    poisson_logpmf_dh2d(k_data, r_data, out_data, size);
  });

  m.def("poisson_logpmf_hd2d", [](const py::array_t<int> &k, long r_ptr, long out_ptr)
  {
    const int size = k.size();
    const int *k_data = k.data();
    const double *r_data = reinterpret_cast<double *>(r_ptr);
    double *out_data = reinterpret_cast<double *>(out_ptr);
    poisson_logpmf_hd2d(k_data, r_data, out_data, size);
  });

  m.def("poisson_logpmf_dd2d", [](long k_ptr, long r_ptr, long out_ptr, int size)
  {
    const int *k_data = reinterpret_cast<int *>(k_ptr);
    const double *r_data = reinterpret_cast<double *>(r_ptr);
    double *out_data = reinterpret_cast<double *>(out_ptr);
    poisson_logpmf_dd2d(k_data, r_data, out_data, size);
  });

  m.def("_logpmf", [](long observed_counts_ptr, long counts_ptr, unsigned int n_counts, float base_lambda, float error_rate, long out_ptr)
  {
    unsigned int *observed_counts = reinterpret_cast<unsigned int *>(observed_counts_ptr);
    float *counts = reinterpret_cast<float *>(counts_ptr);
    float *out = reinterpret_cast<float *>(out_ptr);
    poisson_logpmf_experimental(observed_counts, counts, n_counts,
        base_lambda, error_rate, out);
  });
}
