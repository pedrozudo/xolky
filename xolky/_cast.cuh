#pragma once
#include <cstddef>
#include <cuda_runtime.h>

void upcast_f32_to_f64(const float* f32_dev,
                       double* f64_dev,
                       std::size_t n,
                       cudaStream_t stream = 0);

void downcast_f64_to_f32(const double* f64_dev,
                       float* f32_dev,
                       std::size_t n,
                       cudaStream_t stream = 0);