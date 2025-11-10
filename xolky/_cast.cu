#include "_cast.cuh"

#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>


////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
struct float_to_double {
    __device__
    double operator()(float x) const { return static_cast<double>(x); }
};


void upcast_f32_to_f64(const float* f32_dev,
                       double* f64_dev,
                       std::size_t n,
                       cudaStream_t stream)
{
    auto in  = thrust::device_pointer_cast(f32_dev);
    auto out = thrust::device_pointer_cast(f64_dev);

    thrust::transform(thrust::cuda::par.on(stream),
                      in, in + n, out,
                      float_to_double{});
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
struct double_to_float {
    __device__
    float operator()(double x) const { return static_cast<float>(x); }
};

void downcast_f64_to_f32(const double* f64_dev,
                       float* f32_dev,
                       std::size_t n,
                       cudaStream_t stream)
{
    auto in  = thrust::device_pointer_cast(f64_dev);
    auto out = thrust::device_pointer_cast(f32_dev);

    thrust::transform(thrust::cuda::par.on(stream),
                      in, in + n, out,
                      double_to_float{});
}

#ifdef __CUDACC__
#pragma message("_cast.cu compiled with NVCC")
#endif
