#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <cudss.h>
#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;
namespace ffi = xla::ffi;

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " — "  \
                << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

// template <typename T>
// inline void print_device_array(const T *d_ptr, int N,
//                                const char *label = "val") {
//   static_assert(std::is_same<T, float>::value || std::is_same<T,
//   double>::value,
//                 "Only float and double are supported in this printer.");

//   // Allocate host buffer
//   std::vector<T> h_buf(N);

//   // Copy device -> host
//   cudaMemcpy(h_buf.data(), d_ptr, N * sizeof(T), cudaMemcpyDeviceToHost);

//   // Print with appropriate format
//   for (int i = 0; i < N; i++) {
//     if constexpr (std::is_same<T, float>::value) {
//       printf("%s[%d] = %.6f\n", label, i, h_buf[i]);
//     } else if constexpr (std::is_same<T, double>::value) {
//       printf("%s[%d] = %.6lf\n", label, i, h_buf[i]);
//     }
//   }
// }

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

struct cudssMemPool {
public:
  int alloc(void **ptr, size_t size, cudaStream_t stream) {
    // printf("alloc() from the ExampleDeviceMemPool allocates memory at %p "
    //        "(allocation size %zu)\n",
    //        (void *)*ptr, size);
    int status = cudaMallocAsync(ptr, size, stream);
    return status;
  }

  int dealloc(void *ptr, size_t size, cudaStream_t stream) {
    int status = cudaFreeAsync(ptr, stream);
    return status;
  }
};

int cudss_alloc(void *ctx, void **ptr, size_t size, cudaStream_t stream) {
  return reinterpret_cast<cudssMemPool *>(ctx)->alloc(ptr, size, stream);
}

int cudss_dealloc(void *ctx, void *ptr, size_t size, cudaStream_t stream) {
  return reinterpret_cast<cudssMemPool *>(ctx)->dealloc(ptr, size, stream);
}

struct CuDssSparseCholesky {

public:
  cudssHandle_t handle_{};
  cudssConfig_t config_{};
  cudssData_t data_{};
  cudssDeviceMemHandler_t mem_handler_{};

  cudssMatrix_t A_{};
  cudssMatrix_t x_{};
  cudssMatrix_t b_{};

  double *csr_values_64_ = nullptr;
  double *x_64_ = nullptr;
  double *b_64_ = nullptr;

  CuDssSparseCholesky() {
    cudssCreate(&handle_);
    cudssConfigCreate(&config_);
    cudssDataCreate(handle_, &data_);

    cudssMemPool pool = cudssMemPool();
    mem_handler_.ctx = reinterpret_cast<void *>(&pool);
    mem_handler_.device_alloc = cudss_alloc;
    mem_handler_.device_free = cudss_dealloc;

    cudssSetDeviceMemHandler(handle_, &mem_handler_);
  }

  ~CuDssSparseCholesky() {
    if (b_64_) {
      cudaFree(b_64_);
    }
    if (x_64_) {
      cudaFree(x_64_);
    }
    if (csr_values_64_) {
      cudaFree(csr_values_64_);
    }

    if (A_)
      cudssMatrixDestroy(A_);
    if (data_ && handle_)
      cudssDataDestroy(handle_, data_);
    if (config_) {
      cudssConfigDestroy(config_);
    }
    if (handle_)
      cudssDestroy(handle_);
  }

  std::intptr_t address() const { return reinterpret_cast<int64_t>(this); }
};

CuDssSparseCholesky *fetchCuDssSparseCholeskyHostPtr(cudaStream_t stream,
                                                     int64_t address) {
  return reinterpret_cast<CuDssSparseCholesky *>(address);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

static ffi::Error XolkyInitStructureImpl(cudaStream_t stream, int64_t address,
                                         int64_t ncols, int64_t nnz,
                                         ffi::Buffer<ffi::S32> csr_indices,
                                         ffi::Buffer<ffi::S32> csr_indptr) {
  auto h = fetchCuDssSparseCholeskyHostPtr(stream, address);
  cudssSetStream(h->handle_, stream);

  int32_t *indptr = csr_indptr.typed_data();
  int32_t *indices = csr_indices.typed_data();

  cudssMatrixCreateCsr(&h->A_, ncols, ncols, nnz, static_cast<void *>(indptr),
                       NULL, static_cast<void *>(indices), NULL, CUDA_R_32I,
                       CUDA_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER,
                       CUDSS_BASE_ZERO);

  int64_t n_rhs = 1;
  int64_t n_rows = ncols;
  int64_t ldx = ncols;
  int64_t ldb = n_rows;

  cudssMatrixCreateDn(&h->x_, ncols, n_rhs, ldx, NULL, CUDA_R_64F,
                      CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&h->b_, ncols, n_rhs, ldb, NULL, CUDA_R_64F,
                      CUDSS_LAYOUT_COL_MAJOR);

  CUDA_CHECK(cudaMalloc((void **)&h->csr_values_64_, nnz * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&h->x_64_, ncols * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&h->b_64_, ncols * sizeof(double)));

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(XolkyInitStructure, XolkyInitStructureImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("address")
                           .Attr<int64_t>("ncols")
                           .Attr<int64_t>("nnz")
                           .Arg<ffi::Buffer<ffi::S32>>()
                           .Arg<ffi::Buffer<ffi::S32>>());

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

static ffi::Error XolkyReorderImpl(cudaStream_t stream, int64_t address) {
  auto h = fetchCuDssSparseCholeskyHostPtr(stream, address);

  // Set ordering to METIS
  // https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t
  cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
  cudssConfigSet(h->config_, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg,
                 sizeof(cudssAlgType_t));
  cudssExecute(h->handle_, CUDSS_PHASE_REORDERING, h->config_, h->data_, h->A_,
               NULL, NULL);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(XolkyReorder, XolkyReorderImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("address"));

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

static ffi::Error XolkyAnalyzeImpl(cudaStream_t stream, int64_t address) {
  auto h = fetchCuDssSparseCholeskyHostPtr(stream, address);
  cudssExecute(h->handle_, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, h->config_,
               h->data_, h->A_, NULL, NULL);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(XolkyAnalyze, XolkyAnalyzeImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("address"));

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

static ffi::Error XolkyFactorizeImpl(cudaStream_t stream, int64_t address,
                                     ffi::Buffer<ffi::F64> csr_values) {
  auto h = fetchCuDssSparseCholeskyHostPtr(stream, address);
  cudssMatrixSetValues(h->A_, csr_values.typed_data());
  // print_device_array(csr_values.typed_data(), csr_values.element_count());
  // print_device_array(h->csr_values_64_, csr_values.element_count());
  cudssExecute(h->handle_, CUDSS_PHASE_FACTORIZATION, h->config_, h->data_,
               h->A_, NULL, NULL);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(XolkyFactorize, XolkyFactorizeImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("address")
                           .Arg<ffi::Buffer<ffi::F64>>());

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

static ffi::Error XolkyRefactorizeImpl(cudaStream_t stream, int64_t address,
                                       ffi::Buffer<ffi::F64> csr_values) {
  auto h = fetchCuDssSparseCholeskyHostPtr(stream, address);
  cudssMatrixSetValues(h->A_, csr_values.typed_data());
  cudssExecute(h->handle_, CUDSS_PHASE_REFACTORIZATION, h->config_, h->data_,
               h->A_, NULL, NULL);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(XolkyRefactorize, XolkyRefactorizeImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("address")
                           .Arg<ffi::Buffer<ffi::F64>>());

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

static ffi::Error XolkySolveImpl(cudaStream_t stream, int64_t address,
                                 ffi::Buffer<ffi::F64> b,
                                 ffi::ResultBuffer<ffi::F64> x) {
  auto h = fetchCuDssSparseCholeskyHostPtr(stream, address);
  cudssMatrixSetValues(h->b_, b.typed_data());
  cudssMatrixSetValues(h->x_, x->typed_data());
  cudssExecute(h->handle_, CUDSS_PHASE_SOLVE, h->config_, h->data_, h->A_,
               h->x_, h->b_);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(XolkySolve, XolkySolveImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("address")
                           .Arg<ffi::Buffer<ffi::F64>>()
                           .Ret<ffi::Buffer<ffi::F64>>(),
                       {xla::ffi::Traits::kCmdBufferCompatible});

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> py::capsule EncapsulateFfiCall(T *fn) {
  // https://docs.jax.dev/en/latest/ffi.html#building-and-registering-an-ffi-handler
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}

PYBIND11_MODULE(_xolky, m) {
  py::class_<CuDssSparseCholesky>(m, "CuDssSparseCholesky")
      .def(py::init<>())
      .def("address", &CuDssSparseCholesky::address);

  m.def("init_structure",
        []() { return EncapsulateFfiCall(XolkyInitStructure); });
  m.def("reorder", []() { return EncapsulateFfiCall(XolkyReorder); });
  m.def("analyze", []() { return EncapsulateFfiCall(XolkyAnalyze); });
  m.def("factorize", []() { return EncapsulateFfiCall(XolkyFactorize); });
  m.def("refactorize", []() { return EncapsulateFfiCall(XolkyRefactorize); });
  m.def("solve", []() { return EncapsulateFfiCall(XolkySolve); });
}
