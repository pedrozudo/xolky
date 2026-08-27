#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <cudss.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;
namespace ffi = xla::ffi;

namespace {

using SolverId = std::uint64_t;

std::string CudaErrorMessage(cudaError_t status, const char *operation) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(status) << " ("
          << static_cast<int>(status) << ")";
  return message.str();
}

std::string CudssErrorMessage(cudssStatus_t status, const char *operation) {
  std::ostringstream message;
  message << operation << " failed with cuDSS status "
          << static_cast<int>(status);
  return message.str();
}

void CheckCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(status, operation));
  }
}

void CheckCudss(cudssStatus_t status, const char *operation) {
  if (status != CUDSS_STATUS_SUCCESS) {
    throw std::runtime_error(CudssErrorMessage(status, operation));
  }
}

class DeviceGuard {
public:
  explicit DeviceGuard(int device) {
    CheckCuda(cudaGetDevice(&previous_device_), "cudaGetDevice");
    if (previous_device_ != device) {
      CheckCuda(cudaSetDevice(device), "cudaSetDevice");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      cudaSetDevice(previous_device_);
    }
  }

  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard &operator=(const DeviceGuard &) = delete;

private:
  int previous_device_ = 0;
  bool restore_ = false;
};

struct SolverEntry {
  SolverEntry(int64_t n_value, int64_t nnz_value, int device_value)
      : n(n_value), nnz(nnz_value), device_ordinal(device_value) {}

  SolverEntry(const SolverEntry &) = delete;
  SolverEntry &operator=(const SolverEntry &) = delete;

  void InitializeResources() {
    DeviceGuard guard(device_ordinal);

    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
    CheckCuda(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming),
              "cudaEventCreateWithFlags(input_ready)");
    CheckCuda(cudaEventCreateWithFlags(&output_ready, cudaEventDisableTiming),
              "cudaEventCreateWithFlags(output_ready)");

    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&csr_indices),
                         static_cast<size_t>(nnz) * sizeof(int32_t)),
              "cudaMalloc(csr_indices)");
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&csr_indptr),
                         static_cast<size_t>(n + 1) * sizeof(int32_t)),
              "cudaMalloc(csr_indptr)");
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&csr_values),
                         static_cast<size_t>(nnz) * sizeof(double)),
              "cudaMalloc(csr_values)");
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&rhs),
                         static_cast<size_t>(n) * sizeof(double)),
              "cudaMalloc(rhs)");
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&solution),
                         static_cast<size_t>(n) * sizeof(double)),
              "cudaMalloc(solution)");

    CheckCudss(cudssCreate(&handle), "cudssCreate");
    CheckCudss(cudssConfigCreate(&config), "cudssConfigCreate");
    CheckCudss(cudssDataCreate(handle, &data), "cudssDataCreate");
    CheckCudss(cudssSetStream(handle, stream), "cudssSetStream");

    CheckCudss(
        cudssMatrixCreateCsr(&matrix, n, n, nnz, csr_indptr, nullptr,
                             csr_indices, csr_values, CUDA_R_32I, CUDA_R_64F,
                             CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER,
                             CUDSS_BASE_ZERO),
        "cudssMatrixCreateCsr");
    CheckCudss(cudssMatrixCreateDn(&solution_matrix, n, 1, n, solution,
                                   CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
               "cudssMatrixCreateDn(solution)");
    CheckCudss(cudssMatrixCreateDn(&rhs_matrix, n, 1, n, rhs, CUDA_R_64F,
                                   CUDSS_LAYOUT_COL_MAJOR),
               "cudssMatrixCreateDn(rhs)");
  }

  ~SolverEntry() noexcept {
    int current_device = 0;
    bool restore_device = false;
    if (cudaGetDevice(&current_device) == cudaSuccess &&
        current_device != device_ordinal) {
      cudaSetDevice(device_ordinal);
      restore_device = true;
    }

    if (stream != nullptr) {
      cudaStreamSynchronize(stream);
    }

    if (rhs_matrix != nullptr) {
      cudssMatrixDestroy(rhs_matrix);
    }
    if (solution_matrix != nullptr) {
      cudssMatrixDestroy(solution_matrix);
    }
    if (matrix != nullptr) {
      cudssMatrixDestroy(matrix);
    }
    if (data != nullptr && handle != nullptr) {
      cudssDataDestroy(handle, data);
    }
    if (config != nullptr) {
      cudssConfigDestroy(config);
    }
    if (handle != nullptr) {
      cudssDestroy(handle);
    }

    if (csr_indices != nullptr) {
      cudaFree(csr_indices);
    }
    if (csr_indptr != nullptr) {
      cudaFree(csr_indptr);
    }
    if (csr_values != nullptr) {
      cudaFree(csr_values);
    }
    if (rhs != nullptr) {
      cudaFree(rhs);
    }
    if (solution != nullptr) {
      cudaFree(solution);
    }
    if (input_ready != nullptr) {
      cudaEventDestroy(input_ready);
    }
    if (output_ready != nullptr) {
      cudaEventDestroy(output_ready);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (restore_device) {
      cudaSetDevice(current_device);
    }
  }

  int64_t n;
  int64_t nnz;
  int device_ordinal;

  cudssHandle_t handle{};
  cudssConfig_t config{};
  cudssData_t data{};
  cudssMatrix_t matrix{};
  cudssMatrix_t solution_matrix{};
  cudssMatrix_t rhs_matrix{};

  int32_t *csr_indices = nullptr;
  int32_t *csr_indptr = nullptr;
  double *csr_values = nullptr;
  double *rhs = nullptr;
  double *solution = nullptr;

  cudaStream_t stream = nullptr;
  cudaEvent_t input_ready = nullptr;
  cudaEvent_t output_ready = nullptr;

  bool initialized = false;
  bool factorized = false;
  bool failed = false;
  std::mutex mutex;
};

class SolverRegistry {
public:
  SolverId Create(int64_t n, int64_t nnz, int device_ordinal) {
    if (n <= 0) {
      throw std::invalid_argument("n must be positive");
    }
    if (nnz <= 0) {
      throw std::invalid_argument("nnz must be positive");
    }

    auto entry = std::make_shared<SolverEntry>(n, nnz, device_ordinal);
    entry->InitializeResources();

    SolverId id = next_id_.fetch_add(1, std::memory_order_relaxed);
    if (id == 0) {
      throw std::overflow_error("xolky solver identifier space exhausted");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    solvers_.emplace(id, std::move(entry));
    return id;
  }

  std::shared_ptr<SolverEntry> Get(SolverId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = solvers_.find(id);
    if (iterator == solvers_.end()) {
      throw std::invalid_argument("unknown or closed xolky solver identifier " +
                                  std::to_string(id));
    }
    return iterator->second;
  }

  bool Destroy(SolverId id) {
    std::shared_ptr<SolverEntry> entry;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto iterator = solvers_.find(id);
      if (iterator == solvers_.end()) {
        return false;
      }
      entry = std::move(iterator->second);
      solvers_.erase(iterator);
    }
    return true;
  }

  void Clear() {
    std::unordered_map<SolverId, std::shared_ptr<SolverEntry>> entries;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      entries.swap(solvers_);
    }
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return solvers_.size();
  }

private:
  std::atomic<SolverId> next_id_{1};
  mutable std::mutex mutex_;
  std::unordered_map<SolverId, std::shared_ptr<SolverEntry>> solvers_;
};

SolverRegistry &Registry() {
  static auto *registry = new SolverRegistry();
  return *registry;
}

SolverId ReadSolverId(ffi::BufferR0<ffi::DataType::U64> solver_id) {
  auto *pointer = solver_id.typed_data();
  cudaPointerAttributes attributes{};
  cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    cudaGetLastError();
    throw std::invalid_argument(
        "solver_id must be a uint64 scalar in pinned host memory");
  }
  if (attributes.type != cudaMemoryTypeHost &&
      attributes.type != cudaMemoryTypeManaged) {
    throw std::invalid_argument(
        "solver_id must be a uint64 scalar in pinned host memory");
  }
  SolverId id = *pointer;
  if (id == 0) {
    throw std::invalid_argument("solver identifier 0 is invalid");
  }
  return id;
}

void ValidateDevice(const SolverEntry &entry, int32_t device_ordinal) {
  if (entry.device_ordinal != device_ordinal) {
    throw std::invalid_argument(
        "solver belongs to CUDA device " +
        std::to_string(entry.device_ordinal) + " but the FFI call executes on " +
        std::to_string(device_ordinal));
  }
}

void BeginOperation(SolverEntry &entry, cudaStream_t caller_stream) {
  CheckCuda(cudaEventRecord(entry.input_ready, caller_stream),
            "cudaEventRecord(input_ready)");
  CheckCuda(cudaStreamWaitEvent(entry.stream, entry.input_ready, 0),
            "cudaStreamWaitEvent(input_ready)");
}

void EndOperation(SolverEntry &entry, cudaStream_t caller_stream) {
  CheckCuda(cudaEventRecord(entry.output_ready, entry.stream),
            "cudaEventRecord(output_ready)");
  CheckCuda(cudaStreamWaitEvent(caller_stream, entry.output_ready, 0),
            "cudaStreamWaitEvent(output_ready)");
}

class OperationGuard {
public:
  explicit OperationGuard(SolverEntry &entry) : entry_(entry) {}

  OperationGuard(const OperationGuard &) = delete;
  OperationGuard &operator=(const OperationGuard &) = delete;

  ~OperationGuard() {
    if (started_ && !completed_) {
      entry_.failed = true;
    }
  }

  void Begin(cudaStream_t caller_stream) {
    started_ = true;
    BeginOperation(entry_, caller_stream);
  }

  void Complete(cudaStream_t caller_stream) {
    EndOperation(entry_, caller_stream);
    completed_ = true;
  }

private:
  SolverEntry &entry_;
  bool started_ = false;
  bool completed_ = false;
};

// These handlers deliberately do not claim kCmdBufferCompatible. They use
// solver-owned device allocations plus a private CUDA stream and event
// handoffs. XLA's command-buffer-compatible FFI contract requires device
// allocations to arrive as buffer arguments, and CUDA graph capture across
// this private-stream boundary has not been established as safe. Keep the
// trait disabled unless the native execution model changes and capture is
// validated independently.

ffi::Error SetupImpl(
    cudaStream_t caller_stream, int32_t device_ordinal,
    ffi::BufferR0<ffi::DataType::U64> solver_id,
    ffi::BufferR0<ffi::DataType::U8> sequence,
    ffi::BufferR1<ffi::DataType::S32> csr_indices,
    ffi::BufferR1<ffi::DataType::S32> csr_indptr,
    ffi::ResultBufferR0<ffi::DataType::U8> sequence_out) {
  try {
    SolverId id = ReadSolverId(solver_id);
    auto entry = Registry().Get(id);
    ValidateDevice(*entry, device_ordinal);

    if (static_cast<int64_t>(csr_indices.element_count()) != entry->nnz) {
      return ffi::Error::InvalidArgument("csr_indices has the wrong length");
    }
    if (static_cast<int64_t>(csr_indptr.element_count()) != entry->n + 1) {
      return ffi::Error::InvalidArgument("csr_indptr has the wrong length");
    }

    std::lock_guard<std::mutex> lock(entry->mutex);
    if (entry->failed) {
      return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                        "solver is in a failed state; destroy and recreate it");
    }
    if (entry->initialized) {
      return ffi::Error::InvalidArgument("solver setup was already completed");
    }

    OperationGuard operation(*entry);
    operation.Begin(caller_stream);
    CheckCuda(cudaMemcpyAsync(entry->csr_indices, csr_indices.typed_data(),
                              static_cast<size_t>(entry->nnz) * sizeof(int32_t),
                              cudaMemcpyDeviceToDevice, entry->stream),
              "cudaMemcpyAsync(csr_indices)");
    CheckCuda(cudaMemcpyAsync(entry->csr_indptr, csr_indptr.typed_data(),
                              static_cast<size_t>(entry->n + 1) * sizeof(int32_t),
                              cudaMemcpyDeviceToDevice, entry->stream),
              "cudaMemcpyAsync(csr_indptr)");

    cudssAlgType_t reorder_algorithm = CUDSS_ALG_DEFAULT;
    CheckCudss(cudssConfigSet(entry->config, CUDSS_CONFIG_REORDERING_ALG,
                              &reorder_algorithm, sizeof(reorder_algorithm)),
               "cudssConfigSet(CUDSS_CONFIG_REORDERING_ALG)");
    CheckCudss(cudssExecute(entry->handle, CUDSS_PHASE_ANALYSIS, entry->config,
                            entry->data, entry->matrix, nullptr, nullptr),
               "cudssExecute(CUDSS_PHASE_ANALYSIS)");
    operation.Complete(caller_stream);

    entry->initialized = true;
    return ffi::Error::Success();
  } catch (const std::invalid_argument &error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception &error) {
    return ffi::Error::Internal(error.what());
  }
}

XLA_FFI_DEFINE_HANDLER(
    XolkySetup, SetupImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::BufferR0<ffi::DataType::U64>>()
        .Arg<ffi::BufferR0<ffi::DataType::U8>>()
        .Arg<ffi::BufferR1<ffi::DataType::S32>>()
        .Arg<ffi::BufferR1<ffi::DataType::S32>>()
        .Ret<ffi::BufferR0<ffi::DataType::U8>>());

ffi::Error RefactorImpl(
    cudaStream_t caller_stream, int32_t device_ordinal,
    ffi::BufferR0<ffi::DataType::U64> solver_id,
    ffi::BufferR0<ffi::DataType::U8> sequence,
    ffi::BufferR1<ffi::DataType::F64> csr_values,
    ffi::ResultBufferR0<ffi::DataType::U8> sequence_out) {
  try {
    SolverId id = ReadSolverId(solver_id);
    auto entry = Registry().Get(id);
    ValidateDevice(*entry, device_ordinal);

    if (static_cast<int64_t>(csr_values.element_count()) != entry->nnz) {
      return ffi::Error::InvalidArgument("csr_values has the wrong length");
    }

    std::lock_guard<std::mutex> lock(entry->mutex);
    if (entry->failed) {
      return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                        "solver is in a failed state; destroy and recreate it");
    }
    if (!entry->initialized) {
      return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                        "solver setup has not completed");
    }

    OperationGuard operation(*entry);
    operation.Begin(caller_stream);
    CheckCuda(cudaMemcpyAsync(entry->csr_values, csr_values.typed_data(),
                              static_cast<size_t>(entry->nnz) * sizeof(double),
                              cudaMemcpyDeviceToDevice, entry->stream),
              "cudaMemcpyAsync(csr_values)");

    int phase = entry->factorized ? CUDSS_PHASE_REFACTORIZATION
                                  : CUDSS_PHASE_FACTORIZATION;
    CheckCudss(cudssExecute(entry->handle, phase, entry->config, entry->data,
                            entry->matrix, nullptr, nullptr),
               entry->factorized
                   ? "cudssExecute(CUDSS_PHASE_REFACTORIZATION)"
                   : "cudssExecute(CUDSS_PHASE_FACTORIZATION)");
    operation.Complete(caller_stream);

    entry->factorized = true;
    return ffi::Error::Success();
  } catch (const std::invalid_argument &error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception &error) {
    return ffi::Error::Internal(error.what());
  }
}

XLA_FFI_DEFINE_HANDLER(
    XolkyRefactor, RefactorImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::BufferR0<ffi::DataType::U64>>()
        .Arg<ffi::BufferR0<ffi::DataType::U8>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Ret<ffi::BufferR0<ffi::DataType::U8>>());

ffi::Error SolveImpl(
    cudaStream_t caller_stream, int32_t device_ordinal,
    ffi::BufferR0<ffi::DataType::U64> solver_id,
    ffi::BufferR0<ffi::DataType::U8> sequence,
    ffi::BufferR1<ffi::DataType::F64> right_hand_side,
    ffi::ResultBufferR0<ffi::DataType::U8> sequence_out,
    ffi::ResultBufferR1<ffi::DataType::F64> result) {
  try {
    SolverId id = ReadSolverId(solver_id);
    auto entry = Registry().Get(id);
    ValidateDevice(*entry, device_ordinal);

    if (static_cast<int64_t>(right_hand_side.element_count()) != entry->n) {
      return ffi::Error::InvalidArgument("right-hand side has the wrong length");
    }
    if (static_cast<int64_t>(result->element_count()) != entry->n) {
      return ffi::Error::InvalidArgument("solution output has the wrong length");
    }

    std::lock_guard<std::mutex> lock(entry->mutex);
    if (entry->failed) {
      return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                        "solver is in a failed state; destroy and recreate it");
    }
    if (!entry->factorized) {
      return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                        "solver has not been factorized");
    }

    OperationGuard operation(*entry);
    operation.Begin(caller_stream);
    CheckCuda(cudaMemcpyAsync(entry->rhs, right_hand_side.typed_data(),
                              static_cast<size_t>(entry->n) * sizeof(double),
                              cudaMemcpyDeviceToDevice, entry->stream),
              "cudaMemcpyAsync(rhs)");
    CheckCudss(cudssExecute(entry->handle, CUDSS_PHASE_SOLVE, entry->config,
                            entry->data, entry->matrix,
                            entry->solution_matrix, entry->rhs_matrix),
               "cudssExecute(CUDSS_PHASE_SOLVE)");
    CheckCuda(cudaMemcpyAsync(result->typed_data(), entry->solution,
                              static_cast<size_t>(entry->n) * sizeof(double),
                              cudaMemcpyDeviceToDevice, entry->stream),
              "cudaMemcpyAsync(solution)");
    operation.Complete(caller_stream);

    return ffi::Error::Success();
  } catch (const std::invalid_argument &error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception &error) {
    return ffi::Error::Internal(error.what());
  }
}

XLA_FFI_DEFINE_HANDLER(
    XolkySolve, SolveImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::BufferR0<ffi::DataType::U64>>()
        .Arg<ffi::BufferR0<ffi::DataType::U8>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Ret<ffi::BufferR0<ffi::DataType::U8>>()
        .Ret<ffi::BufferR1<ffi::DataType::F64>>());

template <typename T> py::capsule EncapsulateFfiCall(T *function) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(function));
}

} // namespace

PYBIND11_MODULE(_xolky, module) {
  module.def("create_solver", [](int64_t n, int64_t nnz, int device_ordinal) {
    return Registry().Create(n, nnz, device_ordinal);
  });
  module.def("destroy_solver", [](SolverId id) { Registry().Destroy(id); });
  module.def("shutdown", []() { Registry().Clear(); });
  module.def("active_solver_count", []() { return Registry().Size(); });
  module.def("_poison_solver_for_testing", [](SolverId id) {
    auto entry = Registry().Get(id);
    std::lock_guard<std::mutex> lock(entry->mutex);
    entry->failed = true;
  });

  module.def("setup", []() { return EncapsulateFfiCall(XolkySetup); });
  module.def("refactor", []() { return EncapsulateFfiCall(XolkyRefactor); });
  module.def("solve", []() { return EncapsulateFfiCall(XolkySolve); });
}
