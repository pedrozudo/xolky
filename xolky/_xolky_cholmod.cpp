#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cholmod.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
namespace ffi = xla::ffi;

namespace {

using SolverId = std::uint64_t;

enum class OrderingPolicy { kAuto, kAmd, kMetis, kNesdis };

OrderingPolicy ParseOrdering(const std::string &ordering) {
  if (ordering == "auto") {
    return OrderingPolicy::kAuto;
  }
  if (ordering == "amd") {
    return OrderingPolicy::kAmd;
  }
  if (ordering == "metis") {
    return OrderingPolicy::kMetis;
  }
  if (ordering == "nesdis") {
    return OrderingPolicy::kNesdis;
  }
  throw std::invalid_argument(
      "ordering must be auto, amd, metis, or nesdis");
}

int ParseFactorization(const std::string &factorization) {
  if (factorization == "auto") {
    return CHOLMOD_AUTO;
  }
  if (factorization == "simplicial") {
    return CHOLMOD_SIMPLICIAL;
  }
  if (factorization == "supernodal") {
    return CHOLMOD_SUPERNODAL;
  }
  throw std::invalid_argument(
      "factorization must be auto, simplicial, or supernodal");
}

std::string CholmodStatusMessage(const cholmod_common &common,
                                 const char *operation) {
  return std::string(operation) + " failed with CHOLMOD status " +
         std::to_string(common.status);
}

struct SolverEntry {
  SolverEntry(int64_t n_value, int64_t nnz_value,
              OrderingPolicy ordering_value, int factorization_value)
      : n(n_value), nnz(nnz_value), ordering(ordering_value),
        factorization(factorization_value) {}

  SolverEntry(const SolverEntry &) = delete;
  SolverEntry &operator=(const SolverEntry &) = delete;

  void InitializeResources() {
    if (!cholmod_start(&common)) {
      throw std::runtime_error("cholmod_start failed");
    }
    common_started = true;

    if (ordering == OrderingPolicy::kAuto) {
      common.nmethods = 0;
    } else {
      common.nmethods = 1;
      if (ordering == OrderingPolicy::kAmd) {
        common.method[0].ordering = CHOLMOD_AMD;
      } else if (ordering == OrderingPolicy::kMetis) {
        common.method[0].ordering = CHOLMOD_METIS;
      } else {
        common.method[0].ordering = CHOLMOD_NESDIS;
      }
    }
    common.postorder = true;
    common.supernodal = factorization;

    // Simplicial CHOLMOD defaults to LDL', which can factor some indefinite
    // matrices. Xolky's contract is SPD Cholesky, so require an LL' result.
    common.final_asis = false;
    common.final_ll = true;
  }

  void InitializeMatrix(const int32_t *input_indices,
                        const int32_t *input_indptr) {
    if (input_indptr[0] != 0) {
      throw std::invalid_argument("csr_indptr must start with zero");
    }
    if (input_indptr[n] != nnz) {
      throw std::invalid_argument("csr_indptr must end with nnz");
    }

    indices.assign(input_indices, input_indices + nnz);
    indptr.assign(input_indptr, input_indptr + n + 1);

    for (int64_t row = 0; row < n; ++row) {
      const int32_t begin = indptr[row];
      const int32_t end = indptr[row + 1];
      if (begin < 0 || end < begin || end > nnz) {
        throw std::invalid_argument(
            "csr_indptr must be monotonic and bounded by nnz");
      }

      bool has_diagonal = false;
      int32_t previous_column = -1;
      for (int32_t offset = begin; offset < end; ++offset) {
        const int32_t column = indices[offset];
        if (column < 0 || column >= n) {
          throw std::invalid_argument("csr_indices contains an out-of-bounds "
                                      "column index");
        }
        if (column > row) {
          throw std::invalid_argument(
              "csr_indices must describe the lower triangle");
        }
        if (column <= previous_column) {
          throw std::invalid_argument(
              "column indices in each CSR row must be strictly increasing");
        }
        previous_column = column;
        has_diagonal = has_diagonal || column == row;
      }
      if (!has_diagonal) {
        throw std::invalid_argument(
            "each CSR row must contain its diagonal entry");
      }
    }

    matrix = {};
    matrix.nrow = static_cast<size_t>(n);
    matrix.ncol = static_cast<size_t>(n);
    matrix.nzmax = static_cast<size_t>(nnz);
    matrix.p = indptr.data();
    matrix.i = indices.data();
    matrix.nz = nullptr;
    matrix.x = nullptr;
    matrix.z = nullptr;
    // Lower-triangular CSR is the same memory representation as
    // upper-triangular CSC of the transpose.
    matrix.stype = 1;
    matrix.itype = CHOLMOD_INT;
    matrix.xtype = CHOLMOD_PATTERN;
    matrix.dtype = CHOLMOD_DOUBLE;
    matrix.sorted = true;
    matrix.packed = true;

    factor = cholmod_analyze(&matrix, &common);
    if (factor == nullptr || common.status < CHOLMOD_OK) {
      throw std::runtime_error(
          CholmodStatusMessage(common, "cholmod_analyze"));
    }
    initialized = true;
  }

  ~SolverEntry() noexcept {
    if (!common_started) {
      return;
    }
    if (solution != nullptr) {
      cholmod_free_dense(&solution, &common);
    }
    if (y_workspace != nullptr) {
      cholmod_free_dense(&y_workspace, &common);
    }
    if (e_workspace != nullptr) {
      cholmod_free_dense(&e_workspace, &common);
    }
    if (factor != nullptr) {
      cholmod_free_factor(&factor, &common);
    }
    cholmod_finish(&common);
  }

  int64_t n;
  int64_t nnz;
  OrderingPolicy ordering;
  int factorization;
  cholmod_common common{};
  cholmod_sparse matrix{};
  cholmod_factor *factor = nullptr;
  cholmod_dense *solution = nullptr;
  cholmod_dense *y_workspace = nullptr;
  cholmod_dense *e_workspace = nullptr;
  std::vector<int32_t> indices;
  std::vector<int32_t> indptr;
  bool common_started = false;
  bool initialized = false;
  bool factorized = false;
  bool failed = false;
  std::mutex mutex;
};

class SolverRegistry {
public:
  SolverId Create(int64_t n, int64_t nnz, OrderingPolicy ordering,
                  int factorization) {
    if (n <= 0) {
      throw std::invalid_argument("n must be positive");
    }
    if (nnz <= 0) {
      throw std::invalid_argument("nnz must be positive");
    }
    auto entry =
        std::make_shared<SolverEntry>(n, nnz, ordering, factorization);
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
  const SolverId id = *solver_id.typed_data();
  if (id == 0) {
    throw std::invalid_argument("solver identifier 0 is invalid");
  }
  return id;
}

ffi::Error SetupImpl(
    ffi::BufferR0<ffi::DataType::U64> solver_id,
    ffi::BufferR0<ffi::DataType::U8> sequence,
    ffi::BufferR1<ffi::DataType::S32> csr_indices,
    ffi::BufferR1<ffi::DataType::S32> csr_indptr,
    ffi::ResultBufferR0<ffi::DataType::U8> sequence_out) {
  try {
    auto entry = Registry().Get(ReadSolverId(solver_id));
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
    entry->InitializeMatrix(csr_indices.typed_data(), csr_indptr.typed_data());
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
        .Arg<ffi::BufferR0<ffi::DataType::U64>>()
        .Arg<ffi::BufferR0<ffi::DataType::U8>>()
        .Arg<ffi::BufferR1<ffi::DataType::S32>>()
        .Arg<ffi::BufferR1<ffi::DataType::S32>>()
        .Ret<ffi::BufferR0<ffi::DataType::U8>>());

ffi::Error RefactorImpl(
    ffi::BufferR0<ffi::DataType::U64> solver_id,
    ffi::BufferR0<ffi::DataType::U8> sequence,
    ffi::BufferR1<ffi::DataType::F64> csr_values,
    ffi::ResultBufferR0<ffi::DataType::U8> sequence_out) {
  try {
    auto entry = Registry().Get(ReadSolverId(solver_id));
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

    entry->matrix.x = csr_values.typed_data();
    entry->matrix.xtype = CHOLMOD_REAL;
    const int success =
        cholmod_factorize(&entry->matrix, entry->factor, &entry->common);
    entry->matrix.x = nullptr;
    entry->matrix.xtype = CHOLMOD_PATTERN;

    if (entry->common.status == CHOLMOD_NOT_POSDEF ||
        entry->factor->minor != static_cast<size_t>(entry->n)) {
      entry->failed = true;
      return ffi::Error::InvalidArgument(
          "matrix is not positive definite; CHOLMOD failed at column " +
          std::to_string(entry->factor->minor));
    }
    if (!success || entry->common.status < CHOLMOD_OK) {
      entry->failed = true;
      return ffi::Error::Internal(
          CholmodStatusMessage(entry->common, "cholmod_factorize"));
    }
    if (entry->common.status != CHOLMOD_OK) {
      entry->failed = true;
      return ffi::Error::Internal(
          CholmodStatusMessage(entry->common, "cholmod_factorize"));
    }
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
        .Arg<ffi::BufferR0<ffi::DataType::U64>>()
        .Arg<ffi::BufferR0<ffi::DataType::U8>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Ret<ffi::BufferR0<ffi::DataType::U8>>());

ffi::Error SolveImpl(
    ffi::BufferR0<ffi::DataType::U64> solver_id,
    ffi::BufferR0<ffi::DataType::U8> sequence,
    ffi::BufferR1<ffi::DataType::F64> right_hand_side,
    ffi::ResultBufferR0<ffi::DataType::U8> sequence_out,
    ffi::ResultBufferR1<ffi::DataType::F64> result) {
  try {
    auto entry = Registry().Get(ReadSolverId(solver_id));
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

    cholmod_dense rhs{};
    rhs.nrow = static_cast<size_t>(entry->n);
    rhs.ncol = 1;
    rhs.nzmax = static_cast<size_t>(entry->n);
    rhs.d = static_cast<size_t>(entry->n);
    rhs.x = right_hand_side.typed_data();
    rhs.z = nullptr;
    rhs.xtype = CHOLMOD_REAL;
    rhs.dtype = CHOLMOD_DOUBLE;

    const int success = cholmod_solve2(
        CHOLMOD_A, entry->factor, &rhs, nullptr, &entry->solution, nullptr,
        &entry->y_workspace, &entry->e_workspace, &entry->common);
    if (!success || entry->solution == nullptr ||
        entry->solution->x == nullptr || entry->common.status < CHOLMOD_OK) {
      entry->failed = true;
      return ffi::Error::Internal(
          CholmodStatusMessage(entry->common, "cholmod_solve2"));
    }
    std::memcpy(result->typed_data(), entry->solution->x,
                static_cast<size_t>(entry->n) * sizeof(double));
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

py::dict Capabilities() {
  int version[3] = {0, 0, 0};
  cholmod_version(version);
  py::dict result;
  result["version"] = py::make_tuple(version[0], version[1], version[2]);
  result["int32"] = true;
#if CHOLMOD_VERSION >= CHOLMOD_VER_CODE(5, 3)
  result["gpl_modules"] = cholmod_query(CHOLMOD_QUERY_HAS_GPL);
  result["supernodal"] = cholmod_query(CHOLMOD_QUERY_HAS_SUPERNODAL);
#else
  result["gpl_modules"] = py::none();
  result["supernodal"] = py::none();
#endif
  return result;
}

} // namespace

PYBIND11_MODULE(_xolky_cholmod, module) {
  module.def("create_solver", [](int64_t n, int64_t nnz, int device_ordinal,
                                 const std::string &ordering,
                                 const std::string &factorization) {
    // Logical JAX CPU devices share host-addressable memory. The ordinal is
    // retained in the common Python/native API but does not constrain CHOLMOD.
    (void)device_ordinal;
    return Registry().Create(n, nnz, ParseOrdering(ordering),
                             ParseFactorization(factorization));
  });
  module.def("destroy_solver", [](SolverId id) { Registry().Destroy(id); });
  module.def("shutdown", []() { Registry().Clear(); });
  module.def("active_solver_count", []() { return Registry().Size(); });
  module.def("capabilities", &Capabilities);
  module.def("version", []() { return Capabilities()["version"]; });
  module.def("_workspace_allocated_for_testing", [](SolverId id) {
    auto entry = Registry().Get(id);
    std::lock_guard<std::mutex> lock(entry->mutex);
    return entry->solution != nullptr;
  });
  module.def("_poison_solver_for_testing", [](SolverId id) {
    auto entry = Registry().Get(id);
    std::lock_guard<std::mutex> lock(entry->mutex);
    entry->failed = true;
  });
  module.def("setup", []() { return EncapsulateFfiCall(XolkySetup); });
  module.def("refactor", []() { return EncapsulateFfiCall(XolkyRefactor); });
  module.def("solve", []() { return EncapsulateFfiCall(XolkySolve); });
}
