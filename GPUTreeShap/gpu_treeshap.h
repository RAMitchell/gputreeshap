/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <algorithm>
#include <cub/cub.cuh>
#include <functional>
#include <iterator>
#include <set>
#include <stdexcept>
#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/system/detail/generic/select_system.h>
#include <utility>
#include <vector>

namespace gpu_treeshap {

struct XgboostSplitCondition {
  XgboostSplitCondition() = default;
  XgboostSplitCondition(float feature_lower_bound, float feature_upper_bound,
                        bool is_missing_branch)
      : feature_lower_bound(feature_lower_bound),
        feature_upper_bound(feature_upper_bound),
        is_missing_branch(is_missing_branch) {
    assert(feature_lower_bound <= feature_upper_bound);
  }

  /*! Feature values >= lower and < upper flow down this path. */
  float feature_lower_bound;
  float feature_upper_bound;
  /*! Do missing values flow down this path? */
  bool is_missing_branch;

  // Does this instance flow down this path?
  __host__ __device__ bool EvaluateSplit(float x) const {
    // is nan
    if (isnan(x)) {
      return is_missing_branch;
    }
    return x >= feature_lower_bound && x < feature_upper_bound;
  }

  // Combine two split conditions on the same feature
  __host__ __device__ void
  Merge(const XgboostSplitCondition &other) { // Combine duplicate features
    feature_lower_bound = max(feature_lower_bound, other.feature_lower_bound);
    feature_upper_bound = min(feature_upper_bound, other.feature_upper_bound);
    is_missing_branch = is_missing_branch && other.is_missing_branch;
  }
};

/*!
 * An element of a unique path through a decision tree. Can implement various
 * types of splits via the templated SplitConditionT. Some decision tree
 * implementations may wish to use double precision or single precision, some
 * may use < or <= as the threshold, missing values can be handled differently,
 * categoricals may be supported.
 *
 * \tparam  SplitConditionT A split condition implementing the methods
 * EvaluateSplit and Merge.
 */
template <typename SplitConditionT> struct PathElement {
  using split_type = SplitConditionT;
  __host__ __device__ PathElement(size_t path_idx, int64_t feature_idx,
                                  int group, SplitConditionT split_condition,
                                  double zero_fraction, float v)
      : path_idx(path_idx), feature_idx(feature_idx), group(group),
        split_condition(split_condition), zero_fraction(zero_fraction), v(v) {}

  PathElement() = default;
  __host__ __device__ bool IsRoot() const { return feature_idx == -1; }

  template <typename DatasetT>
  __host__ __device__ bool EvaluateSplit(DatasetT X, size_t row_idx) const {
    if (this->IsRoot()) {
      return 1.0;
    }
    return split_condition.EvaluateSplit(X.GetElement(row_idx, feature_idx));
  }

  /*! Unique path index. */
  size_t path_idx;
  /*! Feature of this split, -1 indicates bias term. */
  int64_t feature_idx;
  /*! Indicates class for multiclass problems. */
  int group;
  SplitConditionT split_condition;
  /*! Probability of following this path when feature_idx is not in the active
   * set. */
  double zero_fraction;
  float v; // Leaf weight at the end of the path
};

// Helper function that accepts an index into a flat contiguous array and the
// dimensions of a tensor and returns the indices with respect to the tensor
template <typename T, size_t N>
__device__ void FlatIdxToTensorIdx(T flat_idx, const T (&shape)[N],
                                   T (&out_idx)[N]) {
  T current_size = shape[0];
  for (auto i = 1ull; i < N; i++) {
    current_size *= shape[i];
  }
  for (auto i = 0ull; i < N; i++) {
    current_size /= shape[i];
    out_idx[i] = flat_idx / current_size;
    flat_idx -= current_size * out_idx[i];
  }
}

// Given a shape and coordinates into a tensor, return the index into the
// backing storage one-dimensional array
template <typename T, size_t N>
__device__ T TensorIdxToFlatIdx(const T (&shape)[N], const T (&tensor_idx)[N]) {
  T current_size = shape[0];
  for (auto i = 1ull; i < N; i++) {
    current_size *= shape[i];
  }
  T idx = 0;
  for (auto i = 0ull; i < N; i++) {
    current_size /= shape[i];
    idx += tensor_idx[i] * current_size;
  }
  return idx;
}

// Maps values to the phi array according to row, group and column
__host__ __device__ inline size_t IndexPhi(size_t row_idx, size_t num_groups,
                                           size_t group, size_t num_columns,
                                           size_t column_idx) {
  return (row_idx * num_groups + group) * (num_columns + 1) + column_idx;
}

__host__ __device__ inline size_t
IndexPhiInteractions(size_t row_idx, size_t num_groups, size_t group,
                     size_t num_columns, size_t i, size_t j) {
  size_t matrix_size = (num_columns + 1) * (num_columns + 1);
  size_t matrix_offset = (row_idx * num_groups + group) * matrix_size;
  return matrix_offset + i * (num_columns + 1) + j;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

// Convenience vector class using an execution policy
template <typename T, typename ExecutionPolicyT> class Vector {
public:
  using value_type = T;
  using ptr_t = decltype(thrust::malloc<T>(std::declval<ExecutionPolicyT>(),
                                           std::declval<std::size_t>()));

private:
  const ExecutionPolicyT &policy;
  std::size_t n;
  ptr_t ptr;

public:
  Vector(const ExecutionPolicyT &policy, std::size_t n = 0)
      : policy(policy), n(n) {
    ptr = thrust::malloc<T>(policy, n);
  }
  Vector(const ExecutionPolicyT &policy, std::size_t n, T val)
      : policy(policy), n(n) {
    ptr = thrust::malloc<T>(policy, n);
    thrust::fill(policy, this->begin(), this->end(), val);
  }
  template <typename IterT>
  Vector(const ExecutionPolicyT &policy, IterT begin, IterT end)
      : policy(policy), n(end - begin) {
    ptr = thrust::malloc<T>(policy, n);
    typedef typename thrust::iterator_system<IterT>::type System1;
    System1 system1;
    thrust::detail::two_system_copy(system1, policy, begin, end, ptr);
  }
  // construct from same policy
  Vector(const ExecutionPolicyT &policy, ptr_t begin, ptr_t end)
      : policy(policy), n(end - begin) {
    ptr = thrust::malloc<T>(policy, n);
    thrust::copy(policy, begin, end, ptr);
  }
  ~Vector() { thrust::free(policy, ptr); }
  ptr_t begin() { return ptr; }
  ptr_t end() { return ptr + n; }
  ptr_t begin() const { return ptr; }
  ptr_t end() const { return ptr + n; }
  ptr_t data() { return ptr; }
  const ptr_t data() const { return ptr; }
  std::size_t size() const { return n; }
  void resize(std::size_t n) {
    if (n < this->size()) {
      this->n = n;
    }
    if (n > this->size()) {
      auto new_ptr = thrust::malloc<T>(policy, n);
      thrust::copy(policy, this->begin(), this->end(), new_ptr);
      thrust::free(policy, ptr);
      ptr = new_ptr;
      this->n = n;
    }
  }
};

// Shorthand for creating a device vector with an appropriate allocator type
template <class T, class DeviceAllocatorT>
using RebindVector =
    thrust::device_vector<T,
                          typename DeviceAllocatorT::template rebind<T>::other>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 || defined(__clang__)
__device__ __forceinline__ double atomicAddDouble(double *address, double val) {
  return atomicAdd(address, val);
}
#else // In device code and CUDA < 600
__device__ __forceinline__ double atomicAddDouble(double *address,
                                                  double val) { // NOLINT
  unsigned long long int *address_as_ull =                      // NOLINT
      (unsigned long long int *)address;                        // NOLINT
  unsigned long long int old = *address_as_ull, assumed;        // NOLINT

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

__forceinline__ __device__ unsigned int lanemask32_lt() {
  unsigned int lanemask32_lt;
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
  return (lanemask32_lt);
}

// Like a coalesced group, except we can make the assumption that all threads in
// a group are next to each other. This makes shuffle operations much cheaper.
class ContiguousGroup {
public:
  __device__ ContiguousGroup(uint32_t mask) : mask_(mask) {}

  __device__ uint32_t size() const { return __popc(mask_); }
  __device__ uint32_t thread_rank() const {
    return __popc(mask_ & lanemask32_lt());
  }
  template <typename T> __device__ T shfl(T val, uint32_t src) const {
    return __shfl_sync(mask_, val, src + __ffs(mask_) - 1);
  }
  template <typename T> __device__ T shfl_up(T val, uint32_t delta) const {
    return __shfl_up_sync(mask_, val, delta);
  }
  __device__ uint32_t ballot(int predicate) const {
    return __ballot_sync(mask_, predicate) >> (__ffs(mask_) - 1);
  }

  template <typename T, typename OpT> __device__ T reduce(T val, OpT op) {
    for (int i = 1; i < this->size(); i *= 2) {
      T shfl = shfl_up(val, i);
      if (static_cast<int>(thread_rank()) - i >= 0) {
        val = op(val, shfl);
      }
    }
    return shfl(val, size() - 1);
  }
  uint32_t mask_;
};

// Separate the active threads by labels
// This functionality is available in cuda 11.0 on cc >=7.0
// We reimplement for backwards compatibility
// Assumes partitions are contiguous
inline __device__ ContiguousGroup active_labeled_partition(uint32_t mask,
                                                           int label) {
#if __CUDA_ARCH__ >= 700
  uint32_t subgroup_mask = __match_any_sync(mask, label);
#else
  uint32_t subgroup_mask = 0;
  for (int i = 0; i < 32;) {
    int current_label = __shfl_sync(mask, label, i);
    uint32_t ballot = __ballot_sync(mask, label == current_label);
    if (label == current_label) {
      subgroup_mask = ballot;
    }
    uint32_t completed_mask =
        (1 << (32 - __clz(ballot))) - 1; // Threads that have finished
    // Find the start of the next group, mask off completed threads from active
    // threads Then use ffs - 1 to find the position of the next group
    int next_i = __ffs(mask & ~completed_mask) - 1;
    if (next_i == -1)
      break;            // -1 indicates all finished
    assert(next_i > i); // Prevent infinite loops when the constraints not met
    i = next_i;
  }
#endif
  return ContiguousGroup(subgroup_mask);
}

// Group of threads where each thread holds a path element
class GroupPath {
protected:
  const ContiguousGroup &g_;
  // These are combined so we can communicate them in a single 64 bit shuffle
  // instruction
  float zero_one_fraction_[2];
  float pweight_;
  int unique_depth_;

public:
  __device__ GroupPath(const ContiguousGroup &g, float zero_fraction,
                       float one_fraction)
      : g_(g), zero_one_fraction_{zero_fraction, one_fraction},
        pweight_(g.thread_rank() == 0 ? 1.0f : 0.0f), unique_depth_(0) {}

  // Cooperatively extend the path with a group of threads
  // Each thread maintains pweight for its path element in register
  __device__ void Extend() {
    unique_depth_++;

    // Broadcast the zero and one fraction from the newly added path element
    // Combine 2 shuffle operations into 64 bit word
    const size_t rank = g_.thread_rank();
    const float inv_unique_depth =
        __fdividef(1.0f, static_cast<float>(unique_depth_ + 1));
    uint64_t res = g_.shfl(*reinterpret_cast<uint64_t *>(&zero_one_fraction_),
                           unique_depth_);
    const float new_zero_fraction = reinterpret_cast<float *>(&res)[0];
    const float new_one_fraction = reinterpret_cast<float *>(&res)[1];
    float left_pweight = g_.shfl_up(pweight_, 1);

    // pweight of threads with rank < unique_depth_ is 0
    // We use max(x,0) to avoid using a branch
    // pweight_ *=
    // new_zero_fraction * max(unique_depth_ - rank, 0llu) * inv_unique_depth;
    pweight_ = __fmul_rn(
        __fmul_rn(pweight_, new_zero_fraction),
        __fmul_rn(max(unique_depth_ - rank, size_t(0)), inv_unique_depth));

    // pweight_  += new_one_fraction * left_pweight * rank * inv_unique_depth;
    pweight_ = __fmaf_rn(__fmul_rn(new_one_fraction, left_pweight),
                         __fmul_rn(rank, inv_unique_depth), pweight_);
  }

  // Each thread unwinds the path for its feature and returns the sum
  __device__ float UnwoundPathSum() {
    float next_one_portion = g_.shfl(pweight_, unique_depth_);
    float total = 0.0f;
    const float zero_frac_div_unique_depth = __fdividef(
        zero_one_fraction_[0], static_cast<float>(unique_depth_ + 1));
    for (int i = unique_depth_ - 1; i >= 0; i--) {
      float ith_pweight = g_.shfl(pweight_, i);
      float precomputed =
          __fmul_rn((unique_depth_ - i), zero_frac_div_unique_depth);
      const float tmp =
          __fdividef(__fmul_rn(next_one_portion, unique_depth_ + 1), i + 1);
      total = __fmaf_rn(tmp, zero_one_fraction_[1], total);
      next_one_portion = __fmaf_rn(-tmp, precomputed, ith_pweight);
      float numerator =
          __fmul_rn(__fsub_rn(1.0f, zero_one_fraction_[1]), ith_pweight);
      if (precomputed > 0.0f) {
        total += __fdividef(numerator, precomputed);
      }
    }

    return total;
  }
};

// Has different permutation weightings to the above
// Used in Taylor Shapley interaction index
class TaylorGroupPath : GroupPath {
public:
  __device__ TaylorGroupPath(const ContiguousGroup &g, float zero_fraction,
                             float one_fraction)
      : GroupPath(g, zero_fraction, one_fraction) {}

  // Extend the path is normal, all reweighting can happen in UnwoundPathSum
  __device__ void Extend() { GroupPath::Extend(); }

  // Each thread unwinds the path for its feature and returns the sum
  // We use a different permutation weighting for Taylor interactions
  // As if the total number of features was one larger
  __device__ float UnwoundPathSum() {
    float one_fraction = zero_one_fraction_[1];
    float zero_fraction = zero_one_fraction_[0];
    float next_one_portion = g_.shfl(pweight_, unique_depth_) /
                             static_cast<float>(unique_depth_ + 2);

    float total = 0.0f;
    for (int i = unique_depth_ - 1; i >= 0; i--) {
      float ith_pweight =
          g_.shfl(pweight_, i) * (static_cast<float>(unique_depth_ - i + 1) /
                                  static_cast<float>(unique_depth_ + 2));
      if (one_fraction > 0.0f) {
        const float tmp =
            next_one_portion * (unique_depth_ + 2) / ((i + 1) * one_fraction);

        total += tmp;
        next_one_portion =
            ith_pweight - tmp * zero_fraction *
                              ((unique_depth_ - i + 1) /
                               static_cast<float>(unique_depth_ + 2));
      } else if (zero_fraction > 0.0f) {
        total +=
            (ith_pweight / zero_fraction) /
            ((unique_depth_ - i + 1) / static_cast<float>(unique_depth_ + 2));
      }
    }

    return 2 * total;
  }
};

template <typename DatasetT, typename SplitConditionT>
__device__ float ComputePhi(const PathElement<SplitConditionT> &e,
                            size_t row_idx, const DatasetT &X,
                            const ContiguousGroup &group, float zero_fraction) {
  float one_fraction = e.EvaluateSplit(X, row_idx);
  GroupPath path(group, zero_fraction, one_fraction);
  size_t unique_path_length = group.size();

  // Extend the path
  for (auto unique_depth = 1ull; unique_depth < unique_path_length;
       unique_depth++) {
    path.Extend();
  }

  float sum = path.UnwoundPathSum();
  return sum * (one_fraction - zero_fraction) * e.v;
}

inline __host__ __device__ size_t DivRoundUp(size_t a, size_t b) {
  return (a + b - 1) / b;
}

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
void __device__
ConfigureThread(const DatasetT &X, const size_t bins_per_row,
                const PathElement<SplitConditionT> *path_elements,
                const size_t *bin_segments, size_t *start_row, size_t *end_row,
                PathElement<SplitConditionT> *e, bool *thread_active) {
  // Partition work
  // Each warp processes a set of training instances applied to a path
  size_t tid = kBlockSize * blockIdx.x + threadIdx.x;
  const size_t warp_size = 32;
  size_t warp_rank = tid / warp_size;
  if (warp_rank >= bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp)) {
    *thread_active = false;
    return;
  }
  size_t bin_idx = warp_rank % bins_per_row;
  size_t bank = warp_rank / bins_per_row;
  size_t path_start = bin_segments[bin_idx];
  size_t path_end = bin_segments[bin_idx + 1];
  uint32_t thread_rank = threadIdx.x % warp_size;
  if (thread_rank >= path_end - path_start) {
    *thread_active = false;
  } else {
    *e = path_elements[path_start + thread_rank];
    *start_row = bank * kRowsPerWarp;
    *end_row = min((bank + 1) * kRowsPerWarp, X.NumRows());
    *thread_active = true;
  }
}

#define GPUTREESHAP_MAX_THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
__global__ void __launch_bounds__(GPUTREESHAP_MAX_THREADS_PER_BLOCK)
    ShapKernel(DatasetT X, size_t bins_per_row,
               const PathElement<SplitConditionT> *path_elements,
               const size_t *bin_segments, size_t num_groups, double *phis) {
  // Use shared memory for structs, otherwise nvcc puts in local memory
  __shared__ DatasetT s_X;
  s_X = X;
  __shared__ PathElement<SplitConditionT> s_elements[kBlockSize];
  PathElement<SplitConditionT> &e = s_elements[threadIdx.x];

  size_t start_row, end_row;
  bool thread_active;
  ConfigureThread<DatasetT, kBlockSize, kRowsPerWarp>(
      s_X, bins_per_row, path_elements, bin_segments, &start_row, &end_row, &e,
      &thread_active);
  uint32_t mask = __ballot_sync(FULL_MASK, thread_active);
  if (!thread_active)
    return;

  float zero_fraction = e.zero_fraction;
  auto labelled_group = active_labeled_partition(mask, e.path_idx);

  for (int64_t row_idx = start_row; row_idx < end_row; row_idx++) {
    float phi = ComputePhi(e, row_idx, X, labelled_group, zero_fraction);

    if (!e.IsRoot()) {
      atomicAddDouble(&phis[IndexPhi(row_idx, num_groups, e.group, X.NumCols(),
                                     e.feature_idx)],
                      phi);
    }
  }
}

template <typename DatasetT, typename SegmentVectorT, typename PathVectorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void ComputeShap(DatasetT X, const SegmentVectorT &bin_segments,
                 const PathVectorT &path_elements, size_t num_groups,
                 double *phis, const ExecutionPolicyT &policy) {
  size_t bins_per_row = bin_segments.size() - 1;
  const int kBlockThreads = GPUTREESHAP_MAX_THREADS_PER_BLOCK;
  const int warps_per_block = kBlockThreads / 32;
  const int kRowsPerWarp = 1024;
  size_t warps_needed = bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp);

  const uint32_t grid_size = DivRoundUp(warps_needed, warps_per_block);

  ShapKernel<DatasetT, kBlockThreads, kRowsPerWarp>
      <<<grid_size, kBlockThreads, 0>>>(
          X, bins_per_row, path_elements.data().get(),
          bin_segments.data().get(), num_groups, phis);
}

template <typename PathT, typename DatasetT, typename SplitConditionT>
__device__ float ComputePhiCondition(const PathElement<SplitConditionT> &e,
                                     size_t row_idx, const DatasetT &X,
                                     const ContiguousGroup &group,
                                     int64_t condition_feature) {
  float one_fraction = e.EvaluateSplit(X, row_idx);
  PathT path(group, e.zero_fraction, one_fraction);
  size_t unique_path_length = group.size();
  float condition_on_fraction = 1.0f;
  float condition_off_fraction = 1.0f;

  // Extend the path
  for (auto i = 1ull; i < unique_path_length; i++) {
    bool is_condition_feature =
        group.shfl(e.feature_idx, i) == condition_feature;
    float o_i = group.shfl(one_fraction, i);
    float z_i = group.shfl(e.zero_fraction, i);

    if (is_condition_feature) {
      condition_on_fraction = o_i;
      condition_off_fraction = z_i;
    } else {
      path.Extend();
    }
  }
  float sum = path.UnwoundPathSum();
  if (e.feature_idx == condition_feature) {
    return 0.0f;
  }
  float phi = sum * (one_fraction - e.zero_fraction) * e.v;
  return phi * (condition_on_fraction - condition_off_fraction) * 0.5f;
}

// If there is a feature in the path we are conditioning on, swap it to the end
// of the path
template <typename SplitConditionT>
inline __device__ void
SwapConditionedElement(PathElement<SplitConditionT> **e,
                       PathElement<SplitConditionT> *s_elements,
                       uint32_t condition_rank, const ContiguousGroup &group) {
  auto last_rank = group.size() - 1;
  auto this_rank = group.thread_rank();
  if (this_rank == last_rank) {
    *e = &s_elements[(threadIdx.x - this_rank) + condition_rank];
  } else if (this_rank == condition_rank) {
    *e = &s_elements[(threadIdx.x - this_rank) + last_rank];
  }
}

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
__global__ void __launch_bounds__(GPUTREESHAP_MAX_THREADS_PER_BLOCK)
    ShapInteractionsKernel(DatasetT X, size_t bins_per_row,
                           const PathElement<SplitConditionT> *path_elements,
                           const size_t *bin_segments, size_t num_groups,
                           double *phis_interactions) {
  // Use shared memory for structs, otherwise nvcc puts in local memory
  __shared__ DatasetT s_X;
  s_X = X;
  __shared__ PathElement<SplitConditionT> s_elements[kBlockSize];
  PathElement<SplitConditionT> *e = &s_elements[threadIdx.x];

  size_t start_row, end_row;
  bool thread_active;
  ConfigureThread<DatasetT, kBlockSize, kRowsPerWarp>(
      s_X, bins_per_row, path_elements, bin_segments, &start_row, &end_row, e,
      &thread_active);
  uint32_t mask = __ballot_sync(FULL_MASK, thread_active);
  if (!thread_active)
    return;

  auto labelled_group = active_labeled_partition(mask, e->path_idx);

  for (int64_t row_idx = start_row; row_idx < end_row; row_idx++) {
    float phi = ComputePhi(*e, row_idx, X, labelled_group, e->zero_fraction);
    if (!e->IsRoot()) {
      auto phi_offset =
          IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                               e->feature_idx, e->feature_idx);
      atomicAddDouble(phis_interactions + phi_offset, phi);
    }

    for (auto condition_rank = 1ull; condition_rank < labelled_group.size();
         condition_rank++) {
      e = &s_elements[threadIdx.x];
      int64_t condition_feature =
          labelled_group.shfl(e->feature_idx, condition_rank);
      SwapConditionedElement(&e, s_elements, condition_rank, labelled_group);
      float x = ComputePhiCondition<GroupPath>(*e, row_idx, X, labelled_group,
                                               condition_feature);
      if (!e->IsRoot()) {
        auto phi_offset =
            IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                                 e->feature_idx, condition_feature);
        atomicAddDouble(phis_interactions + phi_offset, x);
        // Subtract effect from diagonal
        auto phi_diag =
            IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                                 e->feature_idx, e->feature_idx);
        atomicAddDouble(phis_interactions + phi_diag, -x);
      }
    }
  }
}

template <typename DatasetT, typename SegmentVectorT, typename PathVectorT>
void ComputeShapInteractions(DatasetT X, const SegmentVectorT &bin_segments,
                             const PathVectorT &path_elements,
                             size_t num_groups, double *phis) {
  size_t bins_per_row = bin_segments.size() - 1;
  const int kBlockThreads = GPUTREESHAP_MAX_THREADS_PER_BLOCK;
  const int warps_per_block = kBlockThreads / 32;
  const int kRowsPerWarp = 100;
  size_t warps_needed = bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp);

  const uint32_t grid_size = DivRoundUp(warps_needed, warps_per_block);

  ShapInteractionsKernel<DatasetT, kBlockThreads, kRowsPerWarp>
      <<<grid_size, kBlockThreads>>>(
          X, bins_per_row, path_elements.data().get(),
          bin_segments.data().get(), num_groups, phis);
}

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
__global__ void __launch_bounds__(GPUTREESHAP_MAX_THREADS_PER_BLOCK)
    ShapTaylorInteractionsKernel(
        DatasetT X, size_t bins_per_row,
        const PathElement<SplitConditionT> *path_elements,
        const size_t *bin_segments, size_t num_groups,
        double *phis_interactions) {
  // Use shared memory for structs, otherwise nvcc puts in local memory
  __shared__ DatasetT s_X;
  if (threadIdx.x == 0) {
    s_X = X;
  }
  __syncthreads();
  __shared__ PathElement<SplitConditionT> s_elements[kBlockSize];
  PathElement<SplitConditionT> *e = &s_elements[threadIdx.x];

  size_t start_row, end_row;
  bool thread_active;
  ConfigureThread<DatasetT, kBlockSize, kRowsPerWarp>(
      s_X, bins_per_row, path_elements, bin_segments, &start_row, &end_row, e,
      &thread_active);
  uint32_t mask = __ballot_sync(FULL_MASK, thread_active);
  if (!thread_active)
    return;

  auto labelled_group = active_labeled_partition(mask, e->path_idx);

  for (int64_t row_idx = start_row; row_idx < end_row; row_idx++) {
    for (auto condition_rank = 1ull; condition_rank < labelled_group.size();
         condition_rank++) {
      e = &s_elements[threadIdx.x];
      // Compute the diagonal terms
      // TODO(Rory): this can be more efficient
      float reduce_input =
          e->IsRoot() || labelled_group.thread_rank() == condition_rank
              ? 1.0f
              : e->zero_fraction;
      float reduce =
          labelled_group.reduce(reduce_input, thrust::multiplies<float>());
      if (labelled_group.thread_rank() == condition_rank) {
        float one_fraction = e->split_condition.EvaluateSplit(
            X.GetElement(row_idx, e->feature_idx));
        auto phi_offset =
            IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                                 e->feature_idx, e->feature_idx);
        atomicAddDouble(phis_interactions + phi_offset,
                        reduce * (one_fraction - e->zero_fraction) * e->v);
      }

      int64_t condition_feature =
          labelled_group.shfl(e->feature_idx, condition_rank);

      SwapConditionedElement(&e, s_elements, condition_rank, labelled_group);

      float x = ComputePhiCondition<TaylorGroupPath>(
          *e, row_idx, X, labelled_group, condition_feature);
      if (!e->IsRoot()) {
        auto phi_offset =
            IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                                 e->feature_idx, condition_feature);
        atomicAddDouble(phis_interactions + phi_offset, x);
      }
    }
  }
}

template <typename DatasetT, typename SegmentVectorT, typename PathVectorT>
void ComputeShapTaylorInteractions(DatasetT X,
                                   const SegmentVectorT &bin_segments,
                                   const PathVectorT &path_elements,
                                   size_t num_groups, double *phis) {
  size_t bins_per_row = bin_segments.size() - 1;
  const int kBlockThreads = GPUTREESHAP_MAX_THREADS_PER_BLOCK;
  const int warps_per_block = kBlockThreads / 32;
  const int kRowsPerWarp = 100;
  size_t warps_needed = bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp);

  const uint32_t grid_size = DivRoundUp(warps_needed, warps_per_block);

  ShapTaylorInteractionsKernel<DatasetT, kBlockThreads, kRowsPerWarp>
      <<<grid_size, kBlockThreads>>>(
          X, bins_per_row, path_elements.data().get(),
          bin_segments.data().get(), num_groups, phis);
}

inline __host__ __device__ int64_t Factorial(int64_t x) {
  int64_t y = 1;
  for (auto i = 2; i <= x; i++) {
    y *= i;
  }
  return y;
}

// Compute factorials in log space using lgamma to avoid overflow
inline __host__ __device__ double W(double s, double n) {
  assert(n - s - 1 >= 0);
  return exp(lgamma(s + 1) - lgamma(n + 1) + lgamma(n - s));
}

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
__global__ void __launch_bounds__(GPUTREESHAP_MAX_THREADS_PER_BLOCK)
    ShapInterventionalKernel(DatasetT X, DatasetT R, size_t bins_per_row,
                             const PathElement<SplitConditionT> *path_elements,
                             const size_t *bin_segments, size_t num_groups,
                             double *phis) {
  // Cache W coefficients
  __shared__ float s_W[33][33];
  for (int i = threadIdx.x; i < 33 * 33; i += kBlockSize) {
    auto s = i % 33;
    auto n = i / 33;
    if (n - s - 1 >= 0) {
      s_W[s][n] = W(s, n);
    } else {
      s_W[s][n] = 0.0;
    }
  }

  __syncthreads();

  __shared__ PathElement<SplitConditionT> s_elements[kBlockSize];
  PathElement<SplitConditionT> &e = s_elements[threadIdx.x];

  size_t start_row, end_row;
  bool thread_active;
  ConfigureThread<DatasetT, kBlockSize, kRowsPerWarp>(
      X, bins_per_row, path_elements, bin_segments, &start_row, &end_row, &e,
      &thread_active);

  uint32_t mask = __ballot_sync(FULL_MASK, thread_active);
  if (!thread_active)
    return;

  auto labelled_group = active_labeled_partition(mask, e.path_idx);

  for (int64_t x_idx = start_row; x_idx < end_row; x_idx++) {
    float result = 0.0f;
    bool x_cond = e.EvaluateSplit(X, x_idx);
    uint32_t x_ballot = labelled_group.ballot(x_cond);
    for (int64_t r_idx = 0; r_idx < R.NumRows(); r_idx++) {
      bool r_cond = e.EvaluateSplit(R, r_idx);
      uint32_t r_ballot = labelled_group.ballot(r_cond);
      assert(!e.IsRoot() ||
             (x_cond == r_cond)); // These should be the same for the root
      uint32_t s = __popc(x_ballot & ~r_ballot);
      uint32_t n = __popc(x_ballot ^ r_ballot);
      float tmp = 0.0f;
      // Theorem 1
      if (x_cond && !r_cond) {
        tmp += s_W[s - 1][n];
      }
      tmp -= s_W[s][n] * (r_cond && !x_cond);

      // No foreground samples make it to this leaf, increment bias
      if (e.IsRoot() && s == 0) {
        tmp += 1.0f;
      }
      // If neither foreground or background go down this path, ignore this path
      bool reached_leaf = !labelled_group.ballot(!x_cond && !r_cond);
      tmp *= reached_leaf;
      result += tmp;
    }

    if (result != 0.0) {
      result /= R.NumRows();

      // Root writes bias
      auto feature = e.IsRoot() ? X.NumCols() : e.feature_idx;
      atomicAddDouble(
          &phis[IndexPhi(x_idx, num_groups, e.group, X.NumCols(), feature)],
          result * e.v);
    }
  }
}

template <typename DatasetT, typename SegmentVectorT, typename PathVectorT>
void ComputeShapInterventional(DatasetT X, DatasetT R,
                               const SegmentVectorT &bin_segments,
                               const PathVectorT &path_elements,
                               size_t num_groups, double *phis) {
  size_t bins_per_row = bin_segments.size() - 1;
  const int kBlockThreads = GPUTREESHAP_MAX_THREADS_PER_BLOCK;
  const int warps_per_block = kBlockThreads / 32;
  const int kRowsPerWarp = 100;
  size_t warps_needed = bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp);

  const uint32_t grid_size = DivRoundUp(warps_needed, warps_per_block);

  ShapInterventionalKernel<DatasetT, kBlockThreads, kRowsPerWarp>
      <<<grid_size, kBlockThreads>>>(
          X, R, bins_per_row, path_elements.data().get(),
          bin_segments.data().get(), num_groups, phis);
}

template <typename PathVectorT, typename MapVectorT, typename SegmentVectorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void GetBinSegments(const ExecutionPolicyT &policy, const PathVectorT &paths,
                    const MapVectorT &bin_map, SegmentVectorT *bin_segments) {
  size_t num_bins = thrust::reduce(policy, bin_map.begin(), bin_map.end(),
                                   size_t(0), thrust::maximum<size_t>()) +
                    1;
  bin_segments->resize(num_bins + 1);
  auto d_bin_map = bin_map.data();
  auto constant = thrust::make_constant_iterator(1llu);
  auto discard = thrust::make_discard_iterator();
  thrust::reduce_by_key(policy, paths.begin(), paths.end(), constant, discard,
                        bin_segments->begin(), [=] __device__(auto a, auto b) {
                          return d_bin_map[a.path_idx] == d_bin_map[b.path_idx];
                        });
  thrust::exclusive_scan(policy, bin_segments->begin(), bin_segments->end(),
                         bin_segments->begin());
}

template <typename path_t>
struct DeduplicateKeyTransformOp
    : public thrust::unary_function<path_t, thrust::pair<size_t, int64_t>> {
  __host__ __device__ thrust::pair<size_t, int64_t>
  operator()(const path_t &e) {
    return {e.path_idx, e.feature_idx};
  }
};
template <typename path_t> struct MergeOp {
  __host__ __device__ path_t
  operator()(path_t a, const path_t &b) { // Combine duplicate features
    a.split_condition.Merge(b.split_condition);
    a.zero_fraction *= b.zero_fraction;
    return a;
  }
};

template <typename PathVectorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void DeduplicatePaths(const ExecutionPolicyT &policy, PathVectorT *device_paths,
                      PathVectorT *deduplicated_paths) {
  // Sort by feature
  thrust::sort(policy, device_paths->begin(), device_paths->end(),
               [=] __device__(const auto &a, const auto &b) {
                 if (a.path_idx < b.path_idx)
                   return true;
                 if (b.path_idx < a.path_idx)
                   return false;

                 if (a.feature_idx < b.feature_idx)
                   return true;
                 if (b.feature_idx < a.feature_idx)
                   return false;
                 return false;
               });

  deduplicated_paths->resize(device_paths->size());

  using path_t = typename PathVectorT::value_type;
  auto key_transform = thrust::make_transform_iterator(
      device_paths->begin(), DeduplicateKeyTransformOp<path_t>());
  thrust::equal_to<thrust::pair<size_t, int64_t>> key_compare;
  MergeOp<path_t> op;
  auto discard = thrust::make_discard_iterator();
  auto end = thrust::reduce_by_key(
      policy, key_transform, key_transform + device_paths->size(),
      device_paths->begin(), discard, deduplicated_paths->begin(), key_compare,
      op);

  deduplicated_paths->resize(end.second - deduplicated_paths->begin());
}

template <typename PathVectorT, typename SizeVectorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void SortPaths(const ExecutionPolicyT &policy, PathVectorT *paths,
               const SizeVectorT &bin_map) {

  using path_t = typename PathVectorT::value_type;
  auto d_bin_map = bin_map.data();
  thrust::sort(policy, paths->begin(), paths->end(),
               [=] __device__(const path_t &a, const path_t &b) {
                 size_t a_bin = d_bin_map[a.path_idx];
                 size_t b_bin = d_bin_map[b.path_idx];
                 if (a_bin < b_bin)
                   return true;
                 if (b_bin < a_bin)
                   return false;

                 if (a.path_idx < b.path_idx)
                   return true;
                 if (b.path_idx < a.path_idx)
                   return false;

                 if (a.feature_idx < b.feature_idx)
                   return true;
                 if (b.feature_idx < a.feature_idx)
                   return false;
                 return false;
               });
}

using kv = std::pair<size_t, int>;

struct BFDCompare {
  bool operator()(const kv &lhs, const kv &rhs) const {
    if (lhs.second == rhs.second) {
      return lhs.first < rhs.first;
    }
    return lhs.second < rhs.second;
  }
};

// Best Fit Decreasing bin packing
// Efficient O(nlogn) implementation with balanced tree using std::set
template <typename IntVectorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
std::vector<size_t>
BFDBinPacking(const IntVectorT &counts, int bin_limit,
              const ExecutionPolicyT &policy = thrust::device) {
  std::vector<int> counts_host(counts.size());
  thrust::detail::two_system_copy(policy, thrust::host, counts.begin(),
                                  counts.end(), counts_host.begin());

  std::vector<kv> path_lengths(counts_host.size());
  for (auto i = 0ull; i < counts_host.size(); i++) {
    path_lengths[i] = {i, counts_host[i]};
  }

  std::sort(path_lengths.begin(), path_lengths.end(),
            [&](const kv &a, const kv &b) {
              std::greater<> op;
              return op(a.second, b.second);
            });

  // map unique_id -> bin
  std::vector<size_t> bin_map(counts_host.size());
  std::set<kv, BFDCompare> bin_capacities;
  bin_capacities.insert({bin_capacities.size(), bin_limit});
  for (auto pair : path_lengths) {
    int new_size = pair.second;
    auto itr = bin_capacities.lower_bound({0, new_size});
    // Does not fit in any bin
    if (itr == bin_capacities.end()) {
      size_t new_bin_idx = bin_capacities.size();
      bin_capacities.insert({new_bin_idx, bin_limit - new_size});
      bin_map[pair.first] = new_bin_idx;
    } else {
      kv entry = *itr;
      entry.second -= new_size;
      bin_map[pair.first] = entry.first;
      bin_capacities.erase(itr);
      bin_capacities.insert(entry);
    }
  }

  return bin_map;
}

template <typename PathVectorT, typename LengthVectorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void GetPathLengths(const ExecutionPolicyT &policy,
                    const PathVectorT &device_paths,
                    LengthVectorT *path_lengths) {
  auto constant = thrust::make_constant_iterator(1llu);
  auto discard = thrust::make_discard_iterator();
  auto result = thrust::reduce_by_key(
      policy, device_paths.begin(), device_paths.end(), constant, discard,
      path_lengths->begin(),
      [=] __device__(auto a, auto b) { return a.path_idx == b.path_idx; });

  path_lengths->resize(result.second - path_lengths->begin());
}

struct PathTooLongOp {
  __device__ size_t operator()(size_t length) { return length > 32; }
};

template <typename SplitConditionT> struct IncorrectVOp {
  const PathElement<SplitConditionT> *paths;
  __device__ size_t operator()(size_t idx) {
    auto a = paths[idx - 1];
    auto b = paths[idx];
    return a.path_idx == b.path_idx && a.v != b.v;
  }
};

template <typename PathVectorT, typename LengthVectorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void ValidatePaths(const ExecutionPolicyT &policy,
                   const PathVectorT &device_paths,
                   const LengthVectorT &path_lengths) {
  PathTooLongOp too_long_op;
  auto invalid_length = thrust::any_of(policy, path_lengths.begin(),
                                       path_lengths.end(), too_long_op);

  if (invalid_length) {
    throw std::invalid_argument("Tree depth must be < 32");
  }

  using split_t = typename PathVectorT::value_type::split_type;
  IncorrectVOp<split_t> incorrect_v_op{device_paths.data().get()};
  auto counting = thrust::counting_iterator<size_t>(0);
  auto incorrect_v = thrust::any_of(
      policy, counting + 1, counting + device_paths.size(), incorrect_v_op);

  if (incorrect_v) {
    throw std::invalid_argument(
        "Leaf value v should be the same across a single path");
  }
}

template <typename PathVectorT, typename SizeVectorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void PreprocessPaths(PathVectorT *device_paths, PathVectorT *deduplicated_paths,
                     SizeVectorT *bin_segments,
                     const ExecutionPolicyT &policy = thrust::device) {
  // Sort paths by length and feature
  using path_t = typename PathVectorT::value_type;
  detail::DeduplicatePaths(policy, device_paths, deduplicated_paths);
  Vector<int, ExecutionPolicyT> path_lengths(policy, device_paths->size());
  detail::GetPathLengths(policy, *deduplicated_paths, &path_lengths);
  auto cpu_bin_map = detail::BFDBinPacking(path_lengths, 32, policy);
  Vector<std::size_t, ExecutionPolicyT> device_bin_map(
      policy, cpu_bin_map.begin(), cpu_bin_map.end());
  detail::ValidatePaths(policy, *deduplicated_paths, path_lengths);
  detail::SortPaths(policy, deduplicated_paths, device_bin_map);
  detail::GetBinSegments(policy, *deduplicated_paths, device_bin_map,
                         bin_segments);
}

struct PathIdxTransformOp {
  template <typename SplitConditionT>
  __device__ size_t operator()(const PathElement<SplitConditionT> &e) {
    return e.path_idx;
  }
};

struct GroupIdxTransformOp {
  template <typename SplitConditionT>
  __device__ size_t operator()(const PathElement<SplitConditionT> &e) {
    return e.group;
  }
};

struct BiasTransformOp {
  template <typename SplitConditionT>
  __device__ double operator()(const PathElement<SplitConditionT> &e) {
    return e.zero_fraction * e.v;
  }
};

// While it is possible to compute bias in the primary kernel, we do it here
// using double precision to avoid numerical stability issues
template <typename PathVectorT, typename DoubleVectorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void ComputeBias(const PathVectorT &device_paths, DoubleVectorT *bias,
                 const ExecutionPolicyT &policy = thrust::device) {

  using path_t = typename PathVectorT::value_type;
  Vector<path_t, ExecutionPolicyT> sorted_paths(policy, device_paths.begin(),
                                                device_paths.end());
  // Make sure groups are contiguous
  thrust::sort(policy, sorted_paths.begin(), sorted_paths.end(),
               [=] __device__(const path_t &a, const path_t &b) {
                 if (a.group < b.group)
                   return true;
                 if (b.group < a.group)
                   return false;

                 if (a.path_idx < b.path_idx)
                   return true;
                 if (b.path_idx < a.path_idx)
                   return false;

                 return false;
               });
  // Combine zero fraction for all paths
  auto path_key = thrust::make_transform_iterator(sorted_paths.begin(),
                                                  PathIdxTransformOp());
  Vector<path_t, ExecutionPolicyT> combined(policy, sorted_paths.size());
  auto combined_out = thrust::reduce_by_key(
      policy, path_key, path_key + sorted_paths.size(), sorted_paths.begin(),
      thrust::make_discard_iterator(), combined.begin(),
      thrust::equal_to<size_t>(),
      [=] __host__ __device__(path_t a, const path_t &b) -> path_t {
        a.zero_fraction *= b.zero_fraction;
        return a;
      });
  size_t num_paths = combined_out.second - combined.begin();
  // Combine bias for each path, over each group
  Vector<std::size_t, ExecutionPolicyT> keys_out(policy, num_paths);
  Vector<double, ExecutionPolicyT> values_out(policy, num_paths);
  auto group_key =
      thrust::make_transform_iterator(combined.begin(), GroupIdxTransformOp());
  auto values =
      thrust::make_transform_iterator(combined.begin(), BiasTransformOp());

  auto out_itr =
      thrust::reduce_by_key(policy, group_key, group_key + num_paths, values,
                            keys_out.begin(), values_out.begin());

  // Write result
  size_t n = out_itr.first - keys_out.begin();
  auto counting = thrust::make_counting_iterator(0llu);
  auto d_keys_out = keys_out.data().get();
  auto d_values_out = values_out.data().get();
  auto d_bias = bias->data().get();
  thrust::for_each_n(policy, counting, n, [=] __device__(size_t idx) {
    d_bias[d_keys_out[idx]] = d_values_out[idx];
  });
}

}; // namespace detail

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/** \defgroup GPUTreeShap
 *  @{
 */

template <typename ExecutionPolicyT, typename path_t> class Cache {
public:
  template <typename PathIteratorT>
  Cache(const ExecutionPolicyT &policy, PathIteratorT begin,
                    PathIteratorT end, size_t num_groups)
      : bias(policy, num_groups, 0.0), deduplicated_paths(policy, 0),
        device_bin_segments(policy, 0), num_groups(num_groups) {
    detail::Vector<path_t, ExecutionPolicyT> device_paths(policy, begin, end);
    detail::ComputeBias(device_paths, &bias, policy);
    detail::PreprocessPaths(&device_paths, &deduplicated_paths,
                            &device_bin_segments, policy);
  }

  detail::Vector<double, ExecutionPolicyT> bias;
  detail::Vector<path_t, ExecutionPolicyT> deduplicated_paths;
  detail::Vector<std::size_t, ExecutionPolicyT> device_bin_segments;
  size_t num_groups;
};

template <typename DatasetT, typename PhiIteratorT, typename PreprocessedModelT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void GPUTreeShap(DatasetT X, const PreprocessedModelT &preprocessed_model,
                 PhiIteratorT phis_begin, PhiIteratorT phis_end,
                 const ExecutionPolicyT &policy = thrust::device) {
  if (X.NumRows() == 0 || X.NumCols() == 0 ||
      preprocessed_model.deduplicated_paths.size() == 0) {
    return;
  }

  if (size_t(phis_end - phis_begin) <
      X.NumRows() * (X.NumCols() + 1) * preprocessed_model.num_groups) {
    throw std::invalid_argument(
        "phis_out must be at least of size X.NumRows() * (X.NumCols() + 1) * "
        "num_groups");
  }

  detail::Vector<double, ExecutionPolicyT> temp_phi(policy,
                                                    phis_end - phis_begin, 0.0);
  auto d_bias = preprocessed_model.bias.data().get();
  auto d_temp_phi = temp_phi.data().get();
  auto num_groups = preprocessed_model.num_groups;
  thrust::for_each_n(policy, thrust::make_counting_iterator(0llu),
                     X.NumRows() * num_groups, [=] __device__(size_t idx) {
                       size_t group = idx % num_groups;
                       size_t row_idx = idx / num_groups;
                       d_temp_phi[IndexPhi(row_idx, num_groups, group,
                                           X.NumCols(), X.NumCols())] +=
                           d_bias[group];
                     });

  detail::ComputeShap(X, preprocessed_model.device_bin_segments,
                      preprocessed_model.deduplicated_paths,
                      preprocessed_model.num_groups, temp_phi.data().get(),
                      policy);
  thrust::copy(policy, temp_phi.begin(), temp_phi.end(), phis_begin);
}

/*!
 * Compute feature contributions on the GPU given a set of unique paths through
 * a tree ensemble and a dataset. Uses device memory proportional to the tree
 * ensemble size.
 *
 * \exception std::invalid_argument Thrown when an invalid argument error
 * condition occurs.
 * \tparam  PathIteratorT     Thrust type iterator, may be
 * thrust::device_ptr for device memory, or stl iterator/raw pointer for host
 * memory.
 * \tparam  PhiIteratorT      Thrust type iterator, may be
 * thrust::device_ptr for device memory, or stl iterator/raw pointer for host
 * memory. Value type must be floating point.
 * \tparam  DatasetT User-specified
 * dataset container.
 * \tparam  DeviceAllocatorT  Optional thrust style
 * allocator.
 *
 * \param X           Thin wrapper over a dataset allocated in device memory. X
 * should be trivially copyable as a kernel parameter (i.e. contain only
 * pointers to actual data) and must implement the methods
 * NumRows()/NumCols()/GetElement(size_t row_idx, size_t col_idx) as __device__
 * functions. GetElement may return NaN where the feature value is missing.
 * \param begin       Iterator to paths, where separate paths are delineated by
 *                    PathElement.path_idx. Each unique path should contain 1
 * root with feature_idx = -1 and zero_fraction = 1.0. The ordering of path
 * elements inside a unique path does not matter - the result will be the same.
 * Paths may contain duplicate features. See the PathElement class for more
 * information.
 * \param end         Path end iterator.
 * \param num_groups  Number
 * of output groups. In multiclass classification the algorithm outputs feature
 * contributions per output class.
 * \param phis_begin  Begin iterator for output
 * phis.
 * \param phis_end    End iterator for output phis.
 */
template <typename DatasetT, typename PathIteratorT, typename PhiIteratorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void GPUTreeShap(DatasetT X, PathIteratorT begin, PathIteratorT end,
                 size_t num_groups, PhiIteratorT phis_begin,
                 PhiIteratorT phis_end,
                 const ExecutionPolicyT &policy = thrust::device) {
  using path_t = typename std::iterator_traits<PathIteratorT>::value_type;
  Cache<ExecutionPolicyT, path_t> preprocessed_model(
      policy, begin, end, num_groups);
  GPUTreeShap(X, preprocessed_model, phis_begin, phis_end, policy);
}

/*!
 * Compute feature interaction contributions on the GPU given a set of unique
 * paths through a tree ensemble and a dataset. Uses device memory
 * proportional to the tree ensemble size.
 *
 * \exception std::invalid_argument Thrown when an invalid argument error
 *                                  condition occurs.
 * \tparam  DeviceAllocatorT  Optional thrust style allocator.
 * \tparam  DatasetT          User-specified dataset container.
 * \tparam  PathIteratorT     Thrust type iterator, may be thrust::device_ptr
 *                            for device memory, or stl iterator/raw pointer for
 *                            host memory.
 * \tparam  PhiIteratorT      Thrust type iterator, may be thrust::device_ptr
 *                            for device memory, or stl iterator/raw pointer for
 *                            host memory. Value type must be floating point.
 *
 * \param X           Thin wrapper over a dataset allocated in device memory. X
 *                    should be trivially copyable as a kernel parameter (i.e.
 *                    contain only pointers to actual data) and must implement
 *                    the methods NumRows()/NumCols()/GetElement(size_t row_idx,
 *                    size_t col_idx) as __device__ functions. GetElement may
 *                    return NaN where the feature value is missing.
 * \param begin       Iterator to paths, where separate paths are delineated by
 *                    PathElement.path_idx. Each unique path should contain 1
 *                    root with feature_idx = -1 and zero_fraction = 1.0. The
 *                    ordering of path elements inside a unique path does not
 *                    matter - the result will be the same. Paths may contain
 *                    duplicate features. See the PathElement class for more
 *                    information.
 * \param end         Path end iterator.
 * \param num_groups  Number of output groups. In multiclass classification the
 *                    algorithm outputs feature contributions per output class.
 * \param phis_begin  Begin iterator for output phis.
 * \param phis_end    End iterator for output phis.
 */
template <typename DatasetT, typename PathIteratorT, typename PhiIteratorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void GPUTreeShapInteractions(DatasetT X, PathIteratorT begin, PathIteratorT end,
                             size_t num_groups, PhiIteratorT phis_begin,
                             PhiIteratorT phis_end,
                             const ExecutionPolicyT &policy = thrust::device) {
  if (X.NumRows() == 0 || X.NumCols() == 0 || end - begin <= 0)
    return;
  if (size_t(phis_end - phis_begin) <
      X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1) * num_groups) {
    throw std::invalid_argument(
        "phis_out must be at least of size X.NumRows() * (X.NumCols() + 1)  * "
        "(X.NumCols() + 1) * "
        "num_groups");
  }

  using path_t = typename std::iterator_traits<PathIteratorT>::value_type;
  Cache<ExecutionPolicyT, path_t> preprocessed_model(
      policy, begin, end, num_groups);
  // Compute the global bias
  detail::Vector<double, ExecutionPolicyT> temp_phi(policy,
                                                    phis_end - phis_begin, 0.0);
  auto d_bias = preprocessed_model.bias.data().get();
  auto d_temp_phi = temp_phi.data().get();
  thrust::for_each_n(
      thrust::make_counting_iterator(0llu), X.NumRows() * num_groups,
      [=] __device__(size_t idx) {
        size_t group = idx % num_groups;
        size_t row_idx = idx / num_groups;
        d_temp_phi[IndexPhiInteractions(row_idx, num_groups, group, X.NumCols(),
                                        X.NumCols(), X.NumCols())] +=
            d_bias[group];
      });

  detail::ComputeShapInteractions(X, preprocessed_model.device_bin_segments,
                                  preprocessed_model.deduplicated_paths,
                                  preprocessed_model.num_groups,
                                  temp_phi.data().get());
  thrust::copy(temp_phi.begin(), temp_phi.end(), phis_begin);
}

/*!
 * Compute feature interaction contributions using the Shapley Taylor index on
 * the GPU, given a set of unique paths through a tree ensemble and a dataset.
 * Uses device memory proportional to the tree ensemble size.
 *
 * \exception std::invalid_argument Thrown when an invalid argument error
 *                                  condition occurs.
 * \tparam  PhiIteratorT      Thrust type iterator, may be thrust::device_ptr
 *                            for device memory, or stl iterator/raw pointer for
 *                            host memory. Value type must be floating point.
 * \tparam  PathIteratorT     Thrust type iterator, may be thrust::device_ptr
 *                            for device memory, or stl iterator/raw pointer for
 *                            host memory.
 * \tparam  DatasetT          User-specified dataset container.
 * \tparam  DeviceAllocatorT  Optional thrust style allocator.
 *
 * \param X           Thin wrapper over a dataset allocated in device memory. X
 *                    should be trivially copyable as a kernel parameter (i.e.
 *                    contain only pointers to actual data) and must implement
 *                    the methods NumRows()/NumCols()/GetElement(size_t row_idx,
 *                    size_t col_idx) as __device__ functions. GetElement may
 *                    return NaN where the feature value is missing.
 * \param begin       Iterator to paths, where separate paths are delineated by
 *                    PathElement.path_idx. Each unique path should contain 1
 *                    root with feature_idx = -1 and zero_fraction = 1.0. The
 *                    ordering of path elements inside a unique path does not
 *                    matter - the result will be the same. Paths may contain
 *                    duplicate features. See the PathElement class for more
 *                    information.
 * \param end         Path end iterator.
 * \param num_groups  Number of output groups. In multiclass classification the
 *                    algorithm outputs feature contributions per output class.
 * \param phis_begin  Begin iterator for output phis.
 * \param phis_end    End iterator for output phis.
 */
template <typename DatasetT, typename PathIteratorT, typename PhiIteratorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void GPUTreeShapTaylorInteractions(
    DatasetT X, PathIteratorT begin, PathIteratorT end, size_t num_groups,
    PhiIteratorT phis_begin, PhiIteratorT phis_end,
    const ExecutionPolicyT &policy = thrust::device) {
  using phis_type = typename std::iterator_traits<PhiIteratorT>::value_type;
  static_assert(std::is_floating_point<phis_type>::value,
                "Phis type must be floating point");

  if (X.NumRows() == 0 || X.NumCols() == 0 || end - begin <= 0)
    return;

  if (size_t(phis_end - phis_begin) <
      X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1) * num_groups) {
    throw std::invalid_argument(
        "phis_out must be at least of size X.NumRows() * (X.NumCols() + 1)  * "
        "(X.NumCols() + 1) * "
        "num_groups");
  }

  using path_t = typename std::iterator_traits<PathIteratorT>::value_type;
  Cache<ExecutionPolicyT, path_t> preprocessed_model(
      policy, begin, end, num_groups);
  detail::Vector<double, ExecutionPolicyT> temp_phi(policy,
                                                    phis_end - phis_begin, 0.0);
  // Compute the global bias
  auto d_bias = preprocessed_model.bias.data().get();
  auto d_temp_phi = temp_phi.data().get();
  thrust::for_each_n(
      thrust::make_counting_iterator(0llu), X.NumRows() * num_groups,
      [=] __device__(size_t idx) {
        size_t group = idx % num_groups;
        size_t row_idx = idx / num_groups;
        d_temp_phi[IndexPhiInteractions(row_idx, num_groups, group, X.NumCols(),
                                        X.NumCols(), X.NumCols())] +=
            d_bias[group];
      });

  detail::ComputeShapTaylorInteractions(
      X, preprocessed_model.device_bin_segments,
      preprocessed_model.deduplicated_paths, preprocessed_model.num_groups,
      temp_phi.data().get());
  thrust::copy(temp_phi.begin(), temp_phi.end(), phis_begin);
}

/*!
 * Compute feature contributions on the GPU given a set of unique paths through
 * a tree ensemble and a dataset. Uses device memory proportional to the tree
 * ensemble size. This variant implements the interventional tree shap algorithm
 * described here: https://drafts.distill.pub/HughChen/its_blog/
 *
 * It requires a background dataset R.
 *
 * \exception std::invalid_argument Thrown when an invalid argument error
 * condition occurs. \tparam  DeviceAllocatorT  Optional thrust style allocator.
 * \tparam  DatasetT          User-specified dataset container.
 * \tparam  PathIteratorT     Thrust type iterator, may be thrust::device_ptr
 * for device memory, or stl iterator/raw pointer for host memory.
 *
 * \param X           Thin wrapper over a dataset allocated in device memory. X
 * should be trivially copyable as a kernel parameter (i.e. contain only
 * pointers to actual data) and must implement the methods
 * NumRows()/NumCols()/GetElement(size_t row_idx, size_t col_idx) as __device__
 * functions. GetElement may return NaN where the feature value is missing.
 * \param R           Background dataset.
 * \param begin       Iterator to paths, where separate paths are delineated by
 *                    PathElement.path_idx. Each unique path should contain 1
 * root with feature_idx = -1 and zero_fraction = 1.0. The ordering of path
 * elements inside a unique path does not matter - the result will be the same.
 * Paths may contain duplicate features. See the PathElement class for more
 * information. \param end         Path end iterator. \param num_groups  Number
 * of output groups. In multiclass classification the algorithm outputs feature
 * contributions per output class. \param phis_begin  Begin iterator for output
 * phis. \param phis_end    End iterator for output phis.
 */
template <typename DatasetT, typename PathIteratorT, typename PhiIteratorT,
          typename ExecutionPolicyT = decltype(thrust::device)>
void GPUTreeShapInterventional(
    DatasetT X, DatasetT R, PathIteratorT begin, PathIteratorT end,
    size_t num_groups, PhiIteratorT phis_begin, PhiIteratorT phis_end,
    const ExecutionPolicyT &policy = thrust::device) {
  if (X.NumRows() == 0 || X.NumCols() == 0 || end - begin <= 0)
    return;

  if (size_t(phis_end - phis_begin) <
      X.NumRows() * (X.NumCols() + 1) * num_groups) {
    throw std::invalid_argument(
        "phis_out must be at least of size X.NumRows() * (X.NumCols() + 1) * "
        "num_groups");
  }

  using path_t = typename std::iterator_traits<PathIteratorT>::value_type;
  Cache<ExecutionPolicyT, path_t> preprocessed_model(
      policy, begin, end, num_groups);
  detail::Vector<double, ExecutionPolicyT> temp_phi(policy,
                                                    phis_end - phis_begin, 0.0);
  detail::ComputeShapInterventional(
      X, R, preprocessed_model.device_bin_segments,
      preprocessed_model.deduplicated_paths, preprocessed_model.num_groups,
      temp_phi.data().get());
  thrust::copy(temp_phi.begin(), temp_phi.end(), phis_begin);
}

/** @}*/

} // namespace gpu_treeshap
