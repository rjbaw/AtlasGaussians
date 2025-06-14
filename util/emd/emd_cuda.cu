// EMD approximation module (based on auction algorithm)
// author: Minghua Liu
// modified: rjbaw
#include <ATen/ATen.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include <cuda/pipeline>

#include <cuda_fp4.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 120
using fp_t = __nv_fp4_e2m1;
#define TO_FP(x) static_cast<float>(x)
#define FROM_FP(x) static_cast<__nv_fp4_e2m1>(x)
#elif defined(__CUDA_ARCH__)
#include <cuda_fp16.h>
using fp_t = __half;
#define TO_FP(x) __half2float(x)
#define FROM_FP(x) __float2half(x)
#else
using fp_t = float;
#define TO_FP(x) (x)
#define FROM_FP(x) (x)
#endif

__device__ __forceinline__ float atomicMax_f32(float *address, float val) {
  int *address_int = reinterpret_cast<int *>(address);
#if __CUDA_ARCH__ >= 600
  int val_int = __float_as_int(val);
  int old_int = atomicMax(address_int, val_int);
  return __int_as_float(old_int);
#else
  int old_int = *address_int;
  int assumed_int;
  do {
    assumed_int = old_int;
    float old_val = __int_as_float(assumed_int);
    float max_val = fmaxf(old_val, val);
    int new_val_int = __float_as_int(max_val);
    old_int = atomicCAS(address_int, assumed_int, new_val_int);
  } while (assumed_int != old_int);
  return __int_as_float(old_int);
#endif
}

__global__ void clear(int b, int *cnt_tmp, int *unass_cnt) {
  for (int i = threadIdx.x; i < b; i += blockDim.x) {
    cnt_tmp[i] = 0;
    unass_cnt[i] = 0;
  }
}

__global__ void calc_unass_cnt(int b, int n, int *assignment, int *unass_cnt) {
  // count the number of unassigned points in each batch
  const int BLOCK_SIZE = 1024;
  __shared__ int scan_array[BLOCK_SIZE];
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    scan_array[threadIdx.x] =
        assignment[i * n + blockIdx.y * BLOCK_SIZE + threadIdx.x] == -1 ? 1 : 0;
    __syncthreads();

    int stride = 1;
    while (stride <= BLOCK_SIZE / 2) {
      int index = (threadIdx.x + 1) * stride * 2 - 1;
      if (index < BLOCK_SIZE)
        scan_array[index] += scan_array[index - stride];
      stride = stride * 2;
      __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == BLOCK_SIZE - 1) {
      atomicAdd(&unass_cnt[i], scan_array[threadIdx.x]);
    }
    __syncthreads();
  }
}

__global__ void calc_unass_cnt_sum(int b, int *unass_cnt, int *unass_cnt_sum) {
  // count the cumulative sum over over unass_cnt
  const int BLOCK_SIZE = 512; // batch_size <= 512
  __shared__ int scan_array[BLOCK_SIZE];
  scan_array[threadIdx.x] = unass_cnt[threadIdx.x];
  __syncthreads();

  int stride = 1;
  while (stride <= BLOCK_SIZE / 2) {
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < BLOCK_SIZE)
      scan_array[index] += scan_array[index - stride];
    stride = stride * 2;
    __syncthreads();
  }
  __syncthreads();
  stride = BLOCK_SIZE / 4;
  while (stride > 0) {
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < BLOCK_SIZE)
      scan_array[index + stride] += scan_array[index];
    stride = stride / 2;
    __syncthreads();
  }
  __syncthreads();

  // printf("%d\n", unass_cnt_sum[b - 1]);
  unass_cnt_sum[threadIdx.x] = scan_array[threadIdx.x];
}

__global__ void calc_unass_idx(int b, int n, int *assignment, int *unass_idx,
                               int *unass_cnt, int *unass_cnt_sum,
                               int *cnt_tmp) {
  // list all the unassigned points
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    if (assignment[i * n + blockIdx.y * 1024 + threadIdx.x] == -1) {
      int idx = atomicAdd(&cnt_tmp[i], 1);
      unass_idx[unass_cnt_sum[i] - unass_cnt[i] + idx] =
          blockIdx.y * 1024 + threadIdx.x;
    }
  }
}

// template <typename T>
// __global__ void Bid(int b, int n, const float *__restrict__ xyz1,
//                     const float *__restrict__ xyz2, float eps, int
//                     *assignment, int *assignment_inv, T *price, int *bid, T
//                     *bid_increments, float *max_increments, int *unass_cnt,
//                     int *unass_cnt_sum, int *unass_idx) {
//   constexpr int THREADS_PER_UNASS = 128;
//   constexpr int TILE = 512;
//   constexpr int WARP = 32;

//   const int lane = threadIdx.x & (WARP - 1);
//   const int thread_in_unass = threadIdx.x % THREADS_PER_UNASS;
//   const int group_id = threadIdx.x / THREADS_PER_UNASS;
//   const int groups_per_blk = blockDim.x / THREADS_PER_UNASS;

//   extern __shared__ float smem[];
//   float *xyz2_buf = smem;
//   T *price_buf = reinterpret_cast<T *>(xyz2_buf + 3 * TILE);

//   __shared__ float best_buf[TILE];
//   __shared__ float better_buf[TILE];
//   __shared__ int best_i_buf[TILE];

//   for (int i = blockIdx.x; i < b; i += gridDim.x) {
//     const int _unass_cnt = unass_cnt[i];

//     const int block_cnt = gridDim.y;
//     const int unass_per_block = (_unass_cnt + block_cnt - 1) / block_cnt;
//     const int first_unass_idx = blockIdx.y * unass_per_block;
//     const int unass_rank = first_unass_idx + group_id;
//     const bool active = (unass_rank < _unass_cnt);
//     int _unass_id = -1;
//     float x1 = 0.f, y1 = 0.f, z1 = 0.f;
//     if (active) {
//       _unass_id = unass_idx[unass_cnt_sum[i] - _unass_cnt + unass_rank];
//       const int base1 = (i * n + _unass_id) * 3;
//       x1 = xyz1[base1 + 0];
//       y1 = xyz1[base1 + 1];
//       z1 = xyz1[base1 + 2];
//     }
//     float best = -1e9f;
//     float better = -1e9f;
//     int best_i = -1;

//     for (int k2 = 0; k2 < n; k2 += TILE) {
//       const int elems = min(TILE, n - k2);

// #if __CUDA_ARCH__ >= 800
//       cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
//       pipe.producer_acquire();
//       cuda::memcpy_async(xyz2_buf, &xyz2[(i * n + k2) * 3],
//                          sizeof(float) * elems * 3, pipe);
//       cuda::memcpy_async(price_buf, &price[i * n + k2], sizeof(T) * elems,
//                          pipe);
//       pipe.producer_commit();
//       pipe.consumer_wait();
// #else
//       for (int t = threadIdx.x; t < elems * 3; t += blockDim.x)
//         xyz2_buf[t] = xyz2[(i * n + k2) * 3 + t];
//       for (int t = threadIdx.x; t < elems; t += blockDim.x)
//         price_buf[t] = price[i * n + k2 + t];
//       __syncthreads();
// #endif
//       if (active) {
//         for (int j = thread_in_unass; j < elems; j += THREADS_PER_UNASS) {

//           const int idx3 = j * 3;
//           const float dx = xyz2_buf[idx3 + 0] - x1;
//           const float dy = xyz2_buf[idx3 + 1] - y1;
//           const float dz = xyz2_buf[idx3 + 2] - z1;
//           const float score = 9.f - (dx * dx + dy * dy + dz * dz) -
//                               2.f * static_cast<float>(price_buf[j]);
//           const int g_idx = k2 + j;

//           if (score > best) {
//             better = best;
//             best = score;
//             best_i = g_idx;
//           } else if (score > better)
//             better = score;
//         }
//       }
//     }

//     for (int offset = WARP / 2; offset; offset >>= 1) {
//       const float b_val = __shfl_down_sync(0xffffffff, best, offset);
//       const float s_val = __shfl_down_sync(0xffffffff, better, offset);
//       const int b_idx = __shfl_down_sync(0xffffffff, best_i, offset);
//       if (b_val > best) {
//         better = fmaxf(best, s_val);
//         best = b_val;
//         best_i = b_idx;
//       } else if (b_val > better) {
//         better = b_val;
//       } else if (s_val > better) {
//         better = s_val;
//       }
//     }
//     if constexpr (THREADS_PER_UNASS > WARP) {
//       int sm_idx = threadIdx.x / WARP;
//       if (lane == 0) {
//         best_buf[sm_idx] = best;
//         better_buf[sm_idx] = better;
//         best_i_buf[sm_idx] = best_i;
//       }
//       __syncthreads();

//       if (active && lane == 0) {
//         int base = group_id * (THREADS_PER_UNASS / WARP);
//         int limit = base + THREADS_PER_UNASS / WARP;

//         best = best_buf[base];
//         better = better_buf[base];
//         best_i = best_i_buf[base];

//         for (int t = base + 1; t < limit; ++t) {
//           float b_val = best_buf[t];
//           if (b_val > best) {
//             better = fmaxf(best, better_buf[t]);
//             best = b_val;
//             best_i = best_i_buf[t];
//           } else
//             better = fmaxf(better, b_val);
//         }
//       }
//     }

//     if (active && lane == 0) {
//       float inc = best - better + eps;
//       bid[i * n + _unass_id] = best_i;
//       bid_increments[i * n + _unass_id] = static_cast<T>(inc);
//       atomicMax_f32(&max_increments[i * n + best_i], inc);
//     }
//   }
// }

template <typename T>
__global__ void Bid(int b, int n, const float *__restrict__ xyz1,
                    const float *__restrict__ xyz2, float eps, int *assignment,
                    int *assignment_inv, T *price, int *bid, T *bid_increments,
                    float *max_increments, int *unass_cnt, int *unass_cnt_sum,
                    int *unass_idx) {

  const int TILE = 512;
  const int batch = TILE;
  const int block_cnt = gridDim.y;
  constexpr int block_size = 1024;

  extern __shared__ char smem[];
  float *xyz2_buf = reinterpret_cast<float *>(smem);
  T *price_buf = reinterpret_cast<T *>(xyz2_buf + 3 * TILE);

  __shared__ float best_buf[block_size];
  __shared__ float better_buf[block_size];
  __shared__ int best_i_buf[block_size];

  for (int i = blockIdx.x; i < b; i += gridDim.x) {

    int _unass_cnt = unass_cnt[i];
    if (_unass_cnt == 0)
      continue;

    int _unass_cnt_sum = unass_cnt_sum[i];
    int unass_per_block = (_unass_cnt + block_cnt - 1) / block_cnt;
    int thread_per_unass = block_size / unass_per_block;
    int unass_this_block = max(
        min(_unass_cnt - (int)blockIdx.y * unass_per_block, unass_per_block),
        0);

    float x1, y1, z1, best = -1e9, better = -1e9;
    int best_i = -1, _unass_id = -1, thread_in_unass;

    if (threadIdx.x < thread_per_unass * unass_this_block) {
      _unass_id = unass_per_block * blockIdx.y +
                  threadIdx.x / thread_per_unass + _unass_cnt_sum - _unass_cnt;
      _unass_id = unass_idx[_unass_id];
      thread_in_unass = threadIdx.x % thread_per_unass;

      x1 = TO_FP(xyz1[(i * n + _unass_id) * 3 + 0]);
      y1 = TO_FP(xyz1[(i * n + _unass_id) * 3 + 1]);
      z1 = TO_FP(xyz1[(i * n + _unass_id) * 3 + 2]);
    }

    for (int k2 = 0; k2 < n; k2 += batch) {

      int end_k = min(n, k2 + batch) - k2;
      int idx = threadIdx.x;
      while (idx < end_k * 3) {
        xyz2_buf[idx] = xyz2[(i * n + k2) * 3 + idx];
        idx += blockDim.x;
      }
      idx = threadIdx.x;
      while (idx < end_k) {
        price_buf[idx] = price[i * n + k2 + idx];
        idx += blockDim.x;
      }
      __syncthreads();

      if (_unass_id != -1) {
        int delta = (end_k + thread_per_unass - 1) / thread_per_unass;
        int l = thread_in_unass * delta;
        int r = min((thread_in_unass + 1) * delta, end_k);
        for (int k = l; k < r; k++)
        // if (!last || assignment_inv[i * n + k + k2] == -1)
        {
          float x2 = xyz2_buf[k * 3 + 0] - x1;
          float y2 = xyz2_buf[k * 3 + 1] - y1;
          float z2 = xyz2_buf[k * 3 + 2] - z1;
          // the coordinates of points should be normalized to [0, 1]
          // float d = 3.0 - sqrtf(x2 * x2 + y2 * y2 + z2 * z2) -
          price_buf[k];
          float d =
              9.f - (x2 * x2 + y2 * y2 + z2 * z2) - 2.f * TO_FP(price_buf[k]);

          if (d > best) {
            better = best;
            best = d;
            best_i = k + k2;
          } else if (d > better) {
            better = d;
          }
        }
      }
      __syncthreads();
    }

    best_buf[threadIdx.x] = best;
    better_buf[threadIdx.x] = better;
    best_i_buf[threadIdx.x] = best_i;
    __syncthreads();

    if (_unass_id != -1 && thread_in_unass == 0) {
      for (int j = threadIdx.x + 1; j < threadIdx.x + thread_per_unass; j++) {
        if (best_buf[j] > best) {
          better = max(best, better_buf[j]);
          best = best_buf[j];
          best_i = best_i_buf[j];
        } else
          better = max(better, best_buf[j]);
      }
      bid[i * n + _unass_id] = best_i;
      bid_increments[i * n + _unass_id] = FROM_FP(best - better + eps);
      atomicMax_f32(&max_increments[i * n + best_i], best - better + eps);
    }
  }
}

__global__ void GetMax(int b, int n, int *assignment, int *bid,
                       float *bid_increments, float *max_increments,
                       int *max_idx) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    int j = threadIdx.x + blockIdx.y * blockDim.x;
    if (assignment[i * n + j] == -1) {
      int bid_id = bid[i * n + j];
      float bid_inc = bid_increments[i * n + j];
      float max_inc = max_increments[i * n + bid_id];
      if (bid_inc - 1e-6 <= max_inc && max_inc <= bid_inc + 1e-6) {
        max_idx[i * n + bid_id] = j;
      }
    }
  }
}

__global__ void Assign(int b, int n, int *assignment, int *assignment_inv,
                       float *price, int *bid, float *bid_increments,
                       float *max_increments, int *max_idx, bool last) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    int j = threadIdx.x + blockIdx.y * blockDim.x;
    if (assignment[i * n + j] == -1) {
      int bid_id = bid[i * n + j];
      if (last || max_idx[i * n + bid_id] == j) {
        float bid_inc = bid_increments[i * n + j];
        int ass_inv = assignment_inv[i * n + bid_id];
        if (!last && ass_inv != -1) {
          assignment[i * n + ass_inv] = -1;
        }
        assignment_inv[i * n + bid_id] = j;
        assignment[i * n + j] = bid_id;
        price[i * n + bid_id] += bid_inc;
        max_increments[i * n + bid_id] = -1e9;
      }
    }
  }
}

__global__ void CalcDist(int b, int n, float *xyz1, float *xyz2, float *dist,
                         int *assignment) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    int j = threadIdx.x + blockIdx.y * blockDim.x;
    int k = assignment[i * n + j];
    float deltax = xyz1[(i * n + j) * 3 + 0] - xyz2[(i * n + k) * 3 + 0];
    float deltay = xyz1[(i * n + j) * 3 + 1] - xyz2[(i * n + k) * 3 + 1];
    float deltaz = xyz1[(i * n + j) * 3 + 2] - xyz2[(i * n + k) * 3 + 2];
    dist[i * n + j] = deltax * deltax + deltay * deltay + deltaz * deltaz;
  }
}

int emd_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist,
                     at::Tensor assignment, at::Tensor price,
                     at::Tensor assignment_inv, at::Tensor bid,
                     at::Tensor bid_increments, at::Tensor max_increments,
                     at::Tensor unass_idx, at::Tensor unass_cnt,
                     at::Tensor unass_cnt_sum, at::Tensor cnt_tmp,
                     at::Tensor max_idx, float eps, int iters) {

  const auto batch_size = xyz1.size(0);
  const auto n = xyz1.size(1); // num_points point cloud A
  const auto m = xyz2.size(1); // num_points point cloud B

  if (n != m) {
    printf("Input Error! The two point clouds should have the same size.\n");
    return -1;
  }

  if (batch_size > 512) {
    printf("Input Error! The batch size should be less than 512.\n");
    return -1;
  }

  if (n % 1024 != 0) {
    printf("Input Error! The size of the point clouds should be a multiple of "
           "1024.\n");
    return -1;
  }

  // cudaEvent_t start,stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start);
  // int iters = 50;
  for (int i = 0; i < iters; i++) {
    clear<<<1, batch_size>>>(batch_size, cnt_tmp.data_ptr<int>(),
                             unass_cnt.data_ptr<int>());
    calc_unass_cnt<<<dim3(batch_size, n / 1024, 1), 1024>>>(
        batch_size, n, assignment.data_ptr<int>(), unass_cnt.data_ptr<int>());
    calc_unass_cnt_sum<<<1, batch_size>>>(batch_size, unass_cnt.data_ptr<int>(),
                                          unass_cnt_sum.data_ptr<int>());
    calc_unass_idx<<<dim3(batch_size, n / 1024, 1), 1024>>>(
        batch_size, n, assignment.data_ptr<int>(), unass_idx.data_ptr<int>(),
        unass_cnt.data_ptr<int>(), unass_cnt_sum.data_ptr<int>(),
        cnt_tmp.data_ptr<int>());

    constexpr int TILE = 512;
    size_t shared_bytes = sizeof(float) * 3 * TILE + sizeof(fp_t) * TILE;
    Bid<fp_t><<<dim3(batch_size, n / 1024, 1), 1024, shared_bytes>>>(
        batch_size, n, xyz1.data_ptr<float>(), xyz2.data_ptr<float>(), eps,
        assignment.data_ptr<int>(), assignment_inv.data_ptr<int>(),
        price.data_ptr<fp_t>(), bid.data_ptr<int>(),
        bid_increments.data_ptr<fp_t>(), max_increments.data_ptr<float>(),
        unass_cnt.data_ptr<int>(), unass_cnt_sum.data_ptr<int>(),
        unass_idx.data_ptr<int>());

    // constexpr int TILE = 512;
    // size_t shared_bytes = sizeof(float) * 3 * TILE + sizeof(fp_t) * TILE;
    // dim3 grid(batch_size, n / 1024, 1), block(1024);
    // Bid<fp_t><<<grid, block, shared_bytes>>>(
    //     batch_size, n, xyz1.data_ptr<float>(), xyz2.data_ptr<float>(), eps,
    //     assignment.data_ptr<int>(), assignment_inv.data_ptr<int>(),
    //     price.data_ptr<fp_t>(), bid.data_ptr<int>(),
    //     bid_increments.data_ptr<fp_t>(), max_increments.data_ptr<float>(),
    //     unass_cnt.data_ptr<int>(), unass_cnt_sum.data_ptr<int>(),
    //     unass_idx.data_ptr<int>());

    GetMax<<<dim3(batch_size, n / 1024, 1), 1024>>>(
        batch_size, n, assignment.data_ptr<int>(), bid.data_ptr<int>(),
        bid_increments.data_ptr<float>(), max_increments.data_ptr<float>(),
        max_idx.data_ptr<int>());
    Assign<<<dim3(batch_size, n / 1024, 1), 1024>>>(
        batch_size, n, assignment.data_ptr<int>(),
        assignment_inv.data_ptr<int>(), price.data_ptr<float>(),
        bid.data_ptr<int>(), bid_increments.data_ptr<float>(),
        max_increments.data_ptr<float>(), max_idx.data_ptr<int>(),
        i == iters - 1);
  }
  CalcDist<<<dim3(batch_size, n / 1024, 1), 1024>>>(
      batch_size, n, xyz1.data_ptr<float>(), xyz2.data_ptr<float>(),
      dist.data_ptr<float>(), assignment.data_ptr<int>());
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float elapsedTime;
  // cudaEventElapsedTime(&elapsedTime,start,stop);
  // printf("%lf\n", elapsedTime);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in nnd Output: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

__global__ void NmDistanceGradKernel(int b, int n, const float *xyz1,
                                     const float *xyz2, const float *grad_dist,
                                     const int *idx, float *grad_xyz) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
         j += blockDim.x * gridDim.y) {
      float x1 = xyz1[(i * n + j) * 3 + 0];
      float y1 = xyz1[(i * n + j) * 3 + 1];
      float z1 = xyz1[(i * n + j) * 3 + 2];
      int j2 = idx[i * n + j];
      float x2 = xyz2[(i * n + j2) * 3 + 0];
      float y2 = xyz2[(i * n + j2) * 3 + 1];
      float z2 = xyz2[(i * n + j2) * 3 + 2];
      float g = grad_dist[i * n + j] * 2;
      atomicAdd(&(grad_xyz[(i * n + j) * 3 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz[(i * n + j) * 3 + 1]), g * (y1 - y2));
      atomicAdd(&(grad_xyz[(i * n + j) * 3 + 2]), g * (z1 - z2));
    }
  }
}

int emd_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz,
                      at::Tensor graddist, at::Tensor idx) {
  const auto batch_size = xyz1.size(0);
  const auto n = xyz1.size(1);
  const auto m = xyz2.size(1);

  NmDistanceGradKernel<<<dim3(batch_size, n / 1024, 1), 1024>>>(
      batch_size, n, xyz1.data_ptr<float>(), xyz2.data_ptr<float>(),
      graddist.data_ptr<float>(), idx.data_ptr<int>(),
      gradxyz.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}
