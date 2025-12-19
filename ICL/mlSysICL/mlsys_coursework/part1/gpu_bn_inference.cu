#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;
constexpr int BLOCK_SIZE = 1024; // Better on last 2 kernels and slghtly worse than on first 2 kernels.
constexpr int NUM_WARPS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_WARP;

// ----------------- Array Sum Functions -------------------
// I wanted to implement a shfl version as lectures suggested, this operation is very fast.
// Unfortunately, the shfl version was not significantly faster than the version that uses only syncthreads.
// I assume that is because we this array syncing is still a neglegable part of the overall computation.

// Syncthreads version for comparison:
// __inline__ __device__ void sum_arrays_syncthreads(float* shared_sum_data, float* shared_sq_sum_data, const int num_elements) {
//     // Simple reduction using shared memory and __syncthreads()
//     for (int offset = num_elements / 2; offset > 0; offset /= 2) {
//         if (threadIdx.x < offset) {
//             shared_sum_data[threadIdx.x] += shared_sum_data[threadIdx.x + offset];
//             shared_sq_sum_data[threadIdx.x] += shared_sq_sum_data[threadIdx.x + offset];
//         }
//         __syncthreads();
//     }
// }

__inline__ __device__ void sum_arrays_shfl(float* shared_sum_data, float* shared_sq_sum_data) {
    // This implementation assumes that NUM_WARPS_PER_BLOCK <= THREADS_PER_WARP

    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    const int warp_id = threadIdx.x / THREADS_PER_WARP;

    // --------------- Reduction within warps -------------------------
    float local_sum = shared_sum_data[threadIdx.x];
    float local_sq_sum = shared_sq_sum_data[threadIdx.x];

    #pragma unroll
    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
        local_sq_sum += __shfl_down_sync(FULL_MASK, local_sq_sum, offset);
    }

    if (lane_id == 0) {
        shared_sum_data[warp_id] = local_sum;
        shared_sq_sum_data[warp_id] = local_sq_sum;
    }
    __syncthreads();

    // ------------------ Final reduction across warps -------------------------
    if (warp_id == 0) {
        // The final_sum should be 0 for threads beyond NUM_WARPS_PER_BLOCK (that's how many elements we need to sum)
        float final_sum = (lane_id < NUM_WARPS_PER_BLOCK) ? shared_sum_data[lane_id] : 0.0f;
        float final_sq_sum = (lane_id < NUM_WARPS_PER_BLOCK) ? shared_sq_sum_data[lane_id] : 0.0f;

        #pragma unroll
        for (int offset = THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(FULL_MASK, final_sum, offset);
            final_sq_sum += __shfl_down_sync(FULL_MASK, final_sq_sum, offset);
        }

        // Write the final results back to shared memory
        if (lane_id == 0) {
            shared_sum_data[0] = final_sum;
            shared_sq_sum_data[0] = final_sq_sum;
        }
    }

    __syncthreads();
}

// --------------- Kernels --------
__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ y,            // [N,C,H,W]
    const int N, const int C, const int H, const int W,
    const float eps)
{   
    const int c = blockIdx.x; // Channel
    const int tidx = threadIdx.x; // Thread idx within a block
    const int NHW = N * H * W; // Elements per channel
    const int CHW = C * H * W;
    const int HW = H * W;
    const int cHW = c * HW;
    // ------------------- Shared memory -----------------------------------
    __shared__ float s_sum_data[BLOCK_SIZE], s_sq_sum_data[BLOCK_SIZE];

    // ------------------- Mean and Variance Computation -------------------
    float local_sum_contribution = 0.0;
    float local_sq_sum_contribution = 0.0;

    for (int n_idx = 0; n_idx < N; ++n_idx) {
        const size_t base = (size_t) (n_idx * CHW + cHW);
        for (int hw = tidx; hw < HW; hw += BLOCK_SIZE) {
            const size_t index = base + (size_t) hw;

            const float f = __ldg(&x[index]);
            local_sum_contribution += f;
            local_sq_sum_contribution += f * f;
        }
    }
    
    
    s_sum_data[tidx] = local_sum_contribution;
    s_sq_sum_data[tidx] = local_sq_sum_contribution;
    __syncthreads();

    sum_arrays_shfl(s_sum_data, s_sq_sum_data);

    float mean_c = s_sum_data[0] / (float) NHW;
    float var_c = s_sq_sum_data[0] / (float) NHW - (mean_c * mean_c);

    // --------------------------- Normalization + Affine  ---------------------------
    const float g = __ldg(&gamma[c]);
    const float b = __ldg(&beta[c]);
    float inv_std = rsqrtf(var_c + eps); // Fast sqrt inverse
    float neg_mean_times_inv_std = -1 * mean_c * inv_std; // Precomputed for efficiency

    for (int n_idx = 0; n_idx < N; ++n_idx) {
        const size_t base = (size_t) (n_idx * CHW + cHW);
        for (int hw = tidx; hw < HW; hw += BLOCK_SIZE) {
            const size_t index = base + (size_t) hw;
            const float xv = __ldg(&x[index]);
            const float xn = __fmaf_rn(inv_std, xv, neg_mean_times_inv_std);
            y[index] = __fmaf_rn(g, xn, b);
        }
    }
}

void solve(const float* input, // [N,C,H,W] (input shape: batch, channel, height, weight)  
           const float* gamma, // [C] 
           const float* beta, // [C]
           float* output, // [N,C,H,W]
           int N, int C, int H, int W, float eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    dim3 block(BLOCK_SIZE);
    dim3 grid(C); // One block per channel

    gpu_bn_inference_nchw<<<grid, block>>>(
        input,
        gamma,
        beta,
        output,
        N, C, H, W,
        eps
    );
}

