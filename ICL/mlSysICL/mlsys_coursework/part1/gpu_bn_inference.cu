#include "solve.h"
#include <cuda_runtime.h>
#include <cassert>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_WARP;


// ----------------- Array Sum Functions -------------------
// I wanted to implement a shfl version as lectures suggested, this operation is very fast.
// Unfortunately, the shfl version was not significantly faster than the syncthreads version.

__device__ void sum_arrays_syncthreads(double* shared_sum_data, double* shared_sq_sum_data, int num_elements) {
    // Simple reduction using shared memory and __syncthreads()
    for (int offset = num_elements / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_sum_data[threadIdx.x] += shared_sum_data[threadIdx.x + offset];
            shared_sq_sum_data[threadIdx.x] += shared_sq_sum_data[threadIdx.x + offset];
        }
        __syncthreads();
    }
}

__device__ void sum_arrays_shfl(double* shared_sum_data, double* shared_sq_sum_data, int num_elements) {
    // Idea: First reduce with warps using __shfl_down_sync, then final reduction with __syncthreads()
    // This implementation assumes that NUM_OF_WARPS_PER_BLOCK <= THREADS_PER_WARP

    // Reduction within warps
    int lane_id = threadIdx.x % THREADS_PER_WARP;
    int warp_id = threadIdx.x / THREADS_PER_WARP;

    // First stage: Warp-level reduction
    double local_sum = shared_sum_data[threadIdx.x];
    double local_sq_sum = shared_sq_sum_data[threadIdx.x];

    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
        local_sq_sum += __shfl_down_sync(FULL_MASK, local_sq_sum, offset);
    }

    // Write warp results to shared memory
    if (lane_id == 0) {
        shared_sum_data[warp_id] = local_sum;
        shared_sq_sum_data[warp_id] = local_sq_sum;
    }
    __syncthreads();

    // Final reduction using the first warp
    if (warp_id == 0) {
        // The final_sum should be 0 for threads beyond NUM_WARPS_PER_BLOCK (that's how many elements we need to sum)
        double final_sum = (lane_id < NUM_WARPS_PER_BLOCK) ? shared_sum_data[lane_id] : 0.0;
        double final_sq_sum = (lane_id < NUM_WARPS_PER_BLOCK) ? shared_sq_sum_data[lane_id] : 0.0;

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
// It's better to calculate mean and variance in one pass using Var[X] = E[X^2] - (E[X])^2 instead of two separate passes
__global__ void gpu_bn_compute_mean_var(
    const float* __restrict__ x,      // [N,C,H,W]
    double* __restrict__ mean,        // [C]
    double* __restrict__ var,         // [C]
    int N, int C, int H, int W
) {
    size_t NHW = (size_t)N * H * W; // Elements per channel
    size_t CHW = (size_t)C * H * W;
    size_t HW = (size_t)H * W;
    int c = blockIdx.x; // 1d grid for channels
    size_t elements_per_channel = NHW;
    
    if (c >= C) return;

    double local_sum_contribution = 0.0;
    double local_sq_sum_contribution = 0.0;

    // Iterate over batches and spatial HW to avoid division/mod operations
    #pragma unroll
    for (int n_idx = 0; n_idx < N; ++n_idx) {
        size_t base = (size_t)n_idx * CHW + (size_t)c * HW;
        for (int hw = threadIdx.x; hw < (int)HW; hw += BLOCK_SIZE) {
            size_t index = base + (size_t)hw;

            double d = static_cast<double>(x[index]);
            local_sum_contribution += d;
            local_sq_sum_contribution += d * d;
        }
    }
    
    __shared__ double shared_sum_data[BLOCK_SIZE], shared_sq_sum_data[BLOCK_SIZE];
    shared_sum_data[threadIdx.x] = local_sum_contribution;
    shared_sq_sum_data[threadIdx.x] = local_sq_sum_contribution;
    __syncthreads();

    sum_arrays_shfl(shared_sum_data, shared_sq_sum_data, BLOCK_SIZE);

    if (threadIdx.x == 0) {
        mean[c] = shared_sum_data[0] / (double)(NHW);
        var[c] = shared_sq_sum_data[0] / (double)(NHW) - mean[c] * mean[c];
    }
}

__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    const double* __restrict__ mean,  // [C]
    const double* __restrict__ var,   // [C]
    float* __restrict__ y,            // [N,C,H,W]
    int N, int C, int H, int W,
    float eps)
{   
    size_t NHW = (size_t)N * H * W; // Elements per channel
    size_t CHW = (size_t)C * H * W;
    size_t HW = (size_t)H * W;
    int c = blockIdx.x; // 1d grid for channels
    size_t elements_per_channel = NHW;
    
    if (c >= C) return;

    double inv_std = rsqrtf(var[c] + (double)eps); // Fast sqrt inverse computation
    double mean_c = static_cast<double>(mean[c]);
    double g = static_cast<double>(gamma[c]);
    double b = static_cast<double>(beta[c]);

    // Normalize and apply affine transform using nested loops (no / or %)
    #pragma unroll
    for (int n_idx = 0; n_idx < N; ++n_idx) {
        size_t base = (size_t)n_idx * CHW + (size_t)c * HW;
        for (int hw = threadIdx.x; hw < (int)HW; hw += BLOCK_SIZE) {
            size_t index = base + (size_t)hw;
            double xn = (static_cast<double>(x[index]) - mean_c) * inv_std;
            y[index] = static_cast<float>(g * xn + b);
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
    dim3 block(BLOCK_SIZE); // Example block size, can be tuned
    dim3 grid(C);    // One block per channel

    // Allocate mean and variance arrays on device
    double *d_mean, *d_var;

    cudaMalloc(&d_mean, C * sizeof(double));
    cudaMalloc(&d_var, C * sizeof(double));
    // Compute mean and variance
    gpu_bn_compute_mean_var<<<grid, block>>>(
        input,
        d_mean,
        d_var,
        N, C, H, W
    );

    // Assertions useful for sum_arrays function
    gpu_bn_inference_nchw<<<grid, block>>>(
        input,
        gamma,
        beta,
        d_mean,
        d_var,
        output,
        N, C, H, W,
        eps
    );

    cudaFree(d_mean);
    cudaFree(d_var);
}

