#include "solve.h"
#include <cuda_runtime.h>
#include <cassert>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;
constexpr int BLOCK_DIM = 32;
constexpr int BLOCK_SIZE = BLOCK_DIM * BLOCK_DIM;
constexpr int MAX_KERNEL_SIZE = 32 * 32; // Max kernel size supported. Spec says KH, KW <= 32
constexpr int BATCHES_PER_BLOCK = 4;

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Note: 1. External libraries are not allowed
//       2. The solve function signature must keep unchanged
//       3. Store the final result in the array y

__global__ void gpu_dwconv2d(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ k,      // [C,1,KH,KW] flattened as [C*KH*KW]
    float* __restrict__ y,            // [N,C,OH,OW]
    int N, int C, int H, int W,
    int KH, int KW,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int OH, int OW)
{   
    const int c = blockIdx.z % C;
    const int n_base = (blockIdx.z / C) * BATCHES_PER_BLOCK;
    const int tidx = threadIdx.x;

    const int HW = H * W;
    const int CHW = C * HW;
    const int cHW = c * HW;
    const int OHOW = OH * OW;
    const int COHOW = C * OHOW;
    const int KHKW = KH * KW;

    const int ow = blockIdx.x * BLOCK_DIM + (threadIdx.x % BLOCK_DIM);
    const int oh = blockIdx.y * BLOCK_DIM + (threadIdx.x / BLOCK_DIM);
    const int h0 = oh * stride_h - pad_h;
    const int w0 = ow * stride_w - pad_w;

    // -------------------- Preload kernel to shared memory --------------------
    __shared__ float s_k[MAX_KERNEL_SIZE];
    if (tidx < KHKW) {
        s_k[tidx] = __ldg(&k[(size_t)c * KHKW + (size_t)tidx]);
    }
    __syncthreads();


    // ---------------------- Compute convolution --------------------
    if (oh < OH && ow < OW) {
        #pragma unroll
        for (int n_rem = 0; n_rem < BATCHES_PER_BLOCK; ++n_rem) {
            int n = n_base + n_rem;
            if (n >= N) break;

            float acc = 0.0f;

            for (int kh = 0; kh < KH; ++kh) { // kernel height
                int ih = h0 + kh;
                if ((unsigned)ih >= (unsigned)H) continue;

                for (int kw = 0; kw < KW; ++kw) { // kernel width
                    int iw = w0 + kw;
                    if ((unsigned)iw >= (unsigned)W) continue;
                    
                    float xv = __ldg(&x[(size_t)n * CHW + (size_t)cHW + (size_t)ih * W + (size_t)iw]);
                    float kv = s_k[(size_t)kh * KW + (size_t)kw];
                    acc = __fmaf_rn(xv, kv, acc);
                }
            }
            y[(size_t)n * COHOW + (size_t)c * OHOW + (size_t)oh * OW + (size_t)ow] = acc;
        }
    }
}

void solve(const float* input,      // [N,C,H,W] 
           const float* kernel,     // [C,1,KH,KW] flattened as [C*KH*KW]
           float* output,           // [N,C,OH,OW]
           int N, int C, int H, int W, // input shape: batch, channel, height, weight
           int KH, int KW, // kernel height and weight
           int pad_h, int pad_w,    // kernel padding
           int stride_h, int stride_w)  // kernel stride
{
    if (N<=0 || C<=0 || H<=0 || W<=0 || KH<=0 || KW<=0) return;

    int OH = (H + 2*pad_h - KH) / stride_h + 1;
    int OW = (W + 2*pad_w - KW) / stride_w + 1;
    if (OH <= 0 || OW <= 0) return;

    assert(BLOCK_SIZE >= MAX_KERNEL_SIZE);
    assert(KH * KW <= MAX_KERNEL_SIZE);

    dim3 block(BLOCK_DIM * BLOCK_DIM);
    // Each block processes BATCHES_PER_BLOCK batches for a single channel
    dim3 grid(CEIL_DIV(OW, BLOCK_DIM), CEIL_DIV(OH, BLOCK_DIM), C * CEIL_DIV(N, BATCHES_PER_BLOCK));

    gpu_dwconv2d<<<grid, block>>>(
        input, kernel, output,
        N, C, H, W, KH, KW,
        pad_h, pad_w, stride_h, stride_w,
        OH, OW);
}

