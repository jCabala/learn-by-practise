#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;
constexpr int BLOCK_DIM = 32;
constexpr int BLOCK_SIZE = BLOCK_DIM * BLOCK_DIM;

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
    const int c = blockIdx.z; /// N; // Channel index
    //const int n = blockIdx.z % N; // Batch index

    const int HW = H * W;
    const int CHW = C * HW;
    const int cHW = c * HW;
    const int OHOW = OH * OW;
    const int COHOW = C * OHOW;
    const int KHKW = KH * KW;

    // Each location of output is written by one thread

    __shared__ float s_k[BLOCK_SIZE], s_x[BLOCK_SIZE];

    const int ow = blockIdx.x * BLOCK_DIM + (threadIdx.x % BLOCK_DIM);
    const int oh = blockIdx.y * BLOCK_DIM + (threadIdx.x / BLOCK_DIM);

    

    if (oh < OH && ow < OW) {
        for (int n = 0; n < N; ++n) {
            int h0 = oh * stride_h - pad_h;
            int w0 = ow * stride_w - pad_w;

            float acc = 0.0f;
            for (int kh = 0; kh < KH; ++kh) { // kernel height
                int ih = h0 + kh;
                if ((unsigned)ih >= (unsigned)H) continue;
                for (int kw = 0; kw < KW; ++kw) { // kernel width
                    int iw = w0 + kw;
                    if ((unsigned)iw >= (unsigned)W) continue;
                    float xv = __ldg(&x[(size_t)n * CHW + (size_t)cHW + (size_t)ih * W + (size_t)iw]);
                    float kv = __ldg(&k[(size_t)c * KHKW + (size_t)kh * KW + (size_t)kw]);
                    acc = __fmaf_rn(xv, kv, acc);
                }
            }
            y[(size_t)n * COHOW + (size_t)c * OHOW + (size_t)oh * OW + (size_t)ow] = acc;
        }
    }
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
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


    // Idea one: dim Z of grid will be for number of channels
    // Idea two: each grid XY block will compute one output channel for all N images
    // Idea three: At that point we basically have matrix multiplication
   
    dim3 block(BLOCK_DIM * BLOCK_DIM); // Coalescing
    dim3 grid(ceil_div(OW, BLOCK_DIM), ceil_div(OH, BLOCK_DIM), C); // One "XY grid block" per channel

    gpu_dwconv2d<<<grid, block>>>(
        input, kernel, output,
        N, C, H, W, KH, KW,
        pad_h, pad_w, stride_h, stride_w,
        OH, OW);
}

