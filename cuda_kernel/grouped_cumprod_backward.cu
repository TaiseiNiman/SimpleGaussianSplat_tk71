// grouped_cumprod_grad_optimized.cu

#include <torch/extension.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

__global__ void grouped_cumprod_backward_kernel(
    const float* __restrict__ param,
    const float* __restrict__ param_cumprod,
    const float* __restrict__ grad_out,
    const int* __restrict__ inv,
    float* __restrict__ grad_in,
    int n,
    const int* __restrict__ inv_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 誤差逆伝播: backward scan (within group)
    int gid = inv[idx];
    int i_max = inv_len[gid];
    float val = 0.0f;
    float param_idx = (param[idx] != 0.0f) ? param[idx] : 1e-8f;
    for (int i = idx; i < i_max; i++){
        val += grad_out[i] * (param_cumprod[i] / param_idx);
    }
    grad_in[idx] = val;
    // // warp-synchronous backward scan
    // // （各グループ内の連続領域を前提にして逆方向走査）
    // for (int offset = 1; offset < 32; offset *= 2) {
    //     int neighbor = idx + offset;
    //     if (neighbor < n && inv[neighbor] == gid) {
    //         float neighbor_val = __shfl_down_sync(0xffffffff, val, offset);
    //         val += neighbor_val * (param[neighbor] / param[idx]);
    //     }
    // }
    //grad_in[idx] = val;
    //atomicAdd(&grad_in[idx], val);
}

void grouped_cumprod_backward(
    torch::Tensor param,
    torch::Tensor param_cumprod,
    torch::Tensor grad_out,
    torch::Tensor inv,
    torch::Tensor grad_in,
    torch::Tensor inv_len
)
{
    const int n = param.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    grouped_cumprod_backward_kernel<<<blocks, threads>>>(
        param.data_ptr<float>(),
        param_cumprod.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        inv.data_ptr<int>(),
        grad_in.data_ptr<float>(),
        n,
        inv_len.data_ptr<int>()
    );
}


