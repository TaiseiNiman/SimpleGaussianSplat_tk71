from torch.utils.cpp_extension import load
import torch
import time

import torch
import grouped_cumprod


# M = 4_000_000_00  # 実験用、実際は数億要素でも動く
# K = 100_000
# group_len = M // K

# # グループを昇順に設定
# inv = torch.arange(K, device='cuda').repeat_interleave(group_len).to(torch.int32)

# param = torch.rand(M, device='cuda') + 0.1
# grad = torch.rand(M, device='cuda')

param = torch.tensor([0.4,0.2,0.1,0.8,0.2],device="cuda",dtype=torch.float32)
grad = torch.clone(param)
index = torch.tensor([0,0,1,1,2],device="cuda",dtype=torch.int32)
param_cumprod = torch.zeros_like(param)
grouped_cumprod.grouped_cumprod_forward(param,index,param_cumprod) 

out = torch.zeros_like(param)

index_len = torch.tensor([2,4,5],device="cuda",dtype=torch.int32)
start = time.time()
grouped_cumprod.grouped_cumprod_backward(param, param_cumprod, grad, index, out, index_len)
end = time.time()

print(f"cuda time : {(end - start)*1000}[ms]")

print(f"(0.44,0.08,0.74,0.08,0.2)となるはず.勾配:{out}")
