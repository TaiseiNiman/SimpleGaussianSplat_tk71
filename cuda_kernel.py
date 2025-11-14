import os
import torch
from torch.utils.cpp_extension import load
#カーネル呼び出しと自動微分のグラフ構築クラス
class cuda_kernel():
    
    def __init__(self,Log_output=True,cuda_version="v13.0"):
        
        self.cuda_kernel = load(
                                name="grouped_cumprod",
                                sources=["cuda_kernel\\grouped_cumprod.cu","cuda_kernel\\grouped_cumprod_backward.cu"],
                                extra_include_paths=[
                                    f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{cuda_version}\\include",
                                    f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{cuda_version}\\include\cccl"
                                ],
                                verbose=Log_output
                            )
    
a = cuda_kernel()
        