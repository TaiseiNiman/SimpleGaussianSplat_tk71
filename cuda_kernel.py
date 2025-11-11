from torch.utils.cpp_extension import load
import os

class cuda_kernel():
    
    def __init__(self,Log_output=False,cuda_version="v13.0"):
        
        self.cuda_kernel = load(
                                name="test",
                                sources=["cuda_kernel\\grouped_cumprod.cu"],
                                extra_include_paths=[
                                    f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{cuda_version}\\include",
                                    f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{cuda_version}\\include\cccl"
                                ],
                                verbose=Log_output
                            )