from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
cuda_path = os.environ.get("CUDA_PATH")
setup(
    name='grouped_cumprod',
    ext_modules=[
        CUDAExtension(
            name='grouped_cumprod',  # 出力されるモジュール名
            sources=[
                'cuda_kernel\\cuda_kernel.cpp',
                'cuda_kernel\\grouped_cumprod_forward.cu',
                'cuda_kernel\\grouped_cumprod_backward.cu'
            ],
            include_dirs=[
                os.path.join(cuda_path,"include"),
                os.path.join(cuda_path,"include", "cccl")
            ],
            # extra_compile_args={
            #     'cxx':  ["/O2", "/openmp", "/std:c++20", "/DENABLE_BF16"],
            #     'nvcc': [
            #         '-O3',
            #         '--use_fast_math',
            #         '-lineinfo',
            #         '-Xcompiler', '/openmp',
            #     ],
            # }
            # extra_compile_args={
            #     'cxx': ['/O2'],
            #     'nvcc': [
            #         '-O3',
            #         '--use_fast_math',
            #         '-lineinfo',
            #         '-Xcompiler', '/openmp',
            #         '-D__CUDACC_RTC__',               # 重要: RTC用の曖昧性回避
            #         '-DTHRUST_IGNORE_CUB_VERSION_CHECK',
            #         '-DNOMINMAX',                     # Windowsのmin/maxマクロ対策
            #         '-D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING',
            #         '-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS',
            #         '-DWIN32_LEAN_AND_MEAN',          # winuser.h衝突回避
            #     ],
            # }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)