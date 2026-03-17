# Copyright (c) 2024, Jay Shah, Tri Dao.

import os
import glob
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cutlass_include():
    cutlass_path = Path(this_dir).parent / "third_party" / "cutlass" / "include"
    if cutlass_path.exists():
        return str(cutlass_path)
    raise RuntimeError("cutlass not found at third_party/cutlass/include")


sources = [
    "flash_api.cpp",
    "flash_fwd_combine.cu",
    "flash_prepare_scheduler.cu",
] + glob.glob("src/*.cu")

setup(
    name="flash_attn_3",
    version="3.0.0",
    packages=[],
    ext_modules=[
        CUDAExtension(
            name="flash_attn_3",
            sources=sources,
            include_dirs=[
                this_dir,
                os.path.join(this_dir, "src"),
                get_cutlass_include(),
            ],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_90,code=sm_90",
                    "-gencode=arch=compute_100,code=sm_100",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
