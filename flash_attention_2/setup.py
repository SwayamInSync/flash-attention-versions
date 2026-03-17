# Copyright (c) 2024, Tri Dao.

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


sources = ["flash_api.cpp"] + glob.glob("src/*.cu")

setup(
    name="flash_attn_2",
    version="2.0.0",
    packages=[],
    ext_modules=[
        CUDAExtension(
            name="flash_attn_2",
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
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
