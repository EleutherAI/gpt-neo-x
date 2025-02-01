# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from its original version
#

import os
import pathlib
import subprocess
import torch
from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with different order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicitly in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load(neox_args=None):

    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    if torch.version.hip is None:
        _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(
            cpp_extension.CUDA_HOME
        )
        if int(bare_metal_major) >= 11:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_80,code=sm_80")
            if int(bare_metal_minor) >= 1:
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_86,code=sm_86")
            elif int(bare_metal_minor) >= 4:
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_87,code=sm_87")
            elif int(bare_metal_minor) >= 8:
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_89,code=sm_89")
        if int(bare_metal_major) >= 12:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    # Determine verbosity
    verbose = True if neox_args is None else (neox_args.rank == 0)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(
        name, sources, extra_cuda_flags, extra_include_paths
    ):
        if torch.version.hip is not None:
            extra_cuda_cflags = ["-O3"] + extra_cuda_flags + cc_flag
        else:
            extra_cuda_cflags = (
                ["-O3", "-gencode", "arch=compute_70,code=sm_70", "--use_fast_math"]
                + extra_cuda_flags
                + cc_flag
            )

        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
            verbose=verbose,
        )

    # ==============
    # Fused softmax.
    # ==============

    if torch.version.hip is not None:
        extra_include_paths = [os.path.abspath(srcpath)]
    else:
        extra_include_paths = []

    if torch.version.hip is not None:
        extra_cuda_flags = [
            "-D__HIP_NO_HALF_OPERATORS__=1",
            "-D__HIP_NO_HALF_CONVERSIONS__=1",
        ]
    else:
        extra_cuda_flags = [
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]

    # Upper triangular softmax.
    sources = [
        srcpath / "scaled_upper_triang_masked_softmax.cpp",
        srcpath / "scaled_upper_triang_masked_softmax_cuda.cu",
    ]
    scaled_upper_triang_masked_softmax_cuda = _cpp_extention_load_helper(
        "scaled_upper_triang_masked_softmax_cuda",
        sources,
        extra_cuda_flags,
        extra_include_paths,
    )
    # Masked softmax.
    sources = [
        srcpath / "scaled_masked_softmax.cpp",
        srcpath / "scaled_masked_softmax_cuda.cu",
    ]
    scaled_masked_softmax_cuda = _cpp_extention_load_helper(
        "scaled_masked_softmax_cuda", sources, extra_cuda_flags, extra_include_paths
    )
    # fused rope
    sources = [
        srcpath / "fused_rotary_positional_embedding.cpp",
        srcpath / "fused_rotary_positional_embedding_cuda.cu",
    ]
    fused_rotary_positional_embedding = _cpp_extention_load_helper(
        "fused_rotary_positional_embedding",
        sources,
        extra_cuda_flags,
        extra_include_paths,
    )


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")


def load_fused_kernels():
    try:
        import scaled_upper_triang_masked_softmax_cuda
        import scaled_masked_softmax_cuda
        import fused_rotary_positional_embedding
    except (ImportError, ModuleNotFoundError) as e:
        print("\n")
        print(e)
        print("=" * 100)
        print(
            f"ERROR: Fused kernels configured but not properly installed. Please run `from megatron.fused_kernels import load()` then `load()` to load them correctly"
        )
        print("=" * 100)
        exit()
    return
