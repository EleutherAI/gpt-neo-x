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

"""Megatron initialization."""

import random
import os

import numpy as np
import torch

from megatron import fused_kernels
from megatron import mpu
from megatron.mpu import set_model_parallel_rank, set_model_parallel_world_size

import deepspeed
import inspect


def initialize_megatron(neox_args, allow_no_cuda=False):
    """Set initialize distributed and set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    # torch.distributed initialization
    def finish_mpu_init():
        # Pytorch distributed.
        _initialize_distributed(neox_args=neox_args)

        # Random seeds for reproducibility.
        if neox_args.rank == 0:
            print("> setting random seeds to {} ...".format(neox_args.seed))
        _set_random_seed(neox_args.seed)

    # check fused kernels are installed:
    if (
        neox_args.scaled_upper_triang_masked_softmax_fusion
        or neox_args.scaled_masked_softmax_fusion
        or neox_args.rope_fusion
    ):
        fused_kernels.load(neox_args)
        fused_kernels.load_fused_kernels()

    if neox_args.lazy_mpu_init:
        neox_args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        set_model_parallel_world_size(neox_args.model_parallel_size)
        # and return function for external DDP manager to call when it has DDP initialized
        set_model_parallel_rank(neox_args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Compile dataset C++ code.
        if neox_args.local_rank == 0:
            from megatron.data.data_utils import compile_helper

            compile_helper()

        # Write arguments to tensorboard.
        _write_args_to_tensorboard(neox_args=neox_args)
        # No continuation function
        return None


def setup_deepspeed_random_and_activation_checkpointing(neox_args):
    """Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    """
    num_layers = neox_args.num_layers // neox_args.checkpoint_num_layers
    num_layers = (
        num_layers
        if neox_args.num_layers % neox_args.checkpoint_num_layers == 0
        else num_layers + 1
    )

    deepspeed.checkpointing.configure(
        mpu,
        partition_activations=neox_args.partition_activations,
        contiguous_checkpointing=neox_args.contiguous_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=neox_args.checkpoint_in_cpu,
        synchronize=neox_args.synchronize_each_layer,
        profile=neox_args.profile_backward,
    )


def _initialize_distributed(neox_args):
    """Initialize torch.distributed and mpu."""

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if neox_args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        neox_args.rank = torch.distributed.get_rank()
        neox_args.world_size = torch.distributed.get_world_size()

    else:

        if neox_args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = neox_args.rank % device_count
            if neox_args.local_rank is not None:
                assert (
                    neox_args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                neox_args.local_rank = device
            torch.cuda.set_device(device)

        deepspeed.init_distributed(
            dist_backend=neox_args.distributed_backend,
            auto_mpi_discovery=True,
            distributed_port=os.getenv("MASTER_PORT", "6000"),
            verbose=True,
        )

    # Setup 3D topology.
    pp = neox_args.pipe_parallel_size if neox_args.pipe_parallel_size >= 1 else 1
    mp = neox_args.model_parallel_size if neox_args.model_parallel_size >= 1 else 1
    cp = neox_args.context_parallel_size if neox_args.context_parallel_size >= 1 else 1
    assert (
        neox_args.world_size % (pp * mp * cp) == 0
    ), f"world_size={neox_args.world_size}, pp={pp}, mp={mp}, cp={cp}"
    assert (
        neox_args.world_size % (pp * mp) == 0
    ), f"world_size={neox_args.world_size}, pp={pp}, mp={mp}"
    # The data parallel ranks will be used for context parallel
    # to piggy back the gradient all reduce
    dp = neox_args.world_size // (pp * mp)
    assert dp % cp == 0
    from deepspeed.runtime.pipe.topology import ProcessTopology

    topo = ProcessTopology(axes=["pipe", "data", "model"], dims=[pp, dp, mp])

    # Offset base seeds for the interior pipeline stages.
    # TODO: adjust last stage too once IO is improved.
    stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
    if 0 < stage_id < topo.get_dim("pipe") - 1:
        offset = neox_args.seed + 1138
        neox_args.seed = offset + (stage_id * mp)

    # Set the model-parallel / data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print(
                "_initialize_distributed() model parallel is already initialized",
                flush=True,
            )
        else:
            mpu.initialize_model_parallel(
                neox_args.model_parallel_size,
                neox_args.pipe_parallel_size,
                neox_args.context_parallel_size,
                topology=topo,
                fp32_allreduce=neox_args.fp32_allreduce,
            )

    # Init DeepSpeed Activation Checkpointing Features
    setup_deepspeed_random_and_activation_checkpointing(neox_args=neox_args)


def _init_autoresume(neox_args):
    """Set autoresume start time."""

    if neox_args.adlr_autoresume:
        print_rank_0("> enabling autoresume ...")
        sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
        try:
            from userlib.auto_resume import AutoResume
        except BaseException:
            print("> ADLR autoresume is not available, exiting ...", flush=True)
            sys.exit()
        neox_args.adlr_autoresume_object = AutoResume

    if neox_args.adlr_autoresume_object:
        torch.distributed.barrier()
        neox_args.adlr_autoresume_object.init()
        torch.distributed.barrier()


def _set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def _write_args_to_tensorboard(neox_args):

    """Write arguments to tensorboard."""
    if neox_args.tensorboard_writer:
        for arg_name in vars(neox_args):
            neox_args.tensorboard_writer.add_text(
                arg_name, str(getattr(neox_args, arg_name))
            )
