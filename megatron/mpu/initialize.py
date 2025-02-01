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


"""Model and data parallel groups."""

from typing import Optional
import torch

from .utils import ensure_divisibility

# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Pipeline parallel group that the current rank belongs to.
_PIPE_PARALLEL_GROUP = None

# A group used to sync during the IO process. Usually this is data_parallel_group(),
# but with pipeline parallelism it must also involve the last stage (which is not in the
# DP group of rank 0)
_IO_PARALLEL_GROUP = None

# These values enable us to change the mpu sizes on the fly.
_MPU_WORLD_SIZE = None
_MPU_RANK = None

# Used to query 3D topology
_MPU_TOPOLOGY = None

# Get fp32_allreduce flag
_FP32_ALLREDUCE = None


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def initialize_model_parallel(model_parallel_size, topology=None, fp32_allreduce=False):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print("> initializing model parallel with size {}".format(model_parallel_size))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    if world_size < model_parallel_size:
        raise ValueError("world size cannot be smaller than model parallel size")
    ensure_divisibility(world_size, model_parallel_size)
    rank = torch.distributed.get_rank()

    global _MPU_TOPOLOGY
    if topology:
        _MPU_TOPOLOGY = topology

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    if topology:
        for dp_group in topology.get_axis_comm_lists("data"):
            group = torch.distributed.new_group(ranks=dp_group)
            if rank == 0:
                print(f"MPU DP:", dp_group)
            if rank in dp_group:
                _DATA_PARALLEL_GROUP = group
    else:
        for i in range(model_parallel_size):
            ranks = range(i, world_size, model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if i == (rank % model_parallel_size):
                _DATA_PARALLEL_GROUP = group

    # Build pipeline parallel group
    if topology is not None:
        global _PIPE_PARALLEL_GROUP
        for pp_group in topology.get_axis_comm_lists("pipe"):
            group = torch.distributed.new_group(ranks=pp_group)
            if rank == 0:
                print(f"MPU PP:", pp_group)
            if rank in pp_group:
                _PIPE_PARALLEL_GROUP = group

    # Build IO group
    global _IO_PARALLEL_GROUP
    if topology and topology.get_dim("pipe") > 1:
        io_stages = [0, topology.get_dim("pipe") - 1]
        io_group = []
        for stage in io_stages:
            io_group.extend(topology.filter_match(pipe=stage, model=0))
        if rank == 0:
            print(f"MPU IO:", io_group)
        group = torch.distributed.new_group(ranks=io_group)
        if rank in io_group:
            _IO_PARALLEL_GROUP = group
    else:
        _IO_PARALLEL_GROUP = get_data_parallel_group()

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    if topology:
        # Short circuit case without model parallelism.
        # TODO: it would be nice  to avoid this branching case?
        if model_parallel_size == 1:
            for group_rank in range(world_size):
                group = torch.distributed.new_group(ranks=[group_rank])
                if rank == 0:
                    print(f"MPU MP:", [group_rank])
                if rank == group_rank:
                    _MODEL_PARALLEL_GROUP = group
            return

        for mp_group in topology.get_axis_comm_lists("model"):
            group = torch.distributed.new_group(ranks=mp_group)
            if rank == 0:
                print(f"MPU MP:", mp_group)
            if rank in mp_group:
                _MODEL_PARALLEL_GROUP = group

    else:
        for i in range(world_size // model_parallel_size):
            ranks = range(i * model_parallel_size, (i + 1) * model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if i == (rank // model_parallel_size):
                _MODEL_PARALLEL_GROUP = group

    global _FP32_ALLREDUCE
    assert _FP32_ALLREDUCE is None, "fp32_allreduce is already initialized"
    _FP32_ALLREDUCE = fp32_allreduce


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_io_parallel_group():
    """Get the IO parallel group the caller rank belongs to."""
    assert _IO_PARALLEL_GROUP is not None, "IO parallel group is not initialized"
    return _IO_PARALLEL_GROUP


def set_model_parallel_world_size(world_size):
    """Set the model parallel size"""
    global _MPU_WORLD_SIZE
    _MPU_WORLD_SIZE = world_size


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    global _MPU_WORLD_SIZE
    if _MPU_WORLD_SIZE is not None:
        return _MPU_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def set_model_parallel_rank(rank):
    """Set model parallel rank."""
    global _MPU_RANK
    _MPU_RANK = rank


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    global _MPU_RANK
    if _MPU_RANK is not None:
        return _MPU_RANK
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    topo = get_topology()
    if topo is None:
        # we are just using model parallel
        return global_rank % get_model_parallel_world_size()
    else:
        # We are using pipeline parallel
        d = topo.get_axis_comm_lists("data")
        for l in d:
            if global_rank in l:
                return l[0]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_topology():
    return _MPU_TOPOLOGY


def get_pipe_parallel_group():
    """Get the pipe parallel group the caller rank belongs to."""
    assert _PIPE_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _PIPE_PARALLEL_GROUP


def get_pipe_parallel_rank():
    """Return my rank for the pipe parallel group."""
    return torch.distributed.get_rank(group=get_pipe_parallel_group())


def get_pipe_parallel_world_size():
    """Return world size for the pipe parallel group."""
    return torch.distributed.get_world_size(group=get_pipe_parallel_group())


def get_expert_tokens_for_rank(
    routed_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    rank: Optional[int] = None,
):
    """
    Allow user to specify rank, fall back on this device
    """
    # Calculate cumulative sums of tokens_per_expert, ensure the shapes are correct
    world_size = get_model_parallel_world_size()
    if rank is None:
        rank = get_model_parallel_rank()

    # TODO: is this check necessary here/what does it cost us to redundantly do it in multiple places?
    assert tokens_per_expert.shape[0] % world_size == 0

    cumulative_sums = torch.cumsum(tokens_per_expert, dim=0)
    assert cumulative_sums[-1] == routed_tokens.shape[0]

    # select the right starting and ending indices from the cumsum to figure out what tokens to select
    rank_expert_indices = cumulative_sums.chunk(world_size)
    start_index = rank_expert_indices[rank - 1][-1] if rank > 0 else 0
    end_index = rank_expert_indices[rank][-1]

    # Use indices to select the chunk of the tokens matrix
    selected_experts = routed_tokens[start_index:end_index]

    return selected_experts


def get_expert_token_counts_for_rank(
    tokens_per_expert: torch.Tensor, rank: Optional[int] = None
):
    """
    Allow user to specify rank, fall back on this device
    """
    # TODO: add bounds checking of size is 1D for tokens_per_expert
    # should be (num_experts) long
    world_size = get_model_parallel_world_size()
    if rank is None:
        rank = get_model_parallel_rank()

    return tokens_per_expert.chunk(world_size)[rank]


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    set_model_parallel_world_size(world_size)


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    return get_model_parallel_group()


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    return get_model_parallel_rank()


# Needed for MOE. True tensor parallelism todo.
def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_model_parallel_world_size()


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    set_model_parallel_rank(rank)


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_model_parallel_rank()


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _PIPE_PARALLEL_GROUP
    _PIPE_PARALLEL_GROUP = None
    global _IO_PARALLEL_GROUP
    _IO_PARALLEL_GROUP = None
    global _MPU_WORLD_SIZE
    global _MPU_RANK
    _MPU_WORLD_SIZE = None
    _MPU_RANK = None
    global _MPU_TOPOLOGY
    _MPU_TOPOLOGY = None
    global _FP32_ALLREDUCE
    _FP32_ALLREDUCE = None


def get_fp32_allreduce():
    """Get the fp32 allreduce flag"""
    assert _FP32_ALLREDUCE is not None, "fp32_allreduce is not Initialized"
    return _FP32_ALLREDUCE
