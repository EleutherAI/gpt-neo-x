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

import torch

from .initialize import (
    get_expert_tokens_for_rank,
    get_model_parallel_group,
    get_model_parallel_world_size,
    get_model_parallel_rank,
    get_fp32_allreduce,
    get_expert_token_counts_for_rank,
)
from .utils import split_tensor_along_last_dim, split_tensor_along_any_dim


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_model_parallel_world_size() == 1:
        return input_

    # upcast to fp32 if using fp32 allreduce
    dt = input_.dtype
    if get_fp32_allreduce():
        input_ = input_.float()

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_model_parallel_group())

    # reconvert to original Bf16/Fp16 dtype
    if get_fp32_allreduce():
        input_ = input_.to(dt)

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _dmoe_reduce(input_, tokens_per_expert):
    """All-reduce the the dMoE input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if get_model_parallel_world_size() == 1:
        return input_

    # Bf16 convert
    dt = input_.dtype
    if dt == torch.bfloat16 and get_fp32_allreduce():
        input_ = input_.float()

    output = torch.zeros(
        (sum(tokens_per_expert), input_.shape[-1]),
        dtype=input_.dtype,
        device=input_.device,
    )
    world_size = get_model_parallel_world_size()
    rank = get_model_parallel_rank()

    cumulative_sums = torch.cumsum(tokens_per_expert, dim=0)

    # select the right starting and ending indices from the cumsum to figure out what tokens to select
    rank_expert_indices = cumulative_sums.chunk(world_size)
    start_index = rank_expert_indices[rank - 1][-1] if rank > 0 else 0
    end_index = rank_expert_indices[rank][-1]

    output[start_index:end_index] = input_

    # All-reduce.
    torch.distributed.all_reduce(output, group=get_model_parallel_group())

    # Bf16 convert
    if dt == torch.bfloat16 and get_fp32_allreduce():
        output = output.bfloat16()

    return output


def _dmoe_split(input_, tokens_per_expert):
    """Split the tensor along its first dimension according to where tokens
    were routed, keeping the corresponding slice."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension, getting the expert tokens
    output = get_expert_tokens_for_rank(input_, tokens_per_expert)

    return output


def _dmoe_gather(input_: torch.Tensor, tokens_per_expert: torch.Tensor):
    """Gather tensors and concatinate along the first dimension)"""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Bf16 convert
    dt = input_.dtype
    if dt == torch.bfloat16 and get_fp32_allreduce():
        input_ = input_.float()

    # Gather along first dimension
    gather_dim = 0
    rank = get_model_parallel_rank()

    tokens_by_rank = [
        get_expert_token_counts_for_rank(tokens_per_expert, r)
        for r in range(world_size)
    ]
    # print(f"{torch.cuda.current_device()}: tokens_by_rank {tokens_by_rank}")
    tensor_list = [
        torch.empty(sum(r), input_.shape[-1], device=input_.device, dtype=input_.dtype)
        for r in tokens_by_rank
    ]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=gather_dim)

    # Bf16 convert
    if dt == torch.bfloat16 and get_fp32_allreduce():
        output = output.bfloat16()

    return output


def _reduce_scatter_along_seq_dim(input_, seq_dim):
    """Reduce-scatter the input tensor across model parallel group, scattering across sequence dim."""
    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # upcast to fp32 if using fp32 allreduce
    dt = input_.dtype
    if get_fp32_allreduce():
        input_ = input_.float()

    dim_size = list(input_.size())
    assert (
        isinstance(seq_dim, int) and seq_dim < len(dim_size) and seq_dim >= 0
    ), "seq_dim must be a valid tensor dim"
    assert dim_size[seq_dim] % world_size == 0

    if seq_dim == 0:
        # reduce_scatter_tensor is faster but only works correctly on dimension 0
        dim_size[seq_dim] = dim_size[seq_dim] // world_size
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
        )
        torch.distributed.reduce_scatter_tensor(
            output, input_.contiguous(), group=get_model_parallel_group()
        )
    else:
        tensor_list = list(
            torch.split(input_, input_.shape[seq_dim] // world_size, seq_dim)
        )
        output = torch.empty_like(tensor_list[0])
        torch.distributed.reduce_scatter(
            output, tensor_list, group=get_model_parallel_group()
        )

    # reconvert to original Bf16/Fp16 dtype
    if get_fp32_allreduce():
        output = output.to(dt)

    return output


def _gather_along_seq_dim(input_, seq_dim):
    """Gather tensors and concatinate along the (manually-specified) sequence dimension."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
        isinstance(seq_dim, int) and seq_dim < len(dim_size) and seq_dim >= 0
    ), "seq_dim must be a valid tensor dim"
    dim_size[seq_dim] = dim_size[seq_dim] * world_size

    if seq_dim == 0:
        # reduce_gather_tensor is faster but only works correctly on dimension 0
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
        )
        torch.distributed.all_gather_into_tensor(
            output, input_.contiguous(), group=get_model_parallel_group()
        )
    else:
        input_ = input_.contiguous()
        rank = get_model_parallel_rank()
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(
            tensor_list, input_, group=get_model_parallel_group()
        )
        output = torch.cat(tensor_list, dim=seq_dim)

    return output


def _split_along_seq_dim(input_, seq_dim):
    """Split the tensor along the sequence dimension (as manually selected) and keep the
    corresponding slice."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along second dimension.
    input_list = split_tensor_along_any_dim(input_, world_size, seq_dim)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _CopyToExpertModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, tokens_per_expert):
        # TODO: not sure if this is sufficient? not sure how this gets used downstream...
        return get_expert_tokens_for_rank(input_, tokens_per_expert)

    @staticmethod
    def forward(ctx, input_, tokens_per_expert):
        # Save tokens_per_expert in the context for later use in the backward pass
        ctx.save_for_backward(tokens_per_expert)

        return get_expert_tokens_for_rank(input_, tokens_per_expert)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the tokens_per_expert from the context
        (tokens_per_expert,) = ctx.saved_tensors

        # no grad for tokens_per_expert
        # return _dmoe_reduce(grad_output, tokens_per_expert), None
        return _dmoe_gather(grad_output, tokens_per_expert), None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


class _GatherFromExpertModelParallelRegion(torch.autograd.Function):
    """Gather the input from expert model parallel region and concatinate.

    The major difference between this and _GatherFromModelParallelRegion is in the
    dMoE case, we need to gather & split along the first dimension, not the last
    """

    @staticmethod
    def symbolic(graph, input_, tokens_per_expert):
        # TODO: not sure if this is sufficient? not sure how this gets used downstream...
        return _dmoe_gather(input_, tokens_per_expert)

    @staticmethod
    def forward(ctx, input_, tokens_per_expert):
        # Save tokens_per_expert in the context for later use in the backward pass
        ctx.save_for_backward(tokens_per_expert)

        return _dmoe_gather(input_, tokens_per_expert)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the tokens_per_expert from the context
        (tokens_per_expert,) = ctx.saved_tensors

        # no grad for tokens_per_expert
        return _dmoe_split(grad_output, tokens_per_expert), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce-Scatter across sequence parallel region (same as model parallel region.)
    Note: same region as model parallel region
    """

    @staticmethod
    def symbolic(graph, input_, seq_dim):
        return _reduce_scatter_along_seq_dim(input_, seq_dim=seq_dim)

    @staticmethod
    def forward(ctx, input_, seq_dim):
        ctx.seq_dim = seq_dim
        return _reduce_scatter_along_seq_dim(input_, seq_dim=seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        seq_dim = ctx.seq_dim
        return _gather_along_seq_dim(grad_output, seq_dim=seq_dim), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """All-Gather across sequence parallel region (same region as model parallel region.)"""

    @staticmethod
    def symbolic(graph, input_, seq_dim):
        return _gather_along_seq_dim(input_, seq_dim=seq_dim)

    @staticmethod
    def forward(ctx, input_, seq_dim):
        ctx.seq_dim = seq_dim
        return _gather_along_seq_dim(input_, seq_dim=seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        seq_dim = ctx.seq_dim
        return _reduce_scatter_along_seq_dim(grad_output, seq_dim=seq_dim), None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Scatter (split) sequence length across sequence parallel region (=> same region as model parallel.)"""

    @staticmethod
    def symbolic(graph, input_, seq_dim):
        return _split_along_seq_dim(input_, seq_dim=seq_dim)

    @staticmethod
    def forward(ctx, input_, seq_dim):
        ctx.seq_dim = seq_dim
        return _split_along_seq_dim(input_, seq_dim=seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        seq_dim = ctx.seq_dim
        return (
            _gather_along_seq_dim(grad_output, seq_dim=seq_dim),
            None,
        )


# -----------------
# Helper functions.
# -----------------


def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def copy_to_expert_model_parallel_region(input_, tokens_per_expert):
    return _CopyToExpertModelParallelRegion.apply(input_, tokens_per_expert)


def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def gather_from_expert_model_parallel_region(input_, tokens_per_expert):
    return _GatherFromExpertModelParallelRegion.apply(input_, tokens_per_expert)


def reduce_scatter_to_sequence_parallel_region(input_, seq_dim=0):
    return _ReduceScatterToSequenceParallelRegion.apply(input_, seq_dim)


def gather_from_sequence_parallel_region(input_, seq_dim=0):
    return _GatherFromSequenceParallelRegion.apply(input_, seq_dim)


def scatter_to_sequence_parallel_region(
    input_, seq_dim=1
):  # use this fn in scattering input embeds across TP ranks. There, shape of inps is [b, s, h] instead of the usual [s, b, h]
    return _ScatterToSequenceParallelRegion.apply(input_, seq_dim)
