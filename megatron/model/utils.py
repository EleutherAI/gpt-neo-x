# Copyright (c) 2024 EleutherAI
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

"""Utilities for models."""

import torch
from megatron.model.fused_softmax import SoftmaxFusionTypes
from megatron import mpu
from types import GeneratorType
import torch.distributed as dist

import importlib
from typing import List, Dict, Any


def get_params_for_weight_decay_optimization(module: Any, neox_args: Any):
    """
    Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and biases will have no weight decay but the rest will.
    """
    weight_decay_params = {"params": [], "name": "weight_decay_params"}
    no_weight_decay_params = {
        "params": [],
        "weight_decay": 0.0,
        "name": "no_weight_decay_params",
    }

    def is_no_weight_decay_module(module_: Any) -> bool:
        return (
            type(module_).__name__
            in [
                "LayerNorm",
                "RMSNorm",
                "ScaleNorm",
                "TELayerNorm",
                "TERMSNorm",
                "MixedFusedLayerNorm",
                "MixedFusedRMSNorm",
            ]
            or neox_args.weight_decay == 0.0
        )

    for module_ in module.modules():
        if is_no_weight_decay_module(module_):
            no_weight_decay_params["params"].extend(
                [p for p in module_._parameters.values() if p is not None]
            )
        else:
            for name, param in module_._parameters.items():
                if param is None:
                    continue
                if name == "bias" or getattr(param, "_no_weight_decay", False):
                    no_weight_decay_params["params"].append(param)
                else:
                    weight_decay_params["params"].append(param)

    if neox_args.weight_decay == 0.0:
        # Only return a single param group to minimize calls to compressed_allreduce with onebitadam
        return [no_weight_decay_params]
    return weight_decay_params, no_weight_decay_params


def exists(x):
    return x is not None


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class SequentialWrapper(torch.nn.Module):
    """
    Used to convert a deepspeed PipelineModule to an nn.Sequential like model whilst retaining
    activation checkpointing.
    """

    def __init__(
        self,
        layers,
        activation_checkpoint_interval,
        activation_checkpoint_func,
        parent_class_name=None,
    ):
        super().__init__()
        self.sequential = torch.nn.Sequential(*layers)
        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.parent_class_name = parent_class_name
        self.activation_checkpoint_func = activation_checkpoint_func
        self.batch_fn = None

    def _is_checkpointable(self, funcs):
        if self.parent_class_name == "GPT2ModelPipe":
            return all(
                "ParallelTransformerLayerPipe" in f.__class__.__name__ for f in funcs
            )
        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        self.batch_fn = fn

    def inference_mode(self, use_cache=True):
        """
        Sets up the model for inference by turning on k/v caching (if specified) and setting `parallel output` of the final layer to false,
        so logits are gathered across model parallel ranks.

        :param cache: (bool) True if you want to use caching during inference, False otherwise
        """
        _set_use_cache(self.sequential, use_cache)
        recursive_setattr(self.sequential, "training", False)

    def train_mode(self):
        """
        Sets up the model for training by turning off k/v caching.
        """
        _set_use_cache(self.sequential, False)
        recursive_setattr(self.sequential, "training", True)

    def forward(
        self, forward_input, curriculum_seqlen=None, labels=None, neox_args=None
    ):

        if self.batch_fn:
            forward_input = self.batch_fn(forward_input)

        if (
            curriculum_seqlen is not None
            and isinstance(forward_input, tuple)
            and len(forward_input) == 3
        ):
            neox_args.update_value("curriculum_seqlen", curriculum_seqlen)
            tokens = forward_input[0]
            input_ids = forward_input[1]
            attention_mask = forward_input[2]
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                # position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                if labels is not None:
                    labels = labels[:, :curriculum_seqlen].contiguous()
                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[
                    :, :, :curriculum_seqlen, :curriculum_seqlen
                ].contiguous()
            forward_input = (tokens, input_ids, attention_mask)

        moe_losses = []

        def exec_range_func(start, end):
            """Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            """

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.sequential[start:end]):
                    inputs = layer(inputs)
                    if hasattr(layer, "last_moe_loss"):
                        moe_losses.append(layer.last_moe_loss)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.sequential))
            x = func(forward_input)
        else:
            num_layers = len(self.sequential)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(
                    start_idx + self.activation_checkpoint_interval, num_layers
                )

                funcs = self.sequential[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x,)

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(
                        exec_range_func(start_idx, end_idx), *x
                    )
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x, moe_losses

    def clear_cache(self):
        """
        Recursively clears the kv cache on all layers
        """
        recursive_setattr(self.sequential, "layer_past", None)


def recursive_setattr(m, attr, value, assert_type=None, type_filter=None):
    """
    Recursively set attributes on a pytorch module or an iterable of modules.
    If an assert_type is provided, it will assert that the type of the value is the same as the assert_type.
    If a type_filter is provided, it will only set attributes on modules that match that type.
    """
    if assert_type is not None:
        assert isinstance(value, assert_type), "Value is not the correct type."

    # if m is a list or a generator, iterate over the elements
    if isinstance(m, (list, GeneratorType)):
        for i in m:
            recursive_setattr(i, attr, value, assert_type, type_filter)
    elif isinstance(m, torch.nn.Module):
        if hasattr(m, attr):
            if type_filter is None or isinstance(m, type_filter):
                setattr(m, attr, value)
        if hasattr(m, "children"):
            recursive_setattr(m.children(), attr, value, assert_type, type_filter)


def _set_use_cache(modules, value: bool):
    """
    Recursively sets an use_cache to `value` on a list of pytorch modules, if they have a use_cache attribute.
    use_cache is used to decide whether we cache past key value activations or not in inference.
    """
    recursive_setattr(modules, "use_cache", value, assert_type=bool)


def configure_sparse_attention(neox_args, attention_type, num_attention_heads, mpu):
    from deepspeed.ops.sparse_attention import (
        SparseSelfAttention,
        VariableSparsityConfig,
        FixedSparsityConfig,
        BigBirdSparsityConfig,
        BSLongformerSparsityConfig,
    )
    from deepspeed.ops.sparse_attention.sparsity_config import (
        LocalSlidingWindowSparsityConfig,
    )

    if attention_type == "sparse_fixed":
        # you can think of local window size as `block_size` * `num_local_blocks`.
        # so if you wanted to set a local window size of 256, set block size to 16 and `num_local_blocks` to 16
        sparsity_config = FixedSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            different_layout_per_head=neox_args.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_local_blocks=neox_args.sparsity_config.get("num_local_blocks", 4),
            num_global_blocks=neox_args.sparsity_config.get("num_global_blocks", 1),
            num_different_global_patterns=neox_args.sparsity_config.get(
                "num_different_global_patterns", 1
            ),
            attention="unidirectional",
            horizontal_global_attention=False,
        )
    elif attention_type == "sparse_variable":
        sparsity_config = VariableSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            different_layout_per_head=neox_args.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_random_blocks=neox_args.sparsity_config.get("num_random_blocks", 0),
            local_window_blocks=neox_args.sparsity_config.get(
                "local_window_blocks", [4]
            ),
            global_block_indices=neox_args.sparsity_config.get(
                "global_block_indices", [0]
            ),
            global_block_end_indices=neox_args.sparsity_config.get(
                "global_block_end_indices", None
            ),
            attention="unidirectional",
            horizontal_global_attention=False,
        )
    elif attention_type == "local":
        # can configure with `num_local_blocks` or `num_sliding_window_blocks`
        num_local_blocks = neox_args.sparsity_config.get(
            "num_local_blocks",
            neox_args.sparsity_config.get("num_sliding_window_blocks", 4),
        )
        sparsity_config = LocalSlidingWindowSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            num_sliding_window_blocks=num_local_blocks,
            attention="unidirectional",
        )
    elif attention_type == "bigbird":
        sparsity_config = BigBirdSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            different_layout_per_head=neox_args.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_random_blocks=neox_args.sparsity_config.get("num_random_blocks", 1),
            num_sliding_window_blocks=neox_args.sparsity_config.get(
                "num_sliding_window_blocks", 3
            ),
            num_global_blocks=neox_args.sparsity_config.get("num_global_blocks", 1),
            attention="unidirectional",
        )
    elif attention_type == "bslongformer":
        sparsity_config = BSLongformerSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            different_layout_per_head=neox_args.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_sliding_window_blocks=neox_args.sparsity_config.get(
                "num_sliding_window_blocks", 3
            ),
            global_block_indices=neox_args.sparsity_config.get(
                "global_block_indices", [0]
            ),
            global_block_end_indices=neox_args.sparsity_config.get(
                "global_block_end_indices", None
            ),
            attention="unidirectional",
        )
    else:
        raise ValueError(f"Attention type {attention_type} not recognized")
    return SparseSelfAttention(
        sparsity_config=sparsity_config,
        max_seq_length=neox_args.seq_length,
        attn_mask_mode="add",
        mpu=mpu,
    )


def get_fusion_type(neox_args):
    fusion_type = SoftmaxFusionTypes.none
    if neox_args.scaled_upper_triang_masked_softmax_fusion:
        fusion_type = SoftmaxFusionTypes.upper_triang
    elif neox_args.scaled_masked_softmax_fusion:
        fusion_type = SoftmaxFusionTypes.general
    return fusion_type


def reduce_weight_grads_from_model_parallel_region(input_):
    """A hook that can be applied to any weight tensor via .register_hook().
    Allreduces grads for e.g. LN weights across the model parallel group.
    Needed to keep LNs in sync, despite them getting diff data -> diff gradients when using sequence parallel.
    """
    # Bypass the function if no TP -> no comm needed.
    if mpu.get_model_parallel_world_size() == 1:
        return input_

    # Bf16 convert
    dt = input_.dtype
    if dt == torch.bfloat16 and mpu.initialize.get_fp32_allreduce():
        input_ = input_.float()

    # All-reduce.
    dist.all_reduce(input_, group=mpu.get_model_parallel_group())

    # Bf16 convert
    if dt == torch.bfloat16 and mpu.initialize.get_fp32_allreduce():
        input_ = input_.bfloat16()

    return input_


def mark_norms_for_sequence_parallel_grad_sync(module, neox_args):
    """Iterate through the modules in our model, and for any "...Norm" classnames,
    register a hook on each of that module's parameters which will allreduce norms' weights' grads across
    the model (sequence) parallel region.
    """

    if not neox_args.sequence_parallel:
        # if we aren't using sequence parallelism, this is a no-op
        return

    for module_ in module.modules():
        if "norm" in type(module_).__name__.lower():
            # this is a norm, we want to allreduce its weight grads across sequence parallel region
            for name, param in module_.named_parameters():
                if param.requires_grad:
                    param.register_hook(reduce_weight_grads_from_model_parallel_region)


def get_parallel_linear(neox_args):
    if neox_args.te_columnparallel:
        from megatron.model.transformer_engine import (
            TEColumnParallelLinear as ColumnParallelLinear,
        )
    else:
        from megatron.mpu import ColumnParallelLinear
    if neox_args.te_rowparallel:
        from megatron.model.transformer_engine import (
            TERowParallelLinear as RowParallelLinear,
        )
    else:
        from megatron.mpu import RowParallelLinear

    return ColumnParallelLinear, RowParallelLinear
