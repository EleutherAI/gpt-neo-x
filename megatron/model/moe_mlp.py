# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2023 MegaBlocks authors
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
from megatron.model.activations import get_activation

from megatron.mpu.layers import _initialize_affine_weight_gpu
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.utils import divide

from megatron.neox_arguments.arguments import NeoXArgs

from megablocks import grouped_gemm_util as gg


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        return grad * ctx.scale, None


scale_gradient = ScaleGradient.apply


class MemoryOptimizedParallelGroupedMLP(torch.autograd.Function):
    """GroupedMLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w1, w2, batch_sizes, activation_fn):
        # x: [m, k], w1: [n, k], w2: [n, k]
        if not x.is_contiguous() or not w1.is_contiguous() or not w2.is_contiguous():
            raise ValueError("Expected contiguous 'x', 'w1' and 'w2'.")

        # Layer 0: x @ w1.t().
        sdd_out = gg.backend.gmm(x, w1, batch_sizes, trans_b=True)

        # activation_fn
        activation_fn_out = activation_fn(sdd_out)

        # Layer 1: x @ w2.
        dsd_out = gg.backend.gmm(activation_fn_out, w2, batch_sizes)

        # NOTE: Save the input to the layer and the activation_fn input for
        # gradient computation. We'll re-compute the activation_fn forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.x_shape = x.shape
        ctx.sdd_out_shape = sdd_out.shape
        ctx.dtype = x.dtype
        ctx.activation_fn = activation_fn
        ctx.save_for_backward(w1, w2, batch_sizes, x, sdd_out)
        return dsd_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, ddsd_out):
        if (
            not ctx.needs_input_grad[0]
            or not ctx.needs_input_grad[1]
            or not ctx.needs_input_grad[2]
        ):
            raise ValueError("Expected all MLP inputs to need grad.")

        # Unpack saved tensors
        dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, w2 = saved_tensors[:2]
        batch_sizes = saved_tensors[2]
        x = saved_tensors[3]
        sdd_out = saved_tensors[4]

        # Rematerialize activation_fn output.
        activation_fn = ctx.activation_fn
        with torch.set_grad_enabled(True):
            sdd_out.requires_grad = True
            activation_fn_out = activation_fn(sdd_out)
            activation_grad_fn = activation_fn_out.backward

        # Compute dw2 with recomputed activation_fn output.
        dw2 = gg.backend.gmm(activation_fn_out, ddsd_out, batch_sizes, trans_a=True)

        # Compute dactivation_fn_out.
        #
        # NOTE: We reuse the activation_fn_out allocation.
        dactivation_fn_out = activation_fn_out
        gg.backend.gmm(ddsd_out, w2, batch_sizes, trans_b=True, c=dactivation_fn_out)

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dactivation_fn_out allocation.
        if activation_fn is DEFAULT_ACTIVATION_FN:
            dsdd_out = gelu.gelu_backward_(dactivation_fn_out, sdd_out)
        else:
            assert activation_grad_fn is not None
            activation_grad_fn(dactivation_fn_out)
            dsdd_out = sdd_out.grad

        # Compute dw1.
        dw1 = gg.backend.gmm(dsdd_out, x, batch_sizes, trans_a=True)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        gg.backend.gmm(dsdd_out, w1, batch_sizes, c=ddsd_out)
        dx = ddsd_out
        return dx, dw1, dw2, None, None


memory_optimized_grouped_mlp = MemoryOptimizedParallelGroupedMLP.apply


class ParallelGroupedMLP(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
        stride=1,
        multiple_of=256,
    ):
        """
        Copied from SparseMLP
        """
        super(ParallelGroupedMLP, self).__init__()

        self.activation_func, self.activation_fn_is_gated = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)

        self.hidden_size = neox_args.hidden_size

        # Allow custom intermediate size
        if neox_args.intermediate_size is not None:
            per_expert_ff_dim = neox_args.intermediate_size
        # Otherwise, 4 x hidden size, padded to multiple of 256
        else:
            per_expert_ff_dim = 4 * self.hidden_size
            per_expert_ff_dim = self.multiple_of * (
                (per_expert_ff_dim + multiple_of - 1) // multiple_of
            )

        self.per_expert_ff_dim = per_expert_ff_dim
        # number of rows per rank is the number of experts * ff dimension
        self.num_rows_per_rank = self.experts_per_rank * per_expert_ff_dim

        # input
        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w1, init_method, partition_dim=0, stride=stride
        )

        # output
        self.w2 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w2, output_layer_init_method, partition_dim=0, stride=stride
        )

        # TODO: why do we need this? was in original megablocks code
        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size

    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor):
        grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs
        w1 = w1.view(self.experts_per_rank, -1, self.hidden_size)
        w2 = w2.view(self.experts_per_rank, -1, self.hidden_size)

        # Compute the MLP
        x = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes, trans_b=True)
        x = self.activation_func(x)
        return gg.ops.gmm(x, w2, grouped_gemm_batch_sizes)


class MemoryOptimizedParallelGroupedLLaMAMLP(torch.autograd.Function):
    """GroupedMLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w1, w3, w2, batch_sizes, activation_fn):
        # x: [m, k], w1: [n, k], w3: [n, k], w2: [n, k]
        if (
            not x.is_contiguous()
            or not w1.is_contiguous()
            or not w3.is_contiguous()
            or not w2.is_contiguous()
        ):
            raise ValueError("Expected contiguous 'x', 'w1', 'w3' and 'w2'.")

        # Layer 0: x @ w1.t().
        sdd_out = gg.backend.gmm(x, w1, batch_sizes, trans_b=True)
        w3_out = gg.backend.gmm(x, w3, batch_sizes, trans_b=True)

        # GeLU.
        activation_fn_out = activation_fn(sdd_out) * w3_out

        # Layer 1: x @ w2.
        dsd_out = gg.backend.gmm(activation_fn_out, w2, batch_sizes)

        # NOTE: Save the input to the layer and the activation_fn input for
        # gradient computation. We'll re-compute the activation_fn forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.x_shape = x.shape
        ctx.sdd_out_shape = sdd_out.shape
        ctx.dtype = x.dtype
        ctx.activation_fn = activation_fn
        ctx.save_for_backward(w1, w3, w2, batch_sizes, x, sdd_out, w3_out)
        return dsd_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, ddsd_out):
        if (
            not ctx.needs_input_grad[0]
            or not ctx.needs_input_grad[1]
            or not ctx.needs_input_grad[2]
        ):
            raise ValueError("Expected all MLP inputs to need grad.")

        # Unpack saved tensors
        dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, w3, w2 = saved_tensors[:3]
        batch_sizes = saved_tensors[3]
        x = saved_tensors[4]
        sdd_out, w3_out = saved_tensors[5:7]

        # Rematerialize activation_fn output.
        activation_fn = ctx.activation_fn
        with torch.set_grad_enabled(True):
            sdd_out.requires_grad = True
            w3_out.requires_grad = True
            activation_fn_out = activation_fn(sdd_out) * w3_out
            activation_grad_fn = activation_fn_out.backward

        # Compute dw2 with recomputed activation_fn output.
        dw2 = gg.backend.gmm(activation_fn_out, ddsd_out, batch_sizes, trans_a=True)

        # Compute dactivation_fn_out.
        #
        # NOTE: We reuse the activation_fn_out allocation.
        dactivation_fn_out = activation_fn_out
        gg.backend.gmm(ddsd_out, w2, batch_sizes, trans_b=True, c=dactivation_fn_out)

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dactivation_fn_out allocation.
        assert activation_grad_fn is not None
        activation_grad_fn(dactivation_fn_out)
        dsdd_out = sdd_out.grad
        dw3_out = w3_out.grad

        # Compute dw1.
        dw1 = gg.backend.gmm(dsdd_out, x, batch_sizes, trans_a=True)

        # Compute dw3.
        dw3 = gg.backend.gmm(dw3_out, x, batch_sizes, trans_a=True)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        dx = ddsd_out
        gg.backend.gmm(dsdd_out, w1, batch_sizes, c=dx)
        dx += gg.backend.gmm(dw3_out, w3, batch_sizes)
        return dx, dw1, dw3, dw2, None, None


memory_optimized_grouped_llama_mlp = MemoryOptimizedParallelGroupedLLaMAMLP.apply


class ParallelGroupedLLaMAMLP(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
        stride=1,
        multiple_of=256,
    ):
        """
        Copied from SparseMLP
        """
        super(ParallelGroupedLLaMAMLP, self).__init__()

        self.activation_func, self.activation_fn_is_gated = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)

        self.hidden_size = neox_args.hidden_size

        # Allow custom intermediate size
        if neox_args.intermediate_size is not None:
            per_expert_ff_dim = neox_args.intermediate_size
        # Otherwise, 8/3 x hidden size, padded to multiple of 256
        # TODO: why is this how we formulate it this way?
        else:
            per_expert_ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
            per_expert_ff_dim = self.multiple_of * (
                (per_expert_ff_dim + multiple_of - 1) // multiple_of
            )

        self.per_expert_ff_dim = per_expert_ff_dim
        # number of rows per rank is the number of experts * ff dimension per expert
        self.num_rows_per_rank = self.experts_per_rank * per_expert_ff_dim

        # input
        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w1, init_method, partition_dim=0, stride=stride
        )

        # gate
        self.w3 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w3, init_method, partition_dim=0, stride=stride
        )

        # output
        self.w2 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w2, output_layer_init_method, partition_dim=0, stride=stride
        )

        # TODO: why do we need this? was in original megablocks code
        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size

    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor):
        grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w3, w2 = (
            self.scale_grad(self.w1),
            self.scale_grad(self.w3),
            self.scale_grad(self.w2),
        )

        w1 = self.w1.view(self.experts_per_rank, -1, self.hidden_size)
        w3 = w3.view(self.experts_per_rank, -1, self.hidden_size)

        w2 = w2.view(self.experts_per_rank, -1, self.hidden_size)

        # return memory_optimized_grouped_llama_mlp(
        #     x,
        #     w1,
        #     w3,
        #     w2,
        #     grouped_gemm_batch_sizes,
        #     self.activation_func
        # )

        llama_x_w1T = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes, trans_b=True)

        llama_x_w3T = gg.ops.gmm(x, w3, grouped_gemm_batch_sizes, trans_b=True)

        llama_act_x_w1T = self.activation_func(llama_x_w1T)

        # self.w2(self.activation_func(w1_out) * w3_out)
        llama_mlp_out = gg.ops.gmm(
            llama_act_x_w1T
            * llama_x_w3T,  # activation results gated (element-wise) with w3
            w2,  # w2
            grouped_gemm_batch_sizes,  # batch_sizes
        )

        return llama_mlp_out
