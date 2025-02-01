# Copyright (c) 2024, EleutherAI
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

import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from megatron.model.transformer import Gated_Activation
from megatron.model.activations import get_activation
from megatron.mpu.initialize import get_model_parallel_rank
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.initialize import get_tensor_model_parallel_group
from megatron.mpu.mappings import copy_to_model_parallel_region
from megatron.mpu.mappings import gather_from_model_parallel_region
from megatron.mpu.mappings import reduce_from_model_parallel_region
from megatron.mpu.mappings import scatter_to_model_parallel_region
from megatron.mpu.mappings import reduce_scatter_to_sequence_parallel_region
from megatron.mpu.mappings import gather_from_sequence_parallel_region
from megatron.mpu.layers import (
    _initialize_affine_weight_gpu,
    _initialize_affine_weight_cpu,
)
from megatron.mpu.random import get_cuda_rng_tracker
from megatron.mpu.utils import divide
from megatron.mpu.utils import VocabUtility
from functools import partial
from megatron.model.positional_embeddings import RotaryEmbedding
from megatron import mpu

# https://github.com/NVIDIA/TransformerEngine/issues/405
import os
os.environ['NVTE_TORCH_COMPILE'] = str(0)

try:
    import transformer_engine as te
except ImportError:
    raise ImportError(
        "Unable to import transformer-engine. Please refer to "
        "https://github.com/NVIDIA/TransformerEngine for installation instructions."
    )



class TERMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-8, **kwargs):
        """
            A conditional wrapper to initialize an instance of Transformer-Engine's
            `RMSNorm` based on input
        :param dim: model size
        :param eps:  epsilon value, default 1e-8
        """
        super(TERMSNorm, self).__init__()

        self.d = dim
        self.eps = eps
        self.norm = te.pytorch.RMSNorm(
            hidden_size=self.d,
            eps=self.eps,
            **kwargs,
        )

    def forward(self, x):
        return self.norm(x)


class TELayerNorm(torch.nn.Module):
    def __init__(self, dim, eps=1.0e-5, **kwargs):
        """
            A conditional wrapper to initialize an instance of Transformer-Engine's
            `LayerNorm` based on input
        :param dim: model size
        :param eps:  epsilon value, default 1.0e-5
        """
        super(TELayerNorm, self).__init__()

        self.d = dim
        self.eps = eps
        self.norm = te.pytorch.LayerNorm(
            hidden_size=self.d,
            eps=self.eps,
            **kwargs,
        )

    def forward(self, x):
        return self.norm(x)


class TELinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        init_method=init.xavier_normal_,
        stride=1,
        skip_bias_add=False,
        mup_rescale_parameters=False,
        seq_dim=0,
    ):
        self.input_size = input_size
        self.output_size = output_size

        self.skip_bias_add = skip_bias_add
        self.use_bias = bias

        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_dim = seq_dim

        self.init_method = init_method
        self.stride = stride
        self.mup_rescale_parameters = mup_rescale_parameters
        self.use_mup = neox_args.use_mup
        self.params_dtype = neox_args.params_dtype

        super(TELinear, self).__init__(
            in_features=self.input_size,
            out_features=self.output_size,
            bias=self.use_bias,
            init_method=self.init_method,
            get_rng_state_tracker=get_cuda_rng_tracker,
            device=torch.cuda.current_device(),
            return_bias=self.skip_bias_add,
            params_dtype=self.params_dtype,
        )

    def forward(self, inp, **kwargs):
        if self.use_mup and self.mup_rescale_parameters:
            input_ /= self.width_mult()

        output = super(TELinear, self).forward(inp, **kwargs)

        if self.skip_bias_add:
            return output
        else:
            return output, None


class TELayerNormMLP(te.pytorch.LayerNormMLP):
    """
    Wrapper for the Transformer-Engine's `LayerNormMLP` layer that combines
    layernorm and followed by the MLP module, consisting of 2 successive
    linear transformations, separated by the GeLU activation.
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=256,
        MOE=False,
        MoE_mp_size=1,
        bias=True,
    ):
        self.activation_func, self.is_gated = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.multiple_of = multiple_of
        self.bias = bias
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.world_size = world_size
        self.tp_group = get_tensor_model_parallel_group()
        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_len = neox_args.seq_length
        self.micro_batch_size = neox_args.train_micro_batch_size_per_gpu
        self.params_dtype = neox_args.params_dtype
        self.set_parallel_mode = False
        if world_size > 1:
            self.set_parallel_mode = True

        if neox_args.intermediate_size:
            ffn_dim = neox_args.intermediate_size
        elif neox_args.expansion_factor:
            ffn_dim = int(neox_args.expansion_factor * neox_args.hidden_size)
        else:
            # 4h is default for ffn_dim
            ffn_dim = 4 * neox_args.hidden_size
        ffn_dim_in = ffn_dim
        if self.is_gated:
            # set activation function to be gated implementation
            self.activation_func = Gated_Activation(self.activation_func)
            # auto scale so gated activations has equal parameters
            ffn_dim = int(ffn_dim * 2 / 3)
            ffn_dim_in = ffn_dim // 2
        # set multiple
        ffn_dim = int(
            (2 * self.multiple_of)
            * ((ffn_dim + (2 * multiple_of) - 1) // (2 * multiple_of))
        )
        ffn_dim_in = int(
            self.multiple_of * ((ffn_dim_in + multiple_of - 1) // multiple_of)
        )

        if neox_args.norm in ["layernorm", "te_layernorm"]:
            self.eps = 1.0e-5
            self.normalization = "LayerNorm"
        elif neox_args.norm in ["rmsnorm", "te_rmsnorm"]:
            self.eps = 1.0e-8
            self.normalization = "RMSNorm"
        else:
            raise ValueError(
                "Only LayerNorm and RMSNorm are supported with TransformerEngine"
            )

        if self.activation_type not in [
            "gelu",
            "geglu",
            "relu",
            "reglu",
            "squared_relu",
            "swiglu",
            "qgelu",
            "srelu",
        ]:
            raise ValueError(
                "Only gelu, geglu, relu, reglu, squared_relu, swiglu, qgelu, and srelu are supported with TransformerEngine"
            )

        super(TELayerNormMLP, self).__init__(
            hidden_size=neox_args.hidden_size,
            ffn_hidden_size=ffn_dim,
            eps=self.eps,
            bias=self.bias,
            normalization=self.normalization,
            activation=self.activation_type,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            device=torch.cuda.current_device(),
            set_parallel_mode=self.set_parallel_mode,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            tp_size=self.world_size,
            return_bias=True,
            params_dtype=self.params_dtype,
            seq_length=self.seq_len,
            get_rng_state_tracker=get_cuda_rng_tracker,
            micro_batch_size=self.micro_batch_size,
        )


class TEColumnParallelLinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        MOE=False,
        MoE_mp_size=1,
        mup_rescale_parameters=False,
        seq_dim=0,
    ):
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.world_size = world_size
        self.tp_group = get_tensor_model_parallel_group()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.use_bias = bias

        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_dim = seq_dim

        self.init_method = init_method
        self.stride = stride
        self.mup_rescale_parameters = mup_rescale_parameters
        self.use_mup = neox_args.use_mup
        self.params_dtype = neox_args.params_dtype
        self.parallel_mode = "column"

        super(TEColumnParallelLinear, self).__init__(
            in_features=self.input_size,
            out_features=self.output_size,
            bias=self.use_bias,
            init_method=self.init_method,
            get_rng_state_tracker=get_cuda_rng_tracker,
            device=torch.cuda.current_device(),
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            tp_size=self.world_size,
            parallel_mode=self.parallel_mode,
            return_bias=self.skip_bias_add,
            params_dtype=self.params_dtype,
        )

    # Copied from Mup
    def width_mult(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        return self.weight.infshape.width_mult()

    # Copied from Mup
    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def mup_reinitialize_weights(self, neox_args):
        if neox_args.use_cpu_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                partial(self.init_method, use_mup=True),
                stride=self.stride,
                return_master_weight=self.keep_master_weight_for_test,
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=1,
                stride=self.stride,
            )

    def forward(self, inp, **kwargs):
        if self.use_mup and self.mup_rescale_parameters:
            input_ /= self.width_mult()

        output = super(TEColumnParallelLinear, self).forward(inp, **kwargs)
        if self.skip_bias_add:
            return output
        else:
            return output, None


class TERowParallelLinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        MOE=False,
        MoE_mp_size=1,
        parallel_output=False,
        mup_rescale_parameters=False,
    ):
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        # Divide the weight matrix along the last dimension.
        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.world_size = world_size
        self.tp_group = get_tensor_model_parallel_group()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.use_bias = bias
        self.input_is_parallel = input_is_parallel
        self.sequence_parallel = neox_args.sequence_parallel

        self.init_method = init_method
        self.stride = stride
        self.mup_rescale_parameters = mup_rescale_parameters
        self.use_mup = neox_args.use_mup
        self.params_dtype = neox_args.params_dtype
        self.parallel_mode = "row"

        super(TERowParallelLinear, self).__init__(
            in_features=self.input_size,
            out_features=self.output_size,
            bias=self.use_bias,
            init_method=self.init_method,
            get_rng_state_tracker=get_cuda_rng_tracker,
            device=torch.cuda.current_device(),
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            tp_size=self.world_size,
            parallel_mode=self.parallel_mode,
            return_bias=self.skip_bias_add,
            params_dtype=self.params_dtype,
        )

    # Copied from Mup
    def width_mult(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        return self.weight.infshape.width_mult()

    # Copied from Mup
    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def mup_reinitialize_weights(self, neox_args):
        if neox_args.use_cpu_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                partial(self.init_method, use_mup=True),
                stride=self.stride,
                return_master_weight=self.keep_master_weight_for_test,
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=1,
                stride=self.stride,
            )

    def forward(self, inp, **kwargs):
        if self.use_mup and self.mup_rescale_parameters:
            input_ /= self.width_mult()

        output = super(TERowParallelLinear, self).forward(inp, **kwargs)

        if self.skip_bias_add:
            return output
        else:
            return output, None


class TEMultiheadAttention(te.pytorch.MultiheadAttention):
    """
    Wrapper for the Transformer-Engine's `MultiheadAttention` layer that also
    has "flash attention" enabled.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):

        self.neox_args = neox_args
        self.attention_mask_func = attention_mask_func
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        self.layer_number = layer_number + 1

        world_size = get_model_parallel_world_size()
        self.world_size = world_size
        self.tp_group = get_tensor_model_parallel_group()
        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_len = neox_args.seq_length
        self.micro_batch_size = neox_args.train_micro_batch_size_per_gpu
        self.params_dtype = neox_args.params_dtype
        self.set_parallel_mode = False
        if world_size > 1:
            self.set_parallel_mode = True

        if neox_args.norm in ["layernorm", "te_layernorm"]:
            self.eps = 1.0e-5
            self.normalization = "LayerNorm"
        elif neox_args.norm == ["rmsnorm", "te_rmsnorm"]:
            self.eps = 1.0e-8
            self.normalization = "RMSNorm"

        if (
            not neox_args.num_kv_heads
            or neox_args.num_kv_heads == neox_args.num_attention_heads
        ):
            self.gqa = False
            self.num_kv_heads = None
        else:
            self.gqa = True
            self.num_kv_heads = neox_args.num_kv_heads

        super(TEMultiheadAttention, self).__init__(
            hidden_size=neox_args.hidden_size,
            num_attention_heads=neox_args.num_attention_heads,
            attention_dropout=neox_args.attention_dropout,
            layernorm_epsilon=self.eps,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            layer_number=self.layer_number,
            window_size=neox_args.sliding_window_width,
            num_gqa_groups=self.num_kv_heads,
            input_layernorm=False,
            normalization=self.normalization,
            bias=True,
            device=torch.cuda.current_device(),
            get_rng_state_tracker=get_cuda_rng_tracker,
            set_parallel_mode=self.set_parallel_mode,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            tp_size=self.world_size,
            params_dtype=self.params_dtype,
            return_bias=True,
            qkv_format="sbhd",
            fuse_qkv_params=True,
        )

        if neox_args.pos_emb == "rotary":
            self.hidden_size_per_attention_head = mpu.divide(
                neox_args.hidden_size, neox_args.num_attention_heads
            )

            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_embeddings = RotaryEmbedding(
                dim,
                base=neox_args.rotary_emb_base,
                max_seq_len=neox_args.seq_length,
                precision=neox_args.params_dtype,
                save_inv_freqs=neox_args.rotary_save_freqs_buffer,
            )
            self.rope_emb = self.rotary_embeddings.get_emb()

    def forward(
        self, hidden_states, attention_mask, layer_past=None, rope_emb=None, **kwargs
    ):
        output = super(TEMultiheadAttention, self).forward(
            hidden_states, attention_mask, rotary_pos_emb=self.rope_emb, **kwargs
        )
        return output


class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    ##TODO Test with H100
    def __init__(self, neox_args):

        self.neox_args = neox_args
        self.tp_group = get_tensor_model_parallel_group()

        if neox_args.te_fp8_format == "e4m3":
            fp8_format = te.common.recipe.Format.E4M3
        elif neox_args.te_fp8_format == "hybrid":
            fp8_format = te.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        override_linear_precision = (False, False, not neox_args.te_fp8_wgrad)

        super().__init__(
            margin=neox_args.fp8_margin,
            fp8_format=te_fp8_format,
            amax_compute_algo=neox_args.te_fp8_amax_compute_algo,
            amax_history_len=neox_args.te_fp8_amax_history_len,
            override_linear_precision=override_linear_precision,
            fp8_mha=neox_args.te_fp8_mha,
        )

    def fp8_context(self):
        fp8_group = None
        if self.tp_group:
            fp8_group = self.tp_group
        fp8_context = te.pytorch.fp8_autocast(
            enabled=True, fp8_recipe=self, fp8_group=fp8_group
        )

        return get_context
