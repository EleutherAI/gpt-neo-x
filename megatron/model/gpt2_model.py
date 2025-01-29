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

"""GPT-2 model."""

import math
import torch
import torch.nn as nn
from collections import defaultdict

from functools import partial
from megatron.model.utils import Lambda, SequentialWrapper, recursive_setattr
from megatron.model.norms import get_norm
from megatron.model.init_functions import get_init_methods

from megatron import mpu
from megatron.mpu import ParallelRelativePositionBias
from megatron.model.transformer import (
    ParallelTransformerLayerPipe,
    NormPipe,
    ParallelLinearPipe,
    parallel_lm_logits,
    ParallelLinear,
)
from megatron.model.gmlp import GMLPBlock
from megatron.model.rwkv.v6 import RWKVResidualLayerPipe
from megatron.model.mamba import ParallelMambaResidualLayerPipe
from megatron.model.word_embeddings import EmbeddingPipe, SoftEmbedding

# Pipeline parallelism
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from typing import Union, List


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    mask_value = torch.finfo(attention_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(
        mask_value, dtype=attention_scores.dtype, device=attention_scores.device
    )
    attention_scores.masked_fill_(ltor_mask, mask_value)
    return attention_scores


def cross_entropy(output, labels, _fp16=False):
    """From pretrain_gpt2:forward_step()"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss_mask_sum = loss_mask.sum()
    if mpu.get_context_parallel_world_size() > 1:
        dt = loss_mask_sum.dtype
        if dt == torch.bfloat16 and mpu.initialize.get_fp32_allreduce():
            loss_mask_sum = loss_mask_sum.float()
        torch.distributed.all_reduce(
            loss_mask_sum,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_context_parallel_group(),
        )
        if dt == torch.bfloat16 and mpu.initialize.get_fp32_allreduce():
            loss_mask_sum = loss_mask_sum.bfloat16()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask_sum
        if dt == torch.bfloat16 and mpu.initialize.get_fp32_allreduce():
            loss = loss.float()
        torch.distributed.all_reduce(
            loss,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_context_parallel_group(),
        )
        if dt == torch.bfloat16 and mpu.initialize.get_fp32_allreduce():
            loss = loss.bfloat16()
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask_sum
    return loss


def _pre_transformer_block(args):
    # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
    assert len(args) == 2, "Incorrect number of arguments to _pre_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])
    return fn(args)


def _post_transformer_block(args):
    # from (hidden_states, attention_mask)
    # to (hidden_states.T)
    assert len(args) == 2, "Incorrect number of arguments to _post_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
    return fn(args)


class GPT2ModelPipe(PipelineModule, torch.nn.Module):
    """GPT2Model adapted for pipeline parallelism.

    The largest change is flattening the GPTModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.

    :param neox_args: NeoX arguments object (configuration)
    :param num_tokentypes: number of token types (TODO: deprecated, remove)
    :param parallel_output: if true, don't gather the output logits, and calculate loss in parallel. Set to true by default in training for efficiency, but set to false for inference.
    :param topology: deepspeed topology object specifying pipe / model parallelism topology.
    :param use_cache: if true, cache key/value pairs for each layer in inference.
    """

    def __init__(
        self,
        neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=None,
        use_cache=False,
    ):
        self.neox_args = neox_args

        self.use_cache = use_cache
        self.parallel_output = parallel_output
        self.hidden_size = self.neox_args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method, self.output_layer_init_method = get_init_methods(
            self.neox_args
        )
        self.__topology__ = topology

        self.specs = []
        self.init_specs()  # initializes the layer specs (basically a fancy nn.Sequential)

        super().__init__(
            layers=self.specs,
            loss_fn=partial(cross_entropy, _fp16=self.neox_args.fp16_lm_cross_entropy),
            topology=topology,
            activation_checkpoint_interval=self.neox_args.checkpoint_num_layers
            if self.neox_args.checkpoint_activations
            else 0,
            partition_method=neox_args.pipe_partition_method,
            checkpointable_layers=[
                "GMLPBlock",
                "ParallelTransformerLayerPipe",
                "ParallelMambaResidualLayerPipe",
            ],
        )

    def insert_layers(
        self, layers: Union[nn.Module, nn.ModuleList, nn.Sequential, List], idx
    ):
        """
        inserts the layers in `layers` into the pipe model at `idx`.
        """
        if isinstance(layers, nn.Module):
            self.specs.insert(idx, layers)
        elif any(
            [isinstance(layers, nn.ModuleList), isinstance(layers, nn.Sequential)]
        ):
            self.specs[idx:idx] = layers
        elif isinstance(layers, list):
            assert all(
                [hasattr(l, "__call__") for l in layers]
            ), "all items in `layers` must be Callables"
            self.specs[idx:idx] = layers
        else:
            raise ValueError(
                f"layer passed into {self.__class__.__name__}.insert_layer() should be either an nn.Module, an nn.ModuleList, an nn.Sequential object, or a list of callables not a {type(layers)}"
            )

        # re-initialize parent class
        super().__init__(
            layers=self.specs,
            loss_fn=self.loss_fn,
            topology=self.__topology__,
            activation_checkpoint_interval=self.activation_checkpoint_interval,
            partition_method=self.neox_args.pipe_partition_method,
            checkpointable_layers=[
                "GMLPBlock",
                "ParallelTransformerLayerPipe",
                "ParallelMambaResidualLayerPipe",
                "RWKVResidualLayerPipe",
            ],
        )

    def init_specs(self):

        weight_tying = not self.neox_args.no_weight_tying
        self.specs = []

        # Embedding layer
        # input will be (input_ids, position_ids, attention_mask)

        if weight_tying:
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                )
            )

        # NB: the attention mask always needs to be the *last* item in the args when being passed from
        # one stage to the next, because deepspeed is hacks on top of hacks.
        #
        # outputs are now (hidden_states,  attention_mask)

        self.specs.append(_pre_transformer_block)

        # T5 RPE positional embedding
        if self.neox_args.pos_emb == "rpe":
            hidden_size_per_attention_head = mpu.divide(
                self.neox_args.hidden_size, self.neox_args.num_attention_heads
            )
            rpe_scale = math.sqrt(hidden_size_per_attention_head)
            rpe_emb = ParallelRelativePositionBias(
                neox_args=self.neox_args,
                scale=rpe_scale,
                causal=True,
                num_buckets=self.neox_args.rpe_num_buckets,
                max_distance=self.neox_args.rpe_max_distance,
                heads=self.neox_args.num_attention_heads,
            )

        # Transformer layers
        for i in range(self.neox_args.num_layers):
            layer_type = self.neox_args.attention_config[i]
            if layer_type in ["gmlp", "amlp"]:
                self.specs.append(
                    LayerSpec(
                        GMLPBlock,
                        init_method=self.init_method,
                        layer_number=i,
                        output_layer_init_method=self.output_layer_init_method,
                        neox_args=self.neox_args,
                        mask_fn=gpt2_attention_mask_func,
                    )
                )
            elif layer_type == "rwkv":
                self.specs.append(
                    LayerSpec(
                        RWKVResidualLayerPipe,
                        neox_args=self.neox_args,
                        layer_number=i,
                    )
                )
            elif layer_type in ["mamba"]:
                self.specs.append(
                    LayerSpec(
                        ParallelMambaResidualLayerPipe,
                        neox_args=self.neox_args,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method,
                        layer_number=i,
                    )
                )
            else:
                self.specs.append(
                    LayerSpec(
                        ParallelTransformerLayerPipe,
                        neox_args=self.neox_args,
                        attention_mask_func=gpt2_attention_mask_func,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method,
                        layer_number=i,
                        rpe=rpe_emb if self.neox_args.pos_emb == "rpe" else None,
                        rotary=self.neox_args.pos_emb == "rotary",
                        use_cache=self.use_cache,
                    )
                )

        # used to drop attention mask + reshape hidden states
        self.specs.append(_post_transformer_block)

        # NormPipe is a (deprecated) helper class that used to be used to pass presents along the pipeline - since presents are now cached to the `TransformerLayer` class this is no longer needed
        norm, eps = get_norm(self.neox_args)
        self.specs.append(
            LayerSpec(NormPipe, norm, self.neox_args.hidden_size, eps=eps)
        )

        # outputs are now a single tensor: hidden_states

        def _logits_helper(embedding, lm_output):
            """Just a wrapper to massage inputs/outputs from pipeline."""
            if self.neox_args.use_mup:
                # Since we're using pipeline parallelism, we can't directly use MuReadout. Instead, use this workaround that does the same thing as MuReadout.
                # https://github.com/microsoft/mup/issues/6#issuecomment-1082156274
                lm_output = (
                    lm_output
                    / self.tied_modules.embed.word_embeddings.weight.infshape.width_mult()
                )

            logits = parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output,
                seq_parallel=self.neox_args.sequence_parallel,
            )
            return logits

        if weight_tying:
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    forward_fn=_logits_helper,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    ParallelLinearPipe,
                    neox_args=self.neox_args,
                    init_method=self.init_method,
                    parallel_output=self.parallel_output,
                    is_last_layer=True,
                )
            )

    def _set_parallel_output(self, value):
        # sets the parallel output value of the final layer to value
        final_layer = list(self.forward_funcs)[-1]
        if isinstance(final_layer, (ParallelLinearPipe, ParallelLinear)):
            final_layer.final_linear.set_parallel_output(value)

    def inference_mode(self, use_cache=True):
        """
        Sets up the model for inference by turning on k/v caching (if specified) and setting `parallel output` of the final layer to false,
        so logits are gathered across model parallel ranks.

        :param cache: (bool) True if you want to use caching during inference, False otherwise
        """
        # first set caching to true if specified
        recursive_setattr(self.forward_funcs, "use_cache", use_cache, assert_type=bool)
        # then set parallel output of the final layer to false so we don't have to gather the output manually
        self._set_parallel_output(False)
        recursive_setattr(self.forward_funcs, "training", False)

    def train_mode(self):
        """
        Sets up the model for training by turning off k/v caching and setting `parallel output` of the final layer to True,
        so logits are not gathered across model parallel ranks, and loss is computed in parallel (more efficient).
        """
        # set caching to false
        recursive_setattr(self.forward_funcs, "use_cache", False)
        # then set parallel output to true (more efficient training)
        self._set_parallel_output(True)
        recursive_setattr(self.forward_funcs, "training", True)

    def clear_cache(self):
        """
        Recursively clears the kv cache on all layers
        """
        recursive_setattr(self.forward_funcs, "layer_past", None)

    def to_sequential(self):
        """
        Transforms the PipelineModule to a plain nn.Sequential module
        :return:
        """
        layers = []
        tied_layers = defaultdict(list)
        for n, spec in enumerate(self.specs):
            if isinstance(spec, TiedLayerSpec):
                if spec.key in tied_layers:
                    # receiver
                    layers.append(
                        Lambda(lambda x: spec.forward_fn(tied_layers[spec.key][0], x))
                    )
                else:
                    # owner
                    module = spec.build(log=False)
                    layers.append(module)
                    tied_layers[spec.key].append(module)
            elif isinstance(spec, LayerSpec):
                layers.append(spec.build(log=False))
            elif hasattr(spec, "__call__"):
                # check that it's a callable function
                layers.append(Lambda(spec))
            else:
                raise ValueError(f"Layer number {n} ({spec}) Not recognized")
        model = SequentialWrapper(
            layers,
            self.activation_checkpoint_interval,
            self.activation_checkpoint_func,
            parent_class_name=self.__class__.__name__,
        )
        return model
