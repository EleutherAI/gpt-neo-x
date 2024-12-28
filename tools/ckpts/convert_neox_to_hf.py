# Copyright (c) 2023, EleutherAI
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

import os
import sys

import yaml
import argparse
from tqdm import tqdm

import torch
from transformers import (
    MistralConfig,
    LlamaConfig,
    GPTNeoXConfig,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSequenceClassification,
)

from typing import List, Literal

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
from megatron.tokenizer import build_tokenizer


"""
A script for converting saved NeoX Checkpoints to Huggingface (HF) compatible GPT-NeoX type models.

Note that this script does not support all NeoX features.
Please investigate carefully whether your model is compatible with all architectures supported by the GPTNeoXForCausalLM class in HF.

(e.g. position embeddings such as AliBi may not be supported by Huggingface's GPT-NeoX architecture).
"""


# Model definitions: a list of keys, and where they fall in terms of handling them in the presence of TP.
# in format : {model arch: {param type: {param in neox: param in HF}}}
MODEL_KEYS = {
    "neox": {
        "new": {
            "COLUMN_PARALLEL_LINEAR_KEYS": {
                "mlp.linear1.weight": "mlp.dense_h_to_4h.weight",
                "mlp.linear1.bias": "mlp.dense_h_to_4h.bias",
                "attention.query_key_value.weight": "attention.query_key_value.weight",
                "attention.query_key_value.bias": "attention.query_key_value.bias",  # TODO: handle GQA separately?
            },
            "ROW_PARALLEL_LINEAR_KEYS": {
                "attention.dense.weight": "attention.dense.weight",
                "mlp.linear2.weight": "mlp.dense_4h_to_h.weight",
            },
            "ROW_PARALLEL_BIAS_KEYS": {
                "mlp.linear2.bias": "mlp.dense_4h_to_h.bias",
                "attention.dense.bias": "attention.dense.bias",
            },
            "NORM_KEYS": {
                "input_layernorm.weight": "input_layernorm.weight",
                "input_layernorm.bias": "input_layernorm.bias",
                "post_attention_layernorm.weight": "post_attention_layernorm.weight",
                "post_attention_layernorm.bias": "post_attention_layernorm.bias",
            },
            "FINAL_NORM_KEYS": {
                "norm.weight": "weight",
                "norm.bias": "bias",
            },
        },
        "legacy": {
            "COLUMN_PARALLEL_LINEAR_KEYS": {
                "mlp.dense_h_to_4h.weight": "mlp.dense_h_to_4h.weight",
                "mlp.dense_h_to_4h.bias": "mlp.dense_h_to_4h.bias",
                "attention.query_key_value.weight": "attention.query_key_value.weight",
                "attention.query_key_value.bias": "attention.query_key_value.bias",  # TODO: handle GQA separately?
            },
            "ROW_PARALLEL_LINEAR_KEYS": {
                "attention.dense.weight": "attention.dense.weight",
                "mlp.dense_4h_to_h.weight": "mlp.dense_4h_to_h.weight",
            },
            "ROW_PARALLEL_BIAS_KEYS": {
                "mlp.dense_4h_to_h.bias": "mlp.dense_4h_to_h.bias",
                "attention.dense.bias": "attention.dense.bias",
            },
            "NORM_KEYS": {
                "input_layernorm.weight": "input_layernorm.weight",
                "input_layernorm.bias": "input_layernorm.bias",
                "post_attention_layernorm.weight": "post_attention_layernorm.weight",
                "post_attention_layernorm.bias": "post_attention_layernorm.bias",
            },
            "FINAL_NORM_KEYS": {
                "norm.weight": "weight",
                "norm.bias": "bias",
            },
        },
    },
    "llama": {
        "new": {
            "COLUMN_PARALLEL_LINEAR_KEYS": {
                "mlp.linear1.weight": ["mlp.up_proj.weight", "mlp.gate_proj.weight"]
            },
            "ROW_PARALLEL_LINEAR_KEYS": {
                "attention.dense.weight": "self_attn.o_proj.weight",
                "mlp.linear2.weight": "mlp.down_proj.weight",
            },
            "ROW_PARALLEL_BIAS_KEYS": {},  # No biases in RowParallelLinear layers
            "NORM_KEYS": {
                "input_layernorm.scale": "input_layernorm.weight",
                "post_attention_layernorm.scale": "post_attention_layernorm.weight",
            },
            "FINAL_NORM_KEYS": {
                "norm.scale": "weight",
            },
            "GQA_QKV_KEYS": {  # because Llama can have Grouped Query Attention and has separate Q, K, and V linear proj params, handle them separately.
                "attention.query_key_value.weight": [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
            },
        },
        "legacy": {
            "COLUMN_PARALLEL_LINEAR_KEYS": {
                "mlp.w1.weight": "mlp.gate_proj.weight",
                "mlp.w3.weight": "mlp.up_proj.weight",
            },
            "ROW_PARALLEL_LINEAR_KEYS": {
                "attention.dense.weight": "self_attn.o_proj.weight",
                "mlp.w2.weight": "mlp.down_proj.weight",
            },
            "ROW_PARALLEL_BIAS_KEYS": {},  # No biases in RowParallelLinear layers
            "NORM_KEYS": {
                "input_layernorm.scale": "input_layernorm.weight",
                "post_attention_layernorm.scale": "post_attention_layernorm.weight",
            },
            "FINAL_NORM_KEYS": {
                "norm.scale": "weight",
            },
            "GQA_QKV_KEYS": {  # because Llama can have Grouped Query Attention and has separate Q, K, and V linear proj params, handle them separately.
                "attention.query_key_value.weight": [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
            },
        },
    },
}

MODEL_KEYS["mistral"] = MODEL_KEYS["llama"]


def load_partitions(
    input_checkpoint_path: str, mp_partitions: int, layer_idx: int, sequential: bool
) -> List[torch.Tensor]:
    """Returns a list containing all states from a model (across MP partitions)"""

    if sequential:
        filename_format = f"mp_rank_{{i:02}}_model_states.pt"
    else:
        filename_format = f"layer_{layer_idx:02}-model_{{i:02}}-model_states.pt"

    loaded_tp_ranks = [
        torch.load(
            os.path.join(
                input_checkpoint_path,
                filename_format.format(i=i),
            ),
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        for i in range(mp_partitions)
    ]

    return loaded_tp_ranks


def get_state(
    state_dicts: List[torch.Tensor], key: str, layer_idx: int, sequential: bool
) -> torch.Tensor:
    """Helper that returns a list containing a given weight's state from each MP partition, for a given layer in the model."""

    if sequential:
        # use the correct key into the sequential dict for given weight/provided key
        key = f"sequential.{layer_idx}.{key}"

        return [state_dict["module"][key] for state_dict in state_dicts]
    else:
        # For the PipelineModule case, we don't need any key / module prefix. just grab this weight value.
        # layer_idx is also ignored because we've loaded only this layer's weights, ahead of time.
        key = key

        return [state_dict[key] for state_dict in state_dicts]


def get_key(loaded_config, key, default=None):
    """
    Search for a given key in a NeoX yaml. normalizes underscores -> hyphens
    """
    key = key.replace("_", "-")
    try:
        return loaded_config[key]
    except KeyError:
        key = key.replace("-", "_")
        try:
            return loaded_config[key]
        except KeyError:
            return default


def create_config(neox_config, architecture="neox", is_rm=False, pad_token_id=-1):
    """take in a loaded yaml from NeoX and assign relevant values to HF config.
    Returns: GPTNeoXConfig() object
    """

    def gated_size(hidden_dim):
        # takes in a hidden dim and calculates intermediate dim of a LLaMAParallelMLP.
        # (only used if intermediate_size not specified in config)
        # hidden-size * 8 / 3 , rounded up to nearest multiple of 256
        ff_dim = int(2 * hidden_dim * 4 / 3)
        ff_dim = 256 * ((ff_dim + 256 - 1) // 256)
        return ff_dim

    class TokenizerArgs:
        # kinda hacky.
        # this is to get something with the same interface as is used in build_tokenizer()
        # without diving into loading a neox_args object or using argparse etc.
        def __init__(self, neox_config):
            self.make_vocab_size_divisible_by = get_key(
                neox_config, "make-vocab-size-divisible-by", default=128
            )
            self.model_parallel_size = get_key(neox_config, "model-parallel-size")
            self.vocab_file = get_key(neox_config, "vocab-file")
            self.merge_file = get_key(neox_config, "merge-file")
            self.tokenizer_type = get_key(neox_config, "tokenizer-type")

            self.rank = 0

    args = TokenizerArgs(neox_config)
    tokenizer = build_tokenizer(args)
    try:  # GPT2TokenizerFast raises NotImplementedError
        pad_token = tokenizer.pad
    except:
        pad_token = (
            1  # pad defaulting to 1. follows convention from GPT-NeoX-20b tokenizer
        )

    # TODO: change the default value here based on discussion regarding `gpt_j_tied` config parameter's default
    use_tied_lns = get_key(neox_config, "gpt-j-tied", False)

    if use_tied_lns:
        raise NotImplementedError(
            """ERROR: Huggingface Transformers does not yet support a single shared layernorm
                per transformer block for GPT-NeoX models trained  w/ GPT-J parallel residuals.
                See https://github.com/EleutherAI/gpt-neox/pull/481 for further details."""
        )

    # set all config values.

    # shared config parameters.
    args = {
        "vocab_size": args.padded_vocab_size,
        "hidden_size": get_key(neox_config, "hidden-size"),
        "num_hidden_layers": get_key(neox_config, "num-layers"),
        "num_attention_heads": get_key(neox_config, "num-attention-heads"),
        "max_position_embeddings": get_key(neox_config, "max-position-embeddings"),
        "initializer_range": get_key(neox_config, "init-method-std", 0.02),
        "tie_word_embeddings": (not get_key(neox_config, "no-weight-tying", False)),
        "use_cache": True,
    }
    if architecture == "mistral" or architecture == "llama":
        args.update(
            {
                "intermediate_size": get_key(
                    neox_config,
                    "intermediate-size",
                    gated_size(get_key(neox_config, "hidden-size")),
                ),
                "num_key_value_heads": get_key(
                    neox_config,
                    "num-kv-heads",
                    get_key(neox_config, "num-attention-heads"),
                ),
                "hidden_act": get_key(
                    neox_config, "activation", default="silu"
                ).replace("swiglu", "silu"),
                "rms_norm_eps": get_key(neox_config, "rms-norm-epsilon", 1.0e-6),
                "bos_token_id": tokenizer.eod,
                "eos_token_id": tokenizer.eod,
                "rope_theta": get_key(neox_config, "rotary-emb-base", 10000.0),
            }
        )

        if architecture == "mistral":
            # mistral-specific options
            args.update(
                {
                    "sliding_window": get_key(
                        neox_config, "sliding-window-width", 4096
                    ),
                }
            )
            hf_config = MistralConfig(**args)
        elif architecture == "llama":
            # llama-specific options
            args.update(
                {
                    # NeoX library defaults to using bias in attention
                    "attention_bias": get_key(
                        neox_config, "use_bias_in_attn_linear", True
                    ),
                }
            )
            hf_config = LlamaConfig(**args)
    else:
        # GPT-NeoX HF model class-specific options
        args.update(
            {
                "rotary_pct": get_key(neox_config, "rotary-pct", default=1.0),
                "rotary_emb_base": get_key(
                    neox_config, "rotary-emb-base", default=1000.0
                ),
                "use_parallel_residual": get_key(neox_config, "gpt-j-residual", False),
                "layer_norm_eps": get_key(neox_config, "layernorm-epsilon", 1e-5),
                "intermediate_size": get_key(
                    neox_config,
                    "intermediate-size",
                    4 * get_key(neox_config, "hidden-size"),
                ),
            }
        )
        hf_config = GPTNeoXConfig(**args)
    if is_rm:
        hf_config.num_labels = 1
        hf_config.pad_token_id = pad_token_id

    return hf_config


def reshard_and_split_qkv(
    param_mapping: dict,  # a dictionary mapping the QKV weight keys in GPT-NeoX -> a list of keys representing the Q, K, and V weight keys the HF model will use
    hf_config: AutoConfig,  # a HF model config for the model
    loaded_tp_ranks: List[torch.Tensor],
    layer_idx: int,
    sequential: bool,
):
    """
    A helper function which performs reshaping and sharding to make the QKV projection from NeoX compatible with HF Llama models,
    even when grouped-query attention is required.
    """
    for key, hf_keys in param_mapping.items():
        assert (
            isinstance(hf_keys, list) and len(hf_keys) == 3
        ), "Must map QKV to precisely 3 resulting weight matrices."

    for key, hf_keys in param_mapping.items():
        # we first merge the QKV proj. across TP ranks
        sharded_qkv = torch.stack(
            get_state(loaded_tp_ranks, key, layer_idx, sequential), dim=0
        )
        # should now have shape [TP_SIZE, (hidden_size + 2 * kv_hidden_size) / TP_SIZE, hidden_size].

        sharded_qkv = sharded_qkv.view(
            len(loaded_tp_ranks),
            hf_config.num_attention_heads // len(loaded_tp_ranks),
            int(
                hf_config.hidden_size
                // hf_config.num_attention_heads
                * (
                    1
                    + 2 * hf_config.num_key_value_heads / hf_config.num_attention_heads
                )
            ),
            hf_config.hidden_size,
        )  # is meant to convert to shape [TP_SIZE, NUM_QUERY_HEADS_PER_SHARD, dims_per_head * (1 + 2 * kv-to-q head ratio), hidden_size]

        q, k, v = torch.split(
            sharded_qkv,
            [
                hf_config.hidden_size // hf_config.num_attention_heads,
                int(
                    (hf_config.num_key_value_heads / hf_config.num_attention_heads)
                    * hf_config.hidden_size
                    // hf_config.num_attention_heads
                ),
                int(
                    (hf_config.num_key_value_heads / hf_config.num_attention_heads)
                    * hf_config.hidden_size
                    // hf_config.num_attention_heads
                ),
            ],
            dim=2,
        )
        # splits along the (dims_per_head * (1 + 2 * kv-to-q head ratio)_ dim to get 3 tensors:
        # 1 x [TP_SIZE, NUM_Q_HEADS_PER_SHARD, dims_per_head, hidden_size] and 2 x [TP_SIZE, NUM_Q_HEADS_PER_SHARD, (dims_per_head / kv-to-q head ratio), hidden_size]
        # these are the Q, and K, V tensors respectively.

        # we have to do additional reshape for each individual tensor now,
        # into the expected square (or smaller than square, for K/V tensors) shape
        q, k, v = q.squeeze(dim=2), k.squeeze(dim=2), v.squeeze(dim=2)
        q = q.view(
            hf_config.num_attention_heads,
            hf_config.hidden_size // hf_config.num_attention_heads,
            hf_config.hidden_size,
        ).reshape(hf_config.hidden_size, hf_config.hidden_size)
        k = k.reshape(
            hf_config.num_key_value_heads,
            hf_config.hidden_size // hf_config.num_attention_heads,
            hf_config.hidden_size,
        ).reshape(
            hf_config.hidden_size
            // hf_config.num_attention_heads
            * hf_config.num_key_value_heads,
            hf_config.hidden_size,
        )
        v = v.reshape(
            hf_config.num_key_value_heads,
            hf_config.hidden_size // hf_config.num_attention_heads,
            hf_config.hidden_size,
        ).reshape(
            hf_config.hidden_size
            // hf_config.num_attention_heads
            * hf_config.num_key_value_heads,
            hf_config.hidden_size,
        )

        # return these
        state_dict = {}
        for hf_key, proj in zip(hf_keys, [q, k, v]):
            state_dict[hf_key] = proj.clone()
        return state_dict


def get_mlp_naming_convention(loaded_tp_ranks, layer_idx, sequential):
    """Determine whether the checkpoint uses the legacy or new MLP naming convention."""
    for state_dict in loaded_tp_ranks:
        print("------------------------------")
        print(state_dict.keys())
    if any(
        [
            ["mlp.linear1.weight" in key for key in list(state_dict.keys())]
            for state_dict in loaded_tp_ranks
        ]
    ):
        return "new"
    elif any(
        [
            [
                "mlp.dense_h_to_4h.weight" in key
                for key in list(state_dict.keys())
            ]
            for state_dict in loaded_tp_ranks
        ]
    ):
        return "legacy"
    else:
        raise ValueError("Unable to determine MLP naming convention in checkpoint")


def convert(
    input_checkpoint_path,
    loaded_config,
    output_checkpoint_path,
    sequential: bool = True,
    precision: Literal["auto", "fp16", "bf16", "fp32"] = "auto",
    architecture: Literal["neox", "llama", "mistral"] = "neox",
    is_rm: bool = False,
    pad_token_id: int = -1,
):
    """convert a NeoX checkpoint to a HF model format.
    should perform model-parallel merging correctly
    but only supports features allowed by HF GPT-NeoX implementation (e.g. rotary embeddings)
    """

    ARCH = MODEL_KEYS[architecture]

    hf_config = create_config(
        loaded_config, architecture=architecture, is_rm=is_rm, pad_token_id=pad_token_id
    )

    if not is_rm:
        hf_model = AutoModelForCausalLM.from_config(hf_config)
    else:
        hf_model = AutoModelForSequenceClassification.from_config(hf_config)

    if architecture == "neox":
        hf_transformer = hf_model.gpt_neox
    else:
        hf_transformer = hf_model.model

    if precision == "auto":
        print("Auto-detecting precision to save model into...")
        # save model in FP16 if Deepspeed fp16 was used in config, else 32 bit
        fp16 = get_key(loaded_config, "fp16")

        if fp16:
            try:
                # current behavior is to pass "fp16": {"enabled": true}, when using upstream Deepspeed
                if fp16["enabled"]:
                    hf_model.half()
                    print("Saving weights in fp16 precision...")
            except:
                try:
                    # attempt to access bf16 dict in yaml file, if fp16 not enabled
                    bf16 = get_key(loaded_config, "bf16")
                    if bf16:
                        hf_model.to(dtype=torch.bfloat16)
                        print("Saving weights in bf16 precision...")
                except:
                    hf_model.to(dtype=torch.float)
                    print(
                        "Model not trained in fp16 / bf16 mixed precision, saving weights in fp32..."
                    )
    else:
        name_to_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float,
        }
        print(f"Saving model into specified {precision} precision...")
        hf_model.to(dtype=name_to_dtype[precision])

    mp_partitions = get_key(loaded_config, "model-parallel-size")

    # Sequential saves all model states from an MP rank in one file.
    # so we only load the MP ranks only once and index into them with get_state().
    # for the pipeline-parallel case (pipeline-parallel-size >= 1),
    # we must load the correct layer's states at each step.
    # (this does mean that less memory is required for PP conversion.)
    loaded_tp_ranks = load_partitions(
        input_checkpoint_path, mp_partitions, layer_idx=0, sequential=sequential
    )

    ### Embedding layer ###
    # Embedding is layer idx 0
    if architecture == "neox":
        embed_in = hf_transformer.embed_in
    else:
        embed_in = hf_transformer.embed_tokens
    embed_in.load_state_dict(  # TODO: embed_in is not always model's name for embedding
        {
            "weight": torch.cat(
                get_state(
                    loaded_tp_ranks,
                    "word_embeddings.weight",
                    layer_idx=0,
                    sequential=sequential,
                ),
                dim=0,
            )
        }
    )
    assert (
        hf_config.vocab_size == embed_in.weight.shape[0]
    ), f"ERROR: calculated vocab size {hf_config.vocab_size} != embed param size {embed_in.shape[0]}"
    ### End Embedding Layer ###

    # grab from 3rd layer to pass embeddings
    mlp_naming = get_mlp_naming_convention(
        load_partitions(
            input_checkpoint_path,
            mp_partitions,
            layer_idx=3,
            sequential=sequential,
        ),
        0,
        sequential,
    )
    print(f"Detected MLP naming convention: {mlp_naming}")
    ARCH = ARCH[mlp_naming]

    for layer_i in tqdm(range(get_key(loaded_config, "num-layers"))):

        # get layer from hf model
        hf_layer = hf_transformer.layers[layer_i]  # TODO: model module names

        if not sequential:
            # in the non-sequential case, must load from each layer individually.
            # use layer index + 2 bc of embed layer and a dummy _pre_transformer_block, which are "layers 0 and 1"
            loaded_tp_ranks = load_partitions(
                input_checkpoint_path,
                mp_partitions,
                layer_idx=layer_i + 2,
                sequential=sequential,
            )

        # + 2 bc of embed layer and a dummy _pre_transformer_block
        state_dict = {}
        for key, hf_key in ARCH["ROW_PARALLEL_LINEAR_KEYS"].items():
            state_dict[hf_key] = torch.cat(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                ),
                dim=1,
            )

        # average layernorm stats over mp ranks
        for key, hf_key in ARCH["NORM_KEYS"].items():
            state_dict[hf_key] = sum(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                )
            ) / len(loaded_tp_ranks)

        # LinearWithTPMerge
        for key, hf_key in ARCH["COLUMN_PARALLEL_LINEAR_KEYS"].items():
            if type(hf_key) == list:
                # Llama magic - split the weight into two parts for the gate and up proj
                states = [
                    torch.chunk(state, chunks=2, dim=0)
                    for state in get_state(
                        loaded_tp_ranks,
                        key,
                        layer_idx=layer_i + 2,
                        sequential=sequential,
                    )
                ]
                # Set up proj...
                state_dict[hf_key[0]] = torch.cat([state[0] for state in states], dim=0)
                # Set gate proj...
                state_dict[hf_key[1]] = torch.cat([state[1] for state in states], dim=0)
            else:
                state_dict[hf_key] = torch.cat(
                    get_state(
                        loaded_tp_ranks,
                        key,
                        layer_idx=layer_i + 2,
                        sequential=sequential,
                    ),
                    dim=0,
                )

        # LinearWithTPSplitBias
        for key, hf_key in ARCH["ROW_PARALLEL_BIAS_KEYS"].items():
            state_dict[hf_key] = sum(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                )
            )

        # Just take one
        if "attention.bias" in hf_layer.state_dict():
            state_dict["attention.bias"] = hf_layer.state_dict()["attention.bias"]
        if "attention.masked_bias" in hf_layer.state_dict():
            state_dict["attention.masked_bias"] = hf_layer.state_dict()[
                "attention.masked_bias"
            ]

        # some architectures, like Mistral and Llama, have the following which must be handled specially:
        # - Q, K, V projections are performed separately, so we must split apart GPT-NeoX library's single QKV proj
        # - Support for Grouped-Query Attention, meaning the Q and the K, V projections may not be the same size
        if "GQA_QKV_KEYS" in ARCH:
            state_dict.update(
                reshard_and_split_qkv(
                    param_mapping=ARCH["GQA_QKV_KEYS"],
                    hf_config=hf_config,
                    loaded_tp_ranks=loaded_tp_ranks,
                    layer_idx=layer_i + 2,
                    sequential=sequential,
                )
            )
        # load state_dict into layer
        hf_layer.load_state_dict(state_dict)

    if not sequential:
        loaded_tp_ranks = load_partitions(
            input_checkpoint_path,
            mp_partitions,
            get_key(loaded_config, "num-layers") + 3,
            sequential=sequential,
        )
    # Load final layer norm
    norm_state_dict = {}
    for key, hf_key in ARCH["FINAL_NORM_KEYS"].items():
        norm_state_dict[hf_key] = sum(
            get_state(
                loaded_tp_ranks,
                key,
                layer_idx=get_key(loaded_config, "num-layers") + 3,
                sequential=sequential,
            )
        ) / len(loaded_tp_ranks)

    if architecture == "neox":
        final_layer_norm = hf_transformer.final_layer_norm
    else:
        final_layer_norm = hf_transformer.norm

    final_layer_norm.load_state_dict(norm_state_dict)

    # Load output embedding
    if not sequential:
        if get_key(loaded_config, "no-weight-tying", False):
            # if we have trained input + output embedding layers without tied weights
            loaded_tp_ranks = load_partitions(
                input_checkpoint_path,
                mp_partitions,
                get_key(loaded_config, "num-layers") + 4,
                sequential=sequential,
            )
        else:
            # in this case, output embedding layer and input embedding layer are tied.
            # load + save the input embed weights into the output embedding layer's place.
            loaded_tp_ranks = load_partitions(
                input_checkpoint_path,
                mp_partitions,
                layer_idx=0,
                sequential=sequential,
            )
    # output embedding / LM head
    if not is_rm:
        if architecture == "neox":  # name of lm head / final linear proj varies
            lm_head = hf_model.embed_out
        else:
            lm_head = hf_model.lm_head
    else:
        lm_head = hf_model.score

    if get_key(loaded_config, "no-weight-tying", False):
        # save the (untied) final linear into LM head for HF
        lm_head.load_state_dict(
            {
                "weight": torch.cat(
                    get_state(
                        loaded_tp_ranks,
                        "final_linear.weight" if not is_rm else "rm_linear.weight",
                        layer_idx=get_key(loaded_config, "num-layers") + 4,
                        sequential=sequential,
                    ),
                    dim=0 if not is_rm else 1,
                ),
            }
        )
    else:
        # don't need to worry about rm here since you can't really tie them...

        # embedding layers are tied. transpose input layer and save
        lm_head.load_state_dict(
            {
                "weight": torch.cat(
                    get_state(
                        loaded_tp_ranks,
                        "word_embeddings.weight",
                        layer_idx=0,
                        sequential=sequential,
                    ),
                    dim=0,
                ),
            }
        )

    del loaded_tp_ranks

    return hf_model


def main(input_args=None, overwrite_values=None):
    from huggingface_hub import create_repo, HfApi

    parser = argparse.ArgumentParser(
        description="Merge MP partitions and convert to HF Model."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to NeoX checkpoint, e.g. /path/to/model/global_step143000",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to config file for the input NeoX checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output dir, where to save the HF Model, tokenizer, and configs",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        help="What precision to save the model into. Defaults to auto, which auto-detects which 16-bit dtype to save into, or falls back to fp32.",
    )
    parser.add_argument(
        "--no_save_tokenizer",
        action="store_true",
        help="Whether to skip saving the tokenizer alongside a model.",
    )
    parser.add_argument(
        "--vocab-is-hf-tokenizer",
        action="store_true",
        help="Whether the vocab file is in a Huggingface tokenizer path.",
    )
    parser.add_argument(
        "--pad-token-id",
        type=int,
        default=-1,
        help="Pad token id to set in tokenizer. Required for RM style models.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="neox",
        help="What HF model class type to export into.",
    )
    args = parser.parse_args(input_args)

    # validate arguments
    assert args.precision in [
        "auto",
        "fp16",
        "bf16",
        "fp32",
    ], f"expected --precision to be one of 'auto', 'fp16', 'bf16', 'fp32' but got '{args.precision}' !"
    assert args.architecture in [
        "neox",
        "llama",
        "mistral",
    ], f"expected --architecture to be one of 'neox', 'mistral', 'llama', but got '{args.architecture}' !"

    with open(args.config_file) as f:
        loaded_config = yaml.full_load(f)
        if overwrite_values:
            loaded_config.update(overwrite_values)

    # Determine the checkpoint format of the model.
    # DeepSpeed saves models wrapped in a PipelineModule differently from those not.
    # PipelineModule models are saved as per-layer state dicts per TP shard,
    # while Sequential model state dicts are saved all together in one mp_rank_xx_model_states.pt
    # file per tensor/model parallel shard.
    pipeline_world_size = get_key(loaded_config, "pipe-parallel-size", 1)
    is_rm = get_key(loaded_config, "train_impl", "normal") == "rm"
    if is_rm and args.pad_token_id == -1:
        raise ValueError("RM models require a pad token id to be set.")
    if pipeline_world_size == 0:
        sequential = True
        print(
            f"Detected 'pipe-parallel-size' of {pipeline_world_size}, assuming model is saved as Sequential..."
        )
    else:
        sequential = False
        print(
            f"Detected 'pipe-parallel-size' of {pipeline_world_size}, assuming model is saved as PipelineModule..."
        )

    # convert the model to HF.
    hf_model = convert(
        args.input_dir,
        loaded_config,
        args.output_dir,
        sequential=sequential,
        architecture=args.architecture,
        is_rm=is_rm,
        pad_token_id=args.pad_token_id,
    )

    # Save to disk.
    hf_model.save_pretrained(args.output_dir)

    if not args.no_save_tokenizer:
        # save tokenizer to directory as well, for easy loading of model as a HF model.
        tokenizer_type = get_key(loaded_config, "tokenizer-type")
        if args.vocab_is_hf_tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                os.path.dirname(get_key(loaded_config, "vocab-file"))
            )
            if args.pad_token_id != -1:
                tokenizer.pad_token_id = args.pad_token_id
            print("loaded tokenizer: ", tokenizer)
            tokenizer.save_pretrained(args.output_dir)
            print("tokenizer saved!")
        elif tokenizer_type == "HFTokenizer":  # TODO: handle sentencepiece tokenizers?
            print(f"saving tokenizer from file {get_key(loaded_config, 'vocab-file')}")
            print(
                "Warning: please check that your model config and tokenizer end with the correct special tokens (EOS, BOS)."
            )
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=get_key(loaded_config, "vocab-file")
            )
            if args.pad_token_id != -1:
                tokenizer.pad_token_id = args.pad_token_id
            print("loaded tokenizer: ", tokenizer)
            tokenizer.save_pretrained(args.output_dir)
            print("tokenizer saved!")


if __name__ == "__main__":

    # before running script:
    # `pip install --upgrade transformers`
    # `huggingface-cli login`
    #
    main()
