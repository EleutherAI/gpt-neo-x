import sys
import os
import copy
import deepspeed

# import time

import argparse
import torch
import base64
import json

import numpy as np

from functools import reduce
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
from megatron.neox_arguments import NeoXArgs
from megatron.training import get_model, get_optimizer, get_learning_rate_scheduler
from megatron.initialize import initialize_megatron
from megatron import mpu
from megatron.checkpointing import load_checkpoint, save_checkpoint

# from megatron.utils import (
#     Timers,
#     init_wandb,
# )

"""
A script for converting publicly available Huggingface (HF) checkpoints NeoX format.

Note that this script requires access to corresponding config files for equivalent NeoX models to those found in Hugging face.

Example usage: (Converts the 70M Pythia model to NeoX format)
================================================================
OMPI_COMM_WORLD_RANK=0 CUDA_VISIBLE_DEVICES=0 python tools/ckpts/convert_hf_to_sequential.py \
    --hf-model-name pythia-70m-v0 \
    --revision 143000 \
    --output-dir checkpoints/neox_converted/pythia/70m \
    --cache-dir checkpoints/HF \
    --config configs/pythia/70M.yml configs/local_setup.yml \
    --test


For multi-gpu support we must initialize deepspeed:
NOTE: This requires manually changing the arguments below.
================================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./deepy.py tools/ckpts/convert_hf_to_sequential.py \
    -d configs pythia/70M.yml local_setup.yml
"""

MULTI_GPU_ARGS = " ".join(
    [
        "--hf-model-name pythia-70m-v0",
        "--revision 143000",
        "--output-dir checkpoints/neox_converted/pythia/70m",
        "--cache-dir checkpoints/HF",
        "--config configs/pythia/70M.yml configs/local_setup.yml",
        "--test",
    ]
)


def convert_hf_to_sequential(hf_model, seq_state_dict):
    """Converts the weights of a HuggingFace model to neox 2.0 format.

    :param hf_model: the huggingface model
    :param seq_state_dict: the state dict of the equivalent neox model

    returns the updated sequential state dict
    """
    num_layers = hf_model.config.num_hidden_layers
    # Embedding is layer idx 0
    seq_state_dict[
        "sequential.0.word_embeddings.weight"
    ] = hf_model.gpt_neox.embed_in.state_dict()["weight"]

    for layer_hf in range(num_layers):
        # offset by 2
        layer_seq = layer_hf + 2

        # get layer from hf model
        hf_layer = hf_model.gpt_neox.layers[layer_hf]
        hf_layer_sd = hf_layer.state_dict()

        for key in hf_model.gpt_neox.layers[0].state_dict().keys():

            if key in ["attention.bias", "attention.masked_bias"]:
                continue
            seq_state_dict[f"sequential.{layer_seq}.{key}"] = hf_layer_sd[key]

    # Load final layer norm
    layer_seq = num_layers + 3
    seq_state_dict[
        f"sequential.{layer_seq}.norm.weight"
    ] = hf_model.gpt_neox.final_layer_norm.state_dict()["weight"]
    seq_state_dict[
        f"sequential.{layer_seq}.norm.bias"
    ] = hf_model.gpt_neox.final_layer_norm.state_dict()["bias"]

    # output embedding / LM head
    layer_seq += 1
    seq_state_dict[
        f"sequential.{layer_seq}.final_linear.weight"
    ] = hf_model.embed_out.state_dict()["weight"]


def shard_sequential_mp(num_mp_ranks, sequential):
    """Shards the sequential model into model parallel ranks.

    :param num_mp_ranks: the number of model parallel ranks
    :param sequential: the state dict of the sequential model at mp=1

    returns a dict of state dicts for each mp rank
    """
    ranks = {x: dict() for x in range(num_mp_ranks)}
    for k, v in sequential.items():
        if reduce(
            np.logical_or,
            [
                x in k
                for x in [
                    "dense_4h_to_h.bias",
                    "attention.dense.bias",
                ]
            ],
        ):
            # Divide by tp_size since they get added together
            for x in range(num_mp_ranks):
                ranks[x][k] = v / num_mp_ranks
        elif reduce(
            np.logical_or,
            [
                x in k
                for x in [
                    "layernorm",
                    "rotary_emb",
                    "norm.weight",
                    "norm.bias",
                ]
            ],
        ):
            # no splitting
            for x in range(num_mp_ranks):
                ranks[x][k] = v
        else:
            if len(v.shape) == 1:
                size_per_rank = v.shape[0] / num_mp_ranks
                if size_per_rank % 128 != 0.0:
                    padded_size = (128 - (size_per_rank % 128)) + size_per_rank
                    size_diff = int((padded_size * 4) - v.shape[max_])
                    zero_pad = torch.zeros((size_diff))
                    v = torch.cat([v, zero_pad], dim=max_)
                else:
                    padded_size = size_per_rank

                assert size_per_rank % 1.0 == 0.0
                assert padded_size % 1.0 == 0.0

                padded_size = int(padded_size)
                size_per_rank = int(size_per_rank)

                for x in range(num_mp_ranks):
                    if size_per_rank != padded_size:
                        # need to pad
                        ranks[x][k] = v[padded_size * x : padded_size * (x + 1)]
                    else:
                        ranks[x][k] = v[size_per_rank * x : size_per_rank * (x + 1)]

            elif len(v.shape) == 2:

                if reduce(
                    np.logical_or,
                    [
                        x in k
                        for x in [
                            "attention.dense.weight",
                            "mlp.dense_4h_to_h.weight",
                        ]
                    ],
                ):  # column parallel
                    max_, min_ = 1, 0
                elif reduce(
                    np.logical_or,
                    [
                        x in k
                        for x in [
                            "mlp.dense_h_to_4h.weight",
                            "mlp.dense_h_to_4h.bias",
                            "attention.query_key_value.weight",
                            "attention.query_key_value.bias",
                            "word_embeddings.weight",
                            "final_linear.weight",
                        ]
                    ],
                ):
                    # row parallel
                    max_, min_ = 0, 1
                else:
                    raise Exception("Unknown weight to shard: {}".format(k))

                size_per_rank = v.shape[max_] / num_mp_ranks
                if size_per_rank % 128 != 0.0:
                    padded_size = (128 - (size_per_rank % 128)) + size_per_rank
                    size_diff = int((padded_size * num_mp_ranks) - v.shape[max_])

                    assert (
                        size_diff > 0
                    ), "[ERROR] size diff is negative: {} for size_per_rank: {}, k:{}, shape:{}, padded_size:{}".format(
                        size_diff, size_per_rank, k, v.shape, padded_size
                    )

                    zero_pad = (
                        torch.zeros((size_diff, v.shape[min_]))
                        if max_ == 0
                        else torch.zeros((v.shape[min_], size_diff))
                    )

                    v = torch.cat([v, zero_pad], dim=max_)
                else:
                    padded_size = size_per_rank

                assert size_per_rank % 1.0 == 0.0
                assert padded_size % 1.0 == 0.0

                padded_size = int(padded_size)
                size_per_rank = int(size_per_rank)

                for x in range(num_mp_ranks):
                    if size_per_rank != padded_size:
                        # need to pad
                        ranks[x][k] = (
                            v[padded_size * x : padded_size * (x + 1), :]
                            if max_ == 0
                            else v[:, padded_size * x : padded_size * (x + 1)]
                        )
                    else:
                        ranks[x][k] = (
                            v[size_per_rank * x : size_per_rank * (x + 1), ...]
                            if max_ == 0
                            else v[:, size_per_rank * x : size_per_rank * (x + 1)]
                        )

            else:
                raise NotImplementedError()

    return ranks


def replace_sharded_seq(mp_checkpoints, mp_sharded_seq):
    """replaces the values within checkpointed configs with those
    from the sharded sequential object."""

    for mp_idx, shard in mp_sharded_seq.items():
        mp_key = f"mp_rank_{mp_idx:02}_model_states.pt"

        # use for loop instead of direct assignment
        # to check for compatibility
        for k, v in mp_checkpoints[mp_key]["module"].items():
            try:
                mp_checkpoints[mp_key]["module"][k] = shard[k]
            except KeyError:
                print("ERROR key:{} not found in shard.".format(k))


def shard_pp(sequential, mp_rank, num_layers):
    """Shards the model into layers.

    :param sequential: the state dict of the sequential model at mp=1
    :param mp_rank: the model parallel rank of the layers

    returns a dict of state dicts for each layer
    """
    suffix = f"-model_{mp_rank:02}-model_states.pt"

    layers_seq = dict()
    layers_seq[f"layer_00" + suffix] = {
        "word_embeddings.weight": sequential[f"sequential.0.word_embeddings.weight"]
    }
    layers_seq[f"layer_{num_layers+3:02}" + suffix] = {
        "norm.weight": sequential[f"sequential.{num_layers+3}.norm.weight"],
        "norm.bias": sequential[f"sequential.{num_layers+3}.norm.bias"],
    }

    layers_seq[f"layer_{num_layers+4:02}" + suffix] = {
        "final_linear.weight": sequential[
            f"sequential.{num_layers+4}.final_linear.weight"
        ]
    }

    for layer in range(2, num_layers + 2):
        layer_keys = [x for x in sequential if ".{}.".format(layer) in x]
        layers_seq[f"layer_{layer:02}" + suffix] = {
            k.split(".{}.".format(layer))[1]: sequential[k] for k in layer_keys
        }

    return layers_seq


def shard_pp_mp(num_mp_ranks, sequential, num_layers):
    """Shards the model into layers and model parallel ranks.

    :param num_mp_ranks: the number of model parallel ranks
    :param sequential: the state dict of the sequential model at mp=1
    :param num_layers: the number of layers in the model

    returns a dict of state dicts for each layer for each model parallel rank
    """
    mp_sharded = shard_sequential_mp(num_mp_ranks=num_mp_ranks, sequential=sequential)

    layers_pp_mp = {}
    for mp_rank, d in mp_sharded.items():
        layers_pp_mp.update(
            shard_pp(sequential=d, mp_rank=mp_rank, num_layers=num_layers)
        )
    return layers_pp_mp


def convert(hf_model, ckpt_dir, output_dir):
    """Converts a huggingface model to a NeoX checkpoint for different
        model parallel and pipeline parallel settings degrees.

    :param hf_model: the huggingface model
    :param ckpt_dir: the directory containing the NeoX checkpoint
    :param output_dir: the directory to save the converted checkpoint
    returns None
    """

    os.listdir(ckpt_dir)

    ckpts, layers = {}, {}
    for x in os.listdir(ckpt_dir):
        if x.startswith("mp_rank"):
            ckpts[x] = torch.load(os.path.join(ckpt_dir, x))
        elif x.startswith("layer"):
            layers[x] = torch.load(os.path.join(ckpt_dir, x))

    assert len(layers) + len(ckpts) > 0, "No checkpoints found in {}".format(ckpt_dir)

    os.makedirs(output_dir, exist_ok=True)
    seq_state_dict = dict()
    convert_hf_to_sequential(hf_model, seq_state_dict)

    if len(ckpts) == 1 and len(layers) == 0:
        # pp=0, mp=1
        key = list(ckpts.keys())[0]
        ckpts[key]["module"] = seq_state_dict
        to_save = ckpts

    elif len(ckpts) > 1 and len(layers) == 0:
        # pp=0, mp>1
        sharded_seq = shard_sequential_mp(
            num_mp_ranks=len(ckpts), sequential=seq_state_dict
        )
        replace_sharded_seq(mp_checkpoints=ckpts, mp_sharded_seq=sharded_seq)
        to_save = ckpts

    elif len(ckpts) == 1 and len(layers) > 1:
        # pp>0, mp==1
        to_save = shard_pp(
            sequential=seq_state_dict,
            mp_rank=0,
            num_layers=hf_model.config.num_hidden_layers,
        )

    elif len(ckpts) > 1 and len(layers) > 1:
        # pp>0, mp>1
        to_save = shard_pp_mp(
            num_mp_ranks=len(ckpts),
            sequential=seq_state_dict,
            num_layers=hf_model.config.num_hidden_layers,
        )

    else:
        raise NotImplementedError(
            "Not implemented for len(ckpts)={} and len(layers)={}".format(
                len(ckpts), len(layers)
            )
        )

    for k, v in to_save.items():
        print("saving {}...".format(os.path.join(output_dir, k)))
        torch.save(v, os.path.join(ckpt_dir, k))

    # copy the checkpoint to the output_dir
    print("rm {}/*".format(output_dir))
    os.system("rm {}/*".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    print("cp {} {}".format(os.path.join(ckpt_dir, "*"), output_dir))
    os.system("cp {} {}".format(os.path.join(ckpt_dir, "*"), output_dir))

    # set latest file within the output_dir
    latest_file = os.path.join("/".join(output_dir.split("/")[:-1]), "latest")
    os.system("rm " + latest_file)
    with open(latest_file, "w") as f:
        f.write(output_dir.split("/")[-1])


def consume_neox_args2(args_parsed, overwrite_values=None):
    """
    Deepspeed launcher needs to pass the arguments for `pretrain_gpt2.py` across to all machines.

    In order not to have any problems with different configs being mismatched across machines, we instead read the .yaml configuration file from the main rank,
    then serialize the arguments to a dictionary, which the deepspeed launcher broadcasts to all machines (`--megatron_config`).

    We then instantiate a new NeoXArgs from the dictionary (`.from_dict`). This should ensure args are never inconsistent across machines.
    """
    def decode_base64_and_load_json(base64_string):
        try:
            decoded_string = base64.b64decode(base64_string).decode('utf-8')

            json_data = json.loads(decoded_string)
            return json_data
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    megatron_config = decode_base64_and_load_json(args_parsed.megatron_config)
    if args_parsed.deepspeed_config is not None:
        overwrite_values = NeoXArgs.set_up_autotuning(
            args_parsed.deepspeed_config, overwrite_values
        )
    if overwrite_values is not None:
        megatron_config.update(overwrite_values)
    return NeoXArgs.from_dict(args_dict=megatron_config)


def get_non_existing_dir(tmp_dir):
    while os.path.exists(tmp_dir):
        tmp_dir = os.path.join(tmp_dir, "tmp_dir")
    return tmp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face GPT-NeoX model back to a sequential model compatible with GPT-NeoX training."
    )
    parser.add_argument(
        "--revision",
        type=int,
        default=143000,
        help="Revision or step of the Pythia model to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to save the converted GPT-NeoX model checkpoint.",
    )
    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        help="Path to the config file for the equivalent NeoX model.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, will run a test to ensure the conversion was successful.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="If set, script will only download the model and not convert it.",
    )

    parser.add_argument(
        "--ckpt-tmp-dir",
        default="/tmp/ckpt_tmp_dir",
        help="Directory to store cached hugging face checkpoints. [WARNING: MUST BE VISIBLE TO ALL RANKS]",
    )
    parser.add_argument(
        "--hf-model-name",
        type=str,
        help="Name of the hugging face model to download from EleutherAI/{hf-model-name}.}",
    )

    parser.add_argument(
        "--cache-dir",
        default="/gpfs/alpine/csc499/proj-shared/hf_checkpoints",
        help="Directory to store cached hugging face checkpoints.",
    )
    try:
        if int(os.environ["WORLD_SIZE"]) > 1:
            args = parser.parse_args(MULTI_GPU_ARGS.split(" "))
        else:
            args = parser.parse_args()
    except KeyError:
        args = parser.parse_args()

    tmp_cache_dir = get_non_existing_dir(args.ckpt_tmp_dir)

    if args.download_only:
        hf_model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/{args.hf_model_name}",
            revision=f"step{args.revision}",
            cache_dir=os.path.join(
                args.cache_dir, f"{args.hf_model_name}/step{args.revision}"
            ),
        ).half()
        exit(0)
    else:
        print("======================================================================")
        print(
            "Warning the following script will delete files within {}".format(
                args.output_dir
            )
        )
        print(
            "Warning the following script will delete this directory {}".format(
                tmp_cache_dir
            )
        )
        print("======================================================================")
        # time.sleep(5)

    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1)) > 1:
        neox_args = consume_neox_args2(args)
    else:
        neox_args = NeoXArgs.from_ymls(args.config)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()
    neox_args.initialize_tensorboard_writer()
    neox_args.comet()

    # setup logging and timers
    # init_wandb(neox_args=neox_args)
    # timers = Timers(
    #     use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer
    # )
    initialize_megatron(neox_args=neox_args)

    torch.distributed.barrier()

    model = get_model(neox_args=neox_args, use_cache=True)
    optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
    lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        # args=neox_args,
        lr_scheduler=lr_scheduler,
        dist_init_required=False,
        model_parameters=None,
        config_params=neox_args.deepspeed_config,
        mpu=mpu,
    )

    if os.environ.get("OMPI_COMM_WORLD_RANK", "1") == "0":
        os.makedirs(f"{tmp_cache_dir}", exist_ok=True)

    torch.distributed.barrier()
    neox_args.save = tmp_cache_dir

    save_checkpoint(
        neox_args=neox_args,
        iteration=0,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    print(os.listdir(f"{tmp_cache_dir}"))
    ckpt_dir = os.path.join(tmp_cache_dir, "global_step0")

    if torch.distributed.get_rank() == 0:
        config = GPTNeoXConfig.from_pretrained(
            f"EleutherAI/{args.hf_model_name}",
            revision=f"step{args.revision}",
            cache_dir=os.path.join(
                args.cache_dir, f"{args.hf_model_name}/step{args.revision}"
            ),
        )
        # does not change the weights, but is needed to align logits
        config.update({"hidden_act": "gelu_fast"})
        hf_model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/{args.hf_model_name}",
            revision=f"step{args.revision}",
            config=config,
            cache_dir=os.path.join(
                args.cache_dir, f"{args.hf_model_name}/step{args.revision}"
            ),
        ).half()
        print("==========================================")
        print("Loaded Hugging Face model successfully!")
        print("==========================================")
        convert(hf_model, ckpt_dir=ckpt_dir, output_dir=args.output_dir)

        if os.environ.get("OMPI_COMM_WORLD_RANK", "1") == "0":
            # cleanup temp dir
            os.system(f"rm -r {tmp_cache_dir}")

    torch.distributed.barrier()

    # verify the conversion can be loaded
    neox_args.load = "/".join(args.output_dir.split("/")[:-1])
    print(neox_args.load)
    neox_args.finetune = True
    load_checkpoint(
        neox_args=neox_args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        iteration=None,
    )
    print("==========================================")
    print("Converted checkpoint successfully loaded!")
    print("==========================================")

    if args.test and torch.distributed.get_world_size() == 1:
        # only implemented for world size 1

        with torch.no_grad():
            # torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True) #setting the CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable is required for this to work (tested for A6000)
            model.eval()
            hf_model.eval()

            b = 10
            seq_len = 32
            inputs = torch.randint(0, 50304, (b, seq_len), dtype=torch.long).cuda()
            mask = (
                (torch.triu(torch.ones(seq_len, seq_len)) != 1).transpose(0, 1).cuda()
            )
            pos_ids = torch.arange(0, seq_len).unsqueeze(0).cuda()

            torch.manual_seed(0)
            outputs_neox = model.cuda()(
                (inputs, pos_ids, mask.unsqueeze(0).unsqueeze(0)), neox_args=neox_args
            )

            torch.manual_seed(0)
            outputs = hf_model.cuda()(input_ids=inputs)

            print("HF logits   .sum(): ", outputs.logits.to(torch.float32).sum())
            print("NeoX logits .sum(): ", outputs_neox.to(torch.float32).sum())

            print(
                "\nLogit comparison summary for {} sequences of length {}:".format(
                    b, seq_len
                )
            )
            print("=============================================================")
            for i in range(b):
                abs_diff = (
                    outputs.logits[i, ...].to(torch.float32)
                    - outputs_neox[i, ...].to(torch.float32)
                ).abs()
                print(
                    "[Random sequence {}] (hflogits - neoxlogits).abs() -- mean: {:.5f}\tmax: {:.5f}\tmin: {:.5f}\tmedian: {:.5f}".format(
                        i,
                        abs_diff.mean(),
                        abs_diff.max(),
                        abs_diff.min(),
                        abs_diff.median(),
                    )
                )

    elif args.test:
        print(
            "[INFO] Checkpoint conversion logit test not implemented for distributed world_size > 1. Current world_size: {}".format(
                torch.distributed.get_world_size()
            )
        )
