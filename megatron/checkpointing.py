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

"""Input/output checkpointing."""

import json
import math
import os
import re
import shutil
import time
import random
import sys
import numpy as np

try:
    import boto3
except ModuleNotFoundError:
    print(
        "For s3 checkpointing, please install boto3 either using requirements/requirements-s3.txt or https://github.com/boto/boto3"
    )
try:
    import hf_transfer
except ModuleNotFoundError:
    print(
        "For s3 checkpointing, please install hf_transfer either using requirements/requirements-s3.txt or https://github.com/huggingface/hf_transfer"
    )
import torch
from glob import glob

from megatron import mpu
from megatron import print_rank_0
from megatron.utils import natural_sort
from megatron.text_generation_utils import get_batch, forward_model
from pathlib import Path
from pprint import pformat


def check_checkpoint_args(neox_args, checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""

    assert isinstance(checkpoint_args, dict), "args stored in checkpoint is a dict"
    for checkpoint_arg_name, checkpoint_arg_value in checkpoint_args.items():
        args_value = getattr(neox_args, checkpoint_arg_name)
        error_message = "{} value from checkpoint ({}) is not equal to the currently set argument value ({}).".format(
            checkpoint_arg_name, checkpoint_arg_value, args_value
        )
        assert checkpoint_arg_value == args_value, error_message


def do_forward_pass(neox_args, model, inference=False):

    # set to eval mode
    model_was_in_train = model.training
    model.eval()

    # get context tokens
    # always forward full batch size
    context_tokens_tensor = (
        torch.arange(neox_args.seq_length + 1)
        .repeat((neox_args.train_micro_batch_size_per_gpu, 1))
        .cuda()
    )

    # forward
    if inference:
        tokens, attention_mask, position_ids = get_batch(
            neox_args, context_tokens_tensor[:, : neox_args.seq_length]
        )
        model_inputs = (
            tokens,
            position_ids,
            attention_mask,
            torch.Tensor(),
        )
        logits, _ = forward_model(neox_args, model, model_inputs)
    elif neox_args.is_pipe_parallel:
        data_iterator = iter([{"text": context_tokens_tensor}])
        _, logits = model.eval_batch(data_iter=data_iterator, return_logits=True)
    else:
        tokens, attention_mask, position_ids = get_batch(
            neox_args, context_tokens_tensor[:, : neox_args.seq_length]
        )
        output = model((tokens, position_ids, attention_mask))
        logits = output[0] if isinstance(output, tuple) else output


    # reset to train mode, if model was in training before
    if model_was_in_train:
        model.train()

    if logits is not None:
        logits = logits.detach().cpu()[
            0
        ]  # just return first batch item (they are all equal)

    return logits


def check_forward_pass(neox_args, model, checkpoint_logits, inference):
    # do forward pass with loaded checkpoint
    logits = do_forward_pass(neox_args=neox_args, model=model, inference=inference)

    # check
    if (
        logits is not None and checkpoint_logits is not None
    ):  # this could be the case for non-final pipeline stages
        if not (logits == checkpoint_logits).all().item():
            if mpu.get_data_parallel_rank() == 0:
                print(
                    " > WARNING: validate_checkpoint_forward() forward after load of checkpoint does not yield exactly same result"
                )
            assert (
                torch.isclose(logits, checkpoint_logits).all().item()
            ), "validate_checkpoint_forward() forward after load of checkpoint does not yield a close result"


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_name(checkpoints_path, iteration, release=False, mp_rank=None):
    """A unified checkpoint name."""
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    return os.path.join(
        checkpoints_path,
        directory,
        "mp_rank_{:02d}".format(
            mpu.get_model_parallel_rank() if mp_rank is None else mp_rank
        ),
        "model_optim_rng.pt",
    )


def get_checkpoint_tag(iteration: int) -> str:
    return f"global_step{iteration}"


def delete_old_checkpoints(save_dir, n_to_keep):
    if torch.distributed.get_rank() == 0:
        ckpt_dir_regex = r"global_step[\d]*"
        if save_dir.endswith("/"):
            save_dir = save_dir.strip("/")
        all_ckpts = natural_sort(
            [
                i
                for i in glob(f"{save_dir}/*")
                if os.path.isdir(i) and re.search(ckpt_dir_regex, i)
            ]
        )
        n_to_delete = len(all_ckpts) - n_to_keep
        if n_to_delete > 0:
            to_delete = all_ckpts[:n_to_delete]
            print(f"WARNING: Deleting old checkpoints: \n\t{', '.join(to_delete)}")
            for ckpt in to_delete:
                try:
                    shutil.rmtree(ckpt)
                except FileNotFoundError:
                    pass


def save_ds_checkpoint(iteration, model, neox_args):
    """Save a model checkpoint."""
    sd = {
        "iteration": iteration,
        "args": {
            "num_layers": neox_args.num_layers,
            "hidden_size": neox_args.hidden_size,
            "num_attention_heads": neox_args.num_attention_heads,
            "max_position_embeddings": neox_args.max_position_embeddings,
            "make_vocab_size_divisible_by": neox_args.make_vocab_size_divisible_by,
            "padded_vocab_size": neox_args.padded_vocab_size,
            "tokenizer_type": neox_args.tokenizer_type,
            "model_parallel_size": neox_args.model_parallel_size,
        },
    }
    # rng states.
    if not neox_args.no_save_rng:
        sd["random_rng_state"] = random.getstate()
        sd["np_rng_state"] = np.random.get_state()
        sd["torch_rng_state"] = torch.get_rng_state()
        sd["cuda_rng_state"] = torch.cuda.get_rng_state()
        sd["rng_tracker_states"] = mpu.get_cuda_rng_tracker().get_states()

    if neox_args.checkpoint_validation_with_forward_pass:
        logits = do_forward_pass(neox_args=neox_args, model=model)
        sd["checkpoint_validation_logits"] = logits

    # checkpoint folder name
    tag = get_checkpoint_tag(iteration)

    # save checkpoint
    model.save_checkpoint(neox_args.save, tag=tag, client_state=sd)

    # save config files
    if torch.distributed.get_rank() == 0 and neox_args.config_files is not None:
        configs_directory = os.path.join(neox_args.save, tag, "configs")
        os.makedirs(configs_directory, exist_ok=True)
        for config_filename, config_data in neox_args.config_files.items():
            with open(os.path.join(configs_directory, config_filename), "w") as f:
                if isinstance(config_data, str):
                    f.write(config_data)
                else:
                    json.dump(config_data, f)


def multiprocessing_starmap(func, args, num_processes=None):
    """Wrapper to allow for re-usable multiprocessing pools with `spawn` context handling
    Args:
        func (Callable): Function to call
        args (Iterable): Iterable of arguments to pass to `func`
        num_processes (int, optional): Number of processes to spawn. Defaults to `multiprocessing.cpu_count() - 1`
    """
    import multiprocessing

    num_processes = num_processes or (multiprocessing.cpu_count() - 1)
    with multiprocessing.get_context("spawn").Pool(
        processes=num_processes
    ) as process_pool:
        process_pool.starmap(func, args)
        process_pool.terminate()
        process_pool.join()
        del process_pool


def _upload(
    file_path: str,
    s3_key: str,
    chunk_size: int = 104_857_600,
    max_files: int = 64,
    parallel_failures: int = 63,
    max_retries: int = 5,
):
    """Upload local file to S3 using `hf_transfer` library
    Args:
        file_path (str): Local filename to upload
        s3_key (str): S3 key to upload to. E.g. `s3://bucket-name/path/to/file`
        chunk_size (int, optional): Chunk size to use for multipart upload.
            Defaults to 100MiB = 104_857_600
        max_files (int, optional):  Number of open file handles, which determines
            the maximum number of parallel downloads. Defaults to 64
        parallel_failures (int, optional): Number of maximum failures of different
            chunks in parallel (cannot exceed max_files). Defaults to 63
        max_retries (int, optional): Number of retries for each chunk. Defaults to 5
    """
    s3 = boto3.client("s3")
    bucket = s3_key.split("s3://")[1].split("/")[0]
    key = s3_key.split(bucket)[1].lstrip("/")

    # 1. Init multipart upload and obtain unique upload identifier
    upload = s3.create_multipart_upload(
        ACL="bucket-owner-full-control",
        Bucket=bucket,
        Key=key,
    )
    upload_id = upload["UploadId"]

    # 2. Generate presigned URLs for each part
    file_size = os.stat(file_path).st_size
    urls = []
    nb_parts = math.ceil(file_size / chunk_size)
    for part_number in range(1, nb_parts + 1):
        params = {
            "Bucket": bucket,
            "Key": key,
            "PartNumber": part_number,
            "UploadId": upload_id,
        }
        urls.append(
            s3.generate_presigned_url(
                ClientMethod="upload_part", Params=params, ExpiresIn=86400
            )
        )

    # 3. Upload parts in parallel
    responses = hf_transfer.multipart_upload(
        file_path=file_path,
        parts_urls=urls,
        chunk_size=chunk_size,
        max_files=max_files,
        parallel_failures=parallel_failures,
        max_retries=max_retries,
    )

    # 4. Complete multipart upload request with ETag values
    etag_with_parts = []
    for part_number, header in enumerate(responses):
        etag = header.get("etag")
        etag_with_parts.append({"ETag": etag, "PartNumber": part_number + 1})
    parts = {"Parts": etag_with_parts}
    s3.complete_multipart_upload(
        Bucket=bucket, Key=key, MultipartUpload=parts, UploadId=upload_id
    )


def upload_checkpoint(iteration, neox_args):
    local_checkpoint_path = os.path.join(
        os.path.abspath(neox_args.save), get_checkpoint_tag(iteration)
    )
    local_checkpoint_list = sorted(
        filter(
            lambda x: os.path.isfile(x),
            [str(p) for p in Path(local_checkpoint_path).rglob("*")],
        )
    )
    remote_checkpoint_path = os.path.join(
        neox_args.s3_path,
        os.path.basename(neox_args.save),
        get_checkpoint_tag(iteration),
    )
    remote_checkpoint_list = [
        os.path.join(
            remote_checkpoint_path,
            os.path.relpath(local_checkpoint, local_checkpoint_path),
        )
        for local_checkpoint in local_checkpoint_list
    ]
    inputs = zip(
        local_checkpoint_list,
        remote_checkpoint_list,
        [neox_args.s3_chunk_size] * len(local_checkpoint_list),
    )

    print_rank_0(
        f"[RANK {torch.distributed.get_rank()}] Uploading checkpoint `{local_checkpoint_path}` to `{remote_checkpoint_path}`..."
    )
    start = time.time()
    multiprocessing_starmap(_upload, inputs)
    total_time = time.time() - start
    print_rank_0(
        f"[RANK {torch.distributed.get_rank()}] Uploaded checkpoint `{local_checkpoint_path}` to `{remote_checkpoint_path}` in {total_time:.2f}s"
    )


def save_checkpoint(neox_args, iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint."""

    if neox_args.deepspeed:
        save_ds_checkpoint(iteration, model, neox_args)
    else:
        raise ValueError("Must be using deepspeed to use neox")

    torch.distributed.barrier()
    upload_to_s3 = torch.distributed.get_rank() == 0 and neox_args.s3_path is not None
    if upload_to_s3:
        upload_checkpoint(iteration, neox_args)

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    if neox_args.keep_last_n_checkpoints is not None:
        delete_old_checkpoints(neox_args.save, neox_args.keep_last_n_checkpoints)

    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def load_checkpoint(
    neox_args, model, optimizer, lr_scheduler, inference=False, iteration=None
):
    """Load a model checkpoint and return the iteration."""
    if neox_args.deepspeed:
        load_optim_and_scheduler = (
            not neox_args.no_load_optim
        )  # TODO: These should be configured by separate args
        if neox_args.finetune:
            load_optim_and_scheduler = False
        if iteration is not None:
            tag = get_checkpoint_tag(iteration)
        else:
            tag = None
        checkpoint_name, state_dict = model.load_checkpoint(
            neox_args.load,
            load_optimizer_states=load_optim_and_scheduler,
            load_lr_scheduler_states=load_optim_and_scheduler,
            load_module_only=not load_optim_and_scheduler,
            tag=tag,
            load_module_strict=neox_args.train_impl != "rm",
        )

        if checkpoint_name is None:
            # if an iteration is specified, we want to raise an error here rather than
            # continuing silently, since we are trying to load a specific checkpoint
            if iteration is not None:
                available_checkpoints = sorted(
                    [
                        int(i.name.replace("global_step", ""))
                        for i in Path(neox_args.load).glob("global_step*")
                    ]
                )
                raise ValueError(
                    f"Unable to load checkpoint for iteration {iteration}. \nAvailable iterations: {pformat(available_checkpoints)}"
                )
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")

            return 0  # iteration 0, if not checkpoint loaded
    else:
        raise ValueError("Must be using deepspeed to use neox")

    # Set iteration.
    if neox_args.finetune:
        iteration = 0
    else:
        if "iteration" in state_dict:
            iteration = state_dict["iteration"]
        else:
            iteration = state_dict.get(
                "total_iters"
            )  # total_iters backward compatible with older checkpoints
        if iteration is None:
            raise ValueError(
                f"Unable to load iteration from checkpoint {checkpoint_name} with keys {state_dict.keys()}, exiting"
            )

    # Check arguments.
    if "args" in state_dict:
        checkpoint_args = state_dict["args"]
        check_checkpoint_args(neox_args=neox_args, checkpoint_args=checkpoint_args)
        print_rank_0(
            " > validated currently set args with arguments in the checkpoint ..."
        )
    else:
        print_rank_0(" > could not find arguments in the checkpoint for validation...")

    # Check loaded checkpoint with forward pass
    if neox_args.checkpoint_validation_with_forward_pass:
        if "checkpoint_validation_logits" in state_dict:
            check_forward_pass(
                neox_args=neox_args,
                model=model,
                checkpoint_logits=state_dict["checkpoint_validation_logits"],
                inference=inference,
            )
            print_rank_0(" > validated loaded checkpoint with forward pass ...")
        else:
            if mpu.get_data_parallel_rank() == 0:
                print(
                    " > WARNING: checkpoint_validation_with_forward_pass is configured but no checkpoint validation data available in checkpoint {}".format(
                        checkpoint_name
                    )
                )

    # rng states.
    if not neox_args.finetune and not neox_args.no_load_rng:
        try:
            random.setstate(state_dict["random_rng_state"])
            np.random.set_state(state_dict["np_rng_state"])
            torch.set_rng_state(state_dict["torch_rng_state"])
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            mpu.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])
        except KeyError:
            print_rank_0(
                "Unable to load optimizer from checkpoint {}. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the optimizer state, "
                "exiting ...".format(checkpoint_name)
            )
            sys.exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print("  successfully loaded {}".format(checkpoint_name))

    return iteration
