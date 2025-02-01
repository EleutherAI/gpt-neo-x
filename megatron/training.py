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
#
# This file has been modified from its original version
#

"""Pretrain utilities."""
from datetime import datetime
from functools import partial
from collections import defaultdict

import math
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import deepspeed
from deepspeed.runtime.data_pipeline.curriculum_scheduler import CurriculumScheduler
import numpy as np

from megatron.utils import (
    Timers,
    init_wandb,
    get_ltor_masks_and_position_ids,
    reduce_losses,
)

from megatron import print_rank_0, mpu
from megatron.model import (
    GPT2ModelPipe,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
    mark_norms_for_sequence_parallel_grad_sync,
)
from megatron.mpu.mappings import gather_from_model_parallel_region
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.data.data_utils import (
    build_train_valid_test_data_loaders,
    shift_and_wrap_data_loaders,
)
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.logging import tb_wandb_log, training_log
from megatron.utils import (
    OverflowMonitor,
    get_noise_scale_logger,
    get_total_params,
    CharCounter,
)
from megatron.model.weight_server import start_server
from megatron.model.gpt2_model import cross_entropy
from megatron.mpu import vocab_parallel_cross_entropy

from pickle import dump
import os


def mup_weights_reinit(neox_args, model):
    def has_method(o, name):
        return callable(getattr(o, name, None))

    for layer in model.modules():
        # This normally would happen in set_base_shapes if we actually were able to use the MuReadout class
        if hasattr(layer, "mup_rescale_parameters") and layer.mup_rescale_parameters:
            layer._rescale_parameters()

        if has_method(layer, "mup_reinitialize_weights"):
            layer.mup_reinitialize_weights(neox_args)


def save_base_shapes(neox_args, base_shapes, use_cache):

    # Instantiation of the base model fails in the init function (init_functions.py) because we haven't called set_base_shapes on it at this point, so disable it temporarily here
    neox_args.use_mup = False

    base_model = GPT2ModelPipe(
        neox_args=neox_args,
        num_tokentypes=0,
        parallel_output=True if neox_args.train_impl != "rm" else False,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    if not neox_args.is_pipe_parallel:
        base_model = base_model.to_sequential()

    try:
        import mup
    except ModuleNotFoundError:
        print("Please install mup https://github.com/microsoft/mup")
        raise Exception

    base_shapes = mup.get_shapes(base_model)

    del base_model

    old_hidden_size = neox_args.hidden_size
    neox_args.hidden_size = neox_args.hidden_size * neox_args.mup_width_scale

    delta_model = GPT2ModelPipe(
        neox_args=neox_args,
        num_tokentypes=0,
        parallel_output=True if neox_args.train_impl != "rm" else False,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    if not neox_args.is_pipe_parallel:
        delta_model = delta_model.to_sequential()

    delta_shapes = mup.get_shapes(delta_model)

    # change back
    neox_args.use_mup = True
    neox_args.hidden_size = old_hidden_size

    save_shapes = f"{neox_args.base_shapes_file}.{torch.distributed.get_rank()}"
    print(f"saving base shapes at {save_shapes}")
    mup.make_base_shapes(base_shapes, delta_shapes, savefile=save_shapes)
    print(f"base shapes saved...exiting")
    sys.exit(1)


def mup_coord_check(neox_args, timers, lr_scheduler, train_data_iterator):
    from megatron.mup_substitute import get_coord_data
    from mup.coord_check import plot_coord_data

    def lazy_model(hidden_size):
        def gen():
            old_hidden_size = neox_args.hidden_size
            neox_args.hidden_size = hidden_size

            model, optimizer, _, _ = setup_model_and_optimizer(
                neox_args=neox_args, use_cache=False
            )

            neox_args.hidden_size = old_hidden_size

            return model

        return gen

    models = {}

    # Hidden size needs to be divisible by num attention heads
    for hidden_size in (neox_args.num_attention_heads * (2**p) for p in range(2, 9)):
        models[hidden_size] = lazy_model(hidden_size)

    neox_args.use_mup = True
    df_up = get_coord_data(
        neox_args, timers, lr_scheduler, models, train_data_iterator, mup=True
    )
    neox_args.use_mup = False
    df_sp = get_coord_data(
        neox_args, timers, lr_scheduler, models, train_data_iterator, mup=False
    )

    plot_coord_data(df_up, save_to=f"coord_check_up.{torch.distributed.get_rank()}.jpg")
    plot_coord_data(df_sp, save_to=f"coord_check_sp.{torch.distributed.get_rank()}.jpg")

    print_rank_0("Saved coord check plots... exiting")
    sys.exit(1)


def update_iterations(neox_args, data_loaders):
    """
    Compute the number of train iterations if not specified and num_epochs, updates the neox_args object.
    Note that if len(train_dataloader) % gradient_accumulation_steps != 0, this will configure neox
    to do as many iterations as possible while ensuring that each example is seen *at most* train_epochs
    times.
    """
    if (not neox_args.do_train) or (neox_args.train_iters is not None):
        pass
    elif neox_args.train_iters is None and neox_args.train_epochs is None:
        print_rank_0(
            "ERROR:Failed to specify either train_epochs or train_iters in config file"
        )
    else:
        global_rank = torch.distributed.get_rank()

        if global_rank == 0:
            train_dataloader = data_loaders["train"]
            train_epochs = neox_args.train_epochs
            gradient_accumulation_steps = neox_args.gradient_accumulation_steps

            train_dataloader_len = len(train_dataloader)
            train_iterations = (
                train_dataloader_len * train_epochs
            ) // gradient_accumulation_steps

            train_iters_tensor = torch.cuda.LongTensor([train_iterations])
        else:
            train_iters_tensor = torch.cuda.LongTensor([0])

        torch.distributed.broadcast(train_iters_tensor, src=0)

        neox_args.train_iters = train_iters_tensor[0].item()

        print_rank_0(
            f"Training for a total of {neox_args.train_iters} iterations, corresponding to {neox_args.train_epochs} epochs."
        )


def pretrain(neox_args):
    """Main training program.

    This function will run the following in the order provided:
        1) initialize Megatron.
        2) get train/val/test datasets.
        3) setup model, optimizer and lr schedule.
        4) configure data loading
        5) train the model.

    Arguments:
        neox_args: an instance of NeoXArgs containing the configuration for pretrain

    """
    # setup logging and timers
    init_wandb(neox_args=neox_args)
    timers = Timers(
        use_wandb=neox_args.use_wandb,
        tensorboard_writer=neox_args.tensorboard_writer,
        comet_experiment=neox_args.comet_experiment,
    )

    # Initialize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(neox_args=neox_args)

    # Create data loaders
    timers("train/valid/test data loaders").start()
    data_loaders = build_train_valid_test_data_loaders(neox_args=neox_args)
    update_iterations(neox_args=neox_args, data_loaders=data_loaders)
    timers("train/valid/test data loaders").stop()

    # Model, optimizer, and learning rate.
    timers("model and optimizer").start()
    model, optimizer, lr_scheduler, reference_model = setup_model_and_optimizer(
        neox_args=neox_args, use_cache=False, iteration=neox_args.iteration
    )
    timers("model and optimizer").stop()

    if neox_args.serve_model_weights:
        start_server(model)
        # sync...
        torch.distributed.barrier()

    # Start data stuff:

    # Make and configure iterators
    timers("train/valid/test data iterators").start()
    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    ) = shift_and_wrap_data_loaders(neox_args=neox_args, data_loaders=data_loaders)
    timers("train/valid/test data iterators").stop()

    if neox_args.use_mup and neox_args.coord_check:
        mup_coord_check(neox_args, timers, lr_scheduler, train_data_iterator)

    # Print setup timing.
    print_rank_0("done with setups ...")
    timers.log(
        [
            "train/valid/test data loaders",
            "model and optimizer",
            "train/valid/test data iterators",
        ]
    )
    print_rank_0("training ...")

    iteration = neox_args.iteration
    # edge case: save step 0 checkpoint if requested and we're starting from step 0
    if (
        neox_args.save
        and neox_args.extra_save_iters
        and 0 in neox_args.extra_save_iters
        and iteration == 0
    ):
        save_checkpoint(
            neox_args=neox_args,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    if neox_args.do_train and neox_args.train_iters > 0:
        iteration = train(
            neox_args=neox_args,
            timers=timers,
            model=model,
            reference_model=reference_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_data_iterator=train_data_iterator,
            valid_data_iterator=valid_data_iterator,
        )

    if neox_args.do_valid:
        prefix = "the end of training for val data"
        evaluate_and_print_results(
            neox_args=neox_args,
            prefix=prefix,
            forward_step_func=forward_step,
            data_iterator=valid_data_iterator,
            model=model,
            iteration=iteration,
            verbose=False,
            timers=timers,
            reference_model=reference_model,
        )

    if neox_args.save and iteration != 0:
        save_checkpoint(
            neox_args=neox_args,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    if neox_args.do_test:
        # Run on test data.
        prefix = "the end of training for test data"
        evaluate_and_print_results(
            neox_args=neox_args,
            prefix=prefix,
            forward_step_func=forward_step,
            data_iterator=test_data_iterator,
            model=model,
            iteration=iteration,
            verbose=True,
            timers=timers,
            chart_name="test",
            reference_model=reference_model,
        )


def _get_batch(neox_args, tokenizer, keys, data, datatype, label_mask_zero=False):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    data_b = mpu.broadcast_data(keys, data, datatype)
    token_key = keys[0]
    label_key = keys[1] if len(keys) > 1 else None
    # Unpack.
    tokens_ = data_b[token_key].long()
    if label_key in data_b:
        label_mask = (data_b[label_key].long() >= 0)[:, 1:].contiguous()
        labels = torch.where(
            data_b[label_key].long() >= 0,
            data_b[label_key].long(),
            torch.zeros_like(data_b[label_key].long()),
        )[:, 1:].contiguous()
    else:
        label_mask = (tokens_.long() >= 0)[:, 1:].contiguous()
        labels = tokens_[:, 1:].contiguous()
        if label_mask_zero:
            labels = labels * label_mask
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
        sliding_window_width=neox_args.sliding_window_width,
    )

    # combine loss masks from get_ltor_masks_and_position_ids with loss masks from data
    loss_mask = label_mask.to(loss_mask.dtype) * loss_mask
    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch(neox_args, data_iterator):
    """Generate a batch"""

    # Items and their type.
    if neox_args.train_impl in ["normal", "kto", "reinforce"]:
        keys = ["text", "label"] if neox_args.train_label_data_paths else ["text"]
    elif neox_args.train_impl in ["dpo", "rm"]:
        keys = (
            [["pos", "pos_label"], ["neg", "neg_label"]]
            if neox_args.pos_train_label_data_paths
            else [["pos"], ["neg"]]
        )
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    if neox_args.train_impl == "normal":
        return _get_batch(
            neox_args=neox_args,
            tokenizer=neox_args.tokenizer,
            keys=keys,
            data=data,
            datatype=datatype,
        )
    elif neox_args.train_impl == "kto":
        assert (
            neox_args.train_micro_batch_size_per_gpu > 1
        ), "For KTO training, the train_micro_batch_size_per_gpu must be greater than 1."
        tup = _get_batch(
            neox_args=neox_args,
            tokenizer=neox_args.tokenizer,
            keys=keys,
            data=data,
            datatype=datatype,
        )
        # Remove the last token from the reward since we predict the next token, so
        # Reward of <current prediction> will be based on the label of <next token>
        rw_data = mpu.broadcast_data(["reward"], data, torch.float)["reward"][
            :, :-1
        ].contiguous()
        ref_data = (
            mpu.broadcast_data(["ref"], data, torch.float)["ref"][:, :-1].contiguous()
            if neox_args.precompute_model_name
            else None
        )
        return tup + (rw_data, ref_data)
    elif neox_args.train_impl == "reinforce":

        tup = _get_batch(
            neox_args=neox_args,
            tokenizer=neox_args.tokenizer,
            keys=keys,
            data=data,
            datatype=datatype,
        )
        rw_data = mpu.broadcast_data(["reward"], data, torch.float)["reward"]
        raw_rw_data = mpu.broadcast_data(["raw_reward"], data, torch.float)[
            "raw_reward"
        ]
        return tup + (rw_data, raw_rw_data)
    elif neox_args.train_impl in ["dpo", "rm"]:
        pos_tup = _get_batch(
            neox_args=neox_args,
            tokenizer=neox_args.tokenizer,
            keys=keys[0],
            data=data,
            datatype=datatype,
            label_mask_zero=True,
        )
        neg_tup = _get_batch(
            neox_args=neox_args,
            tokenizer=neox_args.tokenizer,
            keys=keys[1],
            data=data,
            datatype=datatype,
            label_mask_zero=True,
        )
        if neox_args.precompute_model_name:
            ref_data = mpu.broadcast_data(["pos_ref", "neg_ref"], data, torch.float)
        else:
            ref_data = {"pos_ref": None}
        return [
            torch.cat((pos_item, neg_item), dim=0)
            for pos_item, neg_item in zip(pos_tup, neg_tup)
        ] + [
            torch.cat((ref_data["pos_ref"], ref_data["neg_ref"]), dim=0)[
                :, :-1
            ].contiguous()
            if ref_data["pos_ref"] is not None
            else None
        ]


def get_batch_pipe(data, neox_args, curr_scheduler=None):
    """A modification of get_batch() to work with the latest batch instead of an iterator."""

    assert neox_args.train_impl not in [
        "kto",
        "dpo",
        "rm",
    ], "Pipeline parallel is currently unsupported when using any of kto, dpo, rm. Set pipe_parallel_size to 0"

    # Items and their type.
    keys = ["text", "label"] if neox_args.train_label_data_paths else ["text"]
    datatype = torch.int64

    tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(
        neox_args, neox_args.tokenizer, keys, data, datatype
    )
    if curr_scheduler is not None:
        # iteration + 1 to align with how/when DeepSpeed updates the buffers
        curriculum_seqlen = curr_scheduler.update_difficulty(neox_args.iteration + 1)
        if curriculum_seqlen < tokens.size()[1]:
            # seqlen-based curriculum learning
            # input_ids, position_ids, labels have size [batch size, seqlen]
            # input_ids = input_ids[:, :curriculum_seqlen].contiguous()
            tokens = tokens[:, :curriculum_seqlen].contiguous()
            position_ids = position_ids[:, :curriculum_seqlen].contiguous()
            if labels is not None:
                labels = labels[:, :curriculum_seqlen].contiguous()
            if loss_mask is not None:
                loss_mask = loss_mask[:, :curriculum_seqlen].contiguous()
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask = attention_mask[
                :, :, :curriculum_seqlen, :curriculum_seqlen
            ].contiguous()

    # unpack data
    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def get_batch_sequential(forward_input, neox_args):
    """A modification of get_batch() to work with the latest batch instead of an iterator."""
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=forward_input[0],
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )
    return (forward_input[0], forward_input[1], attention_mask)


def forward_step(
    data_iterator,
    model,
    neox_args,
    timers,
    return_logits=False,
    is_train=False,
    reference_model=None,
):
    """Forward step."""
    if neox_args.is_pipe_parallel:
        return model.eval_batch(data_iterator, return_logits=return_logits)

    # Get the batch.
    if neox_args.memory_profiling and neox_args.iteration:
        torch.cuda.nvtx.range_push(f"Get batch")
    if timers is not None:
        timers("batch generator").start()
    if neox_args.train_impl == "normal":
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            neox_args=neox_args, data_iterator=data_iterator
        )
    elif neox_args.train_impl == "kto":
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            rewards,
            ref_logp,
        ) = get_batch(neox_args=neox_args, data_iterator=data_iterator)
    elif neox_args.train_impl == "reinforce":
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            rewards,
            raw_rewards,
        ) = get_batch(neox_args=neox_args, data_iterator=data_iterator)
    if neox_args.train_impl in ["dpo", "rm"]:
        tokens, labels, loss_mask, attention_mask, position_ids, ref_logp = get_batch(
            neox_args=neox_args, data_iterator=data_iterator
        )

    if timers is not None:
        timers("batch generator").stop()
    if neox_args.memory_profiling:
        torch.cuda.nvtx.range_pop()

    if neox_args.memory_profiling:
        torch.cuda.nvtx.range_push(f"Forward pass")
    metrics = {}
    if neox_args.train_impl == "normal":
        outputs = model((tokens, position_ids, attention_mask), neox_args=neox_args)
        if (
            is_train
            and neox_args.curriculum_learning
            and neox_args.curriculum_seqlen < neox_args.seq_length
        ):
            loss_mask = loss_mask[:, : neox_args.curriculum_seqlen].contiguous()
            labels = labels[:, : neox_args.curriculum_seqlen].contiguous()
        loss = cross_entropy(
            outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy
        )
    elif neox_args.train_impl == "rm":
        maybe_tuple = model((tokens, position_ids, attention_mask), neox_args=neox_args)
        if type(maybe_tuple) is tuple:
            outputs, _ = maybe_tuple
        else:
            outputs = maybe_tuple
        pos, neg = torch.chunk(outputs, 2, 0)
        pos_loss_mask, neg_loss_mask = torch.chunk(loss_mask, 2, 0)
        # We assume that each pos, neg pair occur in the same order
        # e.g. second nonzero pos is the corresponding second nonzero neg
        # and that there are also an equal number of pos and neg in each sequence.
        pos_indx = pos_loss_mask.nonzero()
        neg_indx = neg_loss_mask.nonzero()
        # indx[:, 0] is the batch index, indx[:, 1] is the token index, we only care about the token index.
        pos_indx = pos_indx[:, 1].unsqueeze(1)
        neg_indx = neg_indx[:, 1].unsqueeze(1)
        pos = torch.gather(pos.squeeze(), dim=1, index=pos_indx)
        neg = torch.gather(neg.squeeze(), dim=1, index=neg_indx)
        with torch.no_grad():
            metrics["pos_values"] = pos.clone().detach().mean()
            metrics["neg_values"] = neg.clone().detach().mean()
            metrics["margin"] = (pos - neg).clone().detach().mean()
            metrics["accuracy"] = ((pos - neg) > 0).clone().detach().float().mean()
        loss = (-F.logsigmoid(pos - neg).mean()) + (
            (neox_args.z_loss * (pos**2 + neg**2)).mean()
        )
    elif neox_args.train_impl == "dpo":
        # Based on https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90
        with torch.inference_mode():
            # So we can gather token logps...
            token_logp_labels = labels.clone()
            pos_loss_mask, neg_loss_mask = torch.chunk(loss_mask, 2, 0)
            if neox_args.dpo_reference_free:
                ref_pos = 0
                ref_neg = 0
            elif ref_logp is None:
                ref_maybe_tuple = reference_model(
                    (tokens, position_ids, attention_mask), neox_args=neox_args
                )
                if type(ref_maybe_tuple) is tuple:
                    # We should ignore MoE losses yeah?
                    ref_outputs, _ = ref_maybe_tuple
                else:
                    ref_outputs = ref_maybe_tuple
                ref_pos, ref_neg = get_pos_neg_logp(
                    ref_outputs, token_logp_labels, neox_args.dpo_fp32
                )
            else:
                ref_pos, ref_neg = torch.chunk(ref_logp, 2, 0)
            ref_pos = (ref_pos * pos_loss_mask).sum(-1)
            ref_neg = (ref_neg * neg_loss_mask).sum(-1)
        chosen_maybe_tuple = model(
            (tokens, position_ids, attention_mask), neox_args=neox_args
        )
        if type(chosen_maybe_tuple) is tuple:
            # We should ignore MoE losses yeah?
            chosen_outputs, _ = chosen_maybe_tuple
        else:
            chosen_outputs = chosen_maybe_tuple
        chosen_pos, chosen_neg = get_pos_neg_logp(
            chosen_outputs, token_logp_labels, neox_args.dpo_fp32
        )
        chosen_pos = (chosen_pos * pos_loss_mask).sum(-1)
        chosen_neg = (chosen_neg * neg_loss_mask).sum(-1)
        with torch.no_grad():
            # Collect metrics...
            if not neox_args.dpo_reference_free:
                metrics["ref_neg"] = ref_neg.clone().detach().mean()
                metrics["ref_pos"] = ref_pos.clone().detach().mean()
            metrics["chosen_neg"] = chosen_neg.clone().detach().mean()
            metrics["chosen_pos"] = chosen_pos.clone().detach().mean()
            if not neox_args.dpo_reference_free:
                chosen_rewards = neox_args.dpo_beta * (
                    chosen_pos.clone().detach() - ref_pos.clone().detach()
                )
                rejected_rewards = neox_args.dpo_beta * (
                    chosen_neg.clone().detach() - ref_neg.clone().detach()
                )
                metrics["chosen_rewards"] = chosen_rewards.mean()
                metrics["rejected_rewards"] = rejected_rewards.mean()
                reward_acc = (chosen_rewards > rejected_rewards).float()
                metrics["reward_acc"] = reward_acc.mean()
                metrics["margins"] = (chosen_rewards - rejected_rewards).mean()
        pi_logrations = chosen_pos - chosen_neg
        ref_logrations = ref_pos - ref_neg
        logits = pi_logrations - ref_logrations
        loss = -F.logsigmoid(neox_args.dpo_beta * logits).mean()
    elif neox_args.train_impl == "kto":
        # Based on https://github.com/huggingface/trl/blob/main/trl/trainer/kto_trainer.py
        # Except we don't have an extra input for KL logp, we just split the batch in half
        with torch.no_grad():
            # So we can gather token logps...
            token_logp_labels = labels.clone()
            token_logp_labels[token_logp_labels == -100] = 0
            if ref_logp is None:
                # Did not precompute logits....
                ref_maybe_tuple = reference_model(
                    (tokens, position_ids, attention_mask), neox_args=neox_args
                )
                if type(ref_maybe_tuple) is tuple:
                    # We should ignore MoE losses yeah?
                    ref_outputs, _ = ref_maybe_tuple
                else:
                    ref_outputs = ref_maybe_tuple
                # gather across tensor parallel group
                ref_outputs = gather_from_model_parallel_region(ref_outputs)

                ref_logp = get_logp(ref_outputs, token_logp_labels, neox_args.kto_fp32)
            else:
                print(f"REF LOGP: {ref_logp.clone().detach().mean()}")
            ref_logp = ref_logp * loss_mask
            scaling = (rewards.sum(-1) > 0.001).float() * neox_args.kto_desirable_weight
            scaling += (
                rewards.sum(-1) < -0.001
            ).float() * neox_args.kto_undesirable_weight
            pos_mask = (rewards > 0.001).float()
            neg_mask = (rewards < -0.001).float()
        chosen_maybe_tuple = model(
            (tokens, position_ids, attention_mask), neox_args=neox_args
        )
        if type(chosen_maybe_tuple) is tuple:
            # We should ignore MoE losses yeah?
            chosen_outputs, _ = chosen_maybe_tuple
        else:
            chosen_outputs = chosen_maybe_tuple
        chosen_outputs = gather_from_model_parallel_region(chosen_outputs)
        chosen_logp = get_logp(chosen_outputs, token_logp_labels, neox_args.kto_fp32)
        chosen_logp = chosen_logp * loss_mask
        with torch.no_grad():
            # Collect metrics...
            metrics["ref_logp"] = ref_logp.clone().detach().sum(-1).mean()
            metrics["policy_logp"] = chosen_logp.clone().detach().sum(-1).mean()
            metrics["pos_ref_logp"] = (
                (ref_logp * pos_mask).clone().detach().sum(-1).mean()
            )
            metrics["neg_ref_logp"] = (
                (ref_logp * neg_mask).clone().detach().sum(-1).mean()
            )
            metrics["pos_policy_logp"] = (
                (chosen_logp * pos_mask).clone().detach().sum(-1).mean()
            )
            metrics["neg_policy_logp"] = (
                (chosen_logp * neg_mask).clone().detach().sum(-1).mean()
            )
            metrics["kl"] = (
                chosen_logp.clone().detach() - ref_logp.clone().detach()
            ).sum() / loss_mask.sum()
            policy_rewards = (
                neox_args.kto_beta
                * rewards
                * (chosen_logp.clone().detach() - ref_logp.clone().detach())
            )
            reward_acc = (policy_rewards.sum(-1) > 0.0).float()
            metrics["reward_acc"] = reward_acc.mean()
            metrics["policy_rewards"] = policy_rewards.sum()
            print(metrics)
        pol_logp1, pol_logp2 = torch.chunk(chosen_logp, 2, 0)
        ref_logp1, ref_logp2 = torch.chunk(ref_logp, 2, 0)
        reward1, reward2 = torch.chunk(rewards, 2, 0)
        scaling1, scaling2 = torch.chunk(scaling, 2, 0)
        kl1 = torch.clamp((pol_logp1 - ref_logp1).sum(-1), min=0).mean()
        kl2 = torch.clamp((pol_logp2 - ref_logp2).sum(-1), min=0).mean()
        log_ratio1 = pol_logp1 - ref_logp1
        log_ratio2 = pol_logp2 - ref_logp2

        # TODO: Add pack_until_overflow sequence support
        loss = (
            0.5
            * scaling1.mean(-1)
            * (
                1
                - F.sigmoid(
                    (
                        neox_args.kto_beta
                        * reward1.mean(-1)
                        * (log_ratio1.sum(-1) - kl2.clone().detach())
                    )
                )
            )
        ) + (
            0.5
            * scaling2.mean(-1)
            * (
                1
                - F.sigmoid(
                    (
                        neox_args.kto_beta
                        * reward2.mean(-1)
                        * (log_ratio2.sum(-1) - kl1.clone().detach())
                    )
                )
            )
        )
        # print(loss.shape)
        loss = loss.mean()
        # print(loss.shape)
    elif neox_args.train_impl == "reinforce":
        if reference_model is not None:
            with torch.no_grad():
                ref_outputs = reference_model(
                    (tokens, position_ids, attention_mask), neox_args=neox_args
                )
                if type(ref_outputs) is tuple:
                    ref_outputs, _ = ref_outputs
                ref_outputs = ref_outputs
                if neox_args.kl_impl == "full":
                    # Have to do the loss over all tokens...
                    ref_outputs = gather_from_model_parallel_region(ref_outputs)
                    if neox_args.fp32_reinforce:
                        ref_outputs = ref_outputs.float()
                    ref_logp = ref_outputs.log_softmax(dim=-1).detach()
                    ref_per_token_logp = torch.gather(
                        ref_logp.clone(), dim=2, index=labels.unsqueeze(2)
                    ).squeeze(2)
                else:
                    ref_per_token_logp = get_logp(
                        ref_outputs, labels, neox_args.fp32_reinforce
                    )
                metrics["ref_logp"] = ref_per_token_logp.clone().detach().mean()
        outputs = model((tokens, position_ids, attention_mask), neox_args=neox_args)
        if type(outputs) is tuple:
            outputs, _ = outputs
        if neox_args.kl_impl == "full":
            # Have to do the loss over all tokens...
            outputs = gather_from_model_parallel_region(outputs)
            if neox_args.fp32_reinforce:
                outputs = outputs.float()
            logp = outputs.log_softmax(dim=-1)
            per_token_logp = torch.gather(
                logp.clone(), dim=2, index=labels.unsqueeze(2)
            ).squeeze(2)
        else:
            per_token_logp = get_logp(outputs, labels, neox_args.fp32_reinforce)
        with torch.no_grad():
            metrics["logp"] = per_token_logp.clone().detach().mean()
            metrics["reward"] = raw_rewards.clone().detach().mean()
            metrics["reward_std"] = raw_rewards.clone().detach().std()
        loss_mask_sum = loss_mask.sum()
        if reference_model is not None:
            if neox_args.kl_impl == "full":
                # Following along with
                # https://github.com/huggingface/trl/blob/104a02d207b63a4a062882aaff68f2d275493399/trl/trainer/ppo_trainer.py#L1109
                kl = F.kl_div(ref_logp, logp, log_target=True, reduction="none").sum(-1)
            else:
                kl = per_token_logp - ref_per_token_logp
                if neox_args.kl_impl == "abs":
                    kl = kl.abs()
                elif neox_args.kl_impl == "mse":
                    kl = 0.5 * (kl).square()
                elif neox_args.kl_impl == "kl":
                    pass
            with torch.no_grad():
                metrics["kl"] = kl.clone().detach().mean()
            loss = (-per_token_logp * rewards) + (neox_args.kl_div_beta * kl)
            loss = (loss * loss_mask).sum(-1) / loss_mask_sum
            loss = loss.mean()
        else:
            loss = -(rewards * per_token_logp)
            loss = (loss * loss_mask).sum(-1) / loss_mask_sum
            loss = loss.mean()
    if neox_args.memory_profiling:
        torch.cuda.nvtx.range_pop()
    if return_logits:
        return loss, outputs, metrics
    return loss, metrics


def get_model(neox_args, use_cache=False):
    """Build the model."""

    # Build model on cpu.
    print_rank_0("building GPT2 model ...")

    # Temporarily disable mup so that the base model does not use the mup init functions before set_base_shapes is called below.
    # If mup isn't being used anyways, this has no effect.
    old_use_mup = neox_args.use_mup
    neox_args.use_mup = False

    if neox_args.zero_stage in [2, 3]:
        if neox_args.pipe_parallel_size == 1:
            print_rank_0(
                "ZeRO stage 2/3 and the PipelineModule are incompatible, please set 'pipe_parallel_size' to 0 instead"
            )
            exit()
        if neox_args.pipe_parallel_size > 1:
            print_rank_0(
                "ZeRO stage 2/3 and pipeline paralleism are not supported simultaneously"
            )
            exit()
        if neox_args.model_parallel_size > 1:
            print_rank_0(
                "ZeRO stage 2/3 and model paralleism are not currently supported simultaneously"
            )
            exit()

    with deepspeed.zero.Init(
        config_dict_or_path=neox_args.deepspeed_config
    ) if neox_args.zero_stage == 3 else nullcontext() as gs:
        model = GPT2ModelPipe(
            neox_args=neox_args,
            num_tokentypes=0,
            parallel_output=True if neox_args.train_impl != "rm" else False,
            topology=mpu.get_topology(),
            use_cache=use_cache,
        )

    ### soft prompt tuning stuff ###
    if neox_args.soft_prompt_tuning is not None and neox_args.soft_prompt_tuning.get(
        "enabled", False
    ):
        soft_prompt = SoftEmbedding(
            neox_args,
            wte=getattr(model, "0").word_embeddings,
            n_tokens=neox_args.soft_prompt_tuning.get("n_tokens", 10),
            init_string=neox_args.soft_prompt_tuning.get("init_string", ""),
            init_range=neox_args.soft_prompt_tuning.get("init_range", 0.5),
        )
        model.insert_layers(
            layers=soft_prompt, idx=1
        )  # insert the soft prompt layer directly after the word embeddings

        # freeze everything but the soft prompt
        for name, param in model.named_parameters():
            if not "soft_embedding" in name:
                param.requires_grad = False

    if not neox_args.is_pipe_parallel:
        # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
        model = model.to_sequential()

    neox_args.use_mup = old_use_mup

    if neox_args.use_mup:
        try:
            import mup
        except ModuleNotFoundError:
            print("Please install mup https://github.com/microsoft/mup")
            raise Exception

        base_shapes = f"{neox_args.base_shapes_file}.{torch.distributed.get_rank()}"

        if neox_args.save_base_shapes:
            save_base_shapes(neox_args, base_shapes, use_cache)

        mup.set_base_shapes(model, base_shapes)

        # Call the mup replacement init functions on the model now that set_base_shapes has given each weight a .infshape attribute
        mup_weights_reinit(neox_args, model)

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_optimizer(model, neox_args, dummy=False):
    """Set up the optimizer."""
    if neox_args.no_load_optim and neox_args.deepspeed:
        # Required to have something so...
        dummy = True
        neox_args.optimizer = {"params": {"lr": 0.0}}
        neox_args.optimizer_type = "adam"
    elif neox_args.no_load_optim:
        return None, None

    if neox_args.optimizer is None:
        print_rank_0(
            f"ERROR: Optimizer is None. Either set the optimizer dict in your config (if training) or set no_load_optim in your config (if inference)"
        )
        exit()
    # Build parameter groups (weight decay and non-decay).
    param_groups = get_params_for_weight_decay_optimization(model, neox_args)
    print_rank_0(
        f'Configuring Optimizer type: {neox_args.optimizer_type} with params: {neox_args.optimizer["params"]}'
    )

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group["params"]:
            if not hasattr(param, "model_parallel"):
                param.model_parallel = False

    # Filter out params that don't require a grad (for soft prompt tuning, etc.)
    _param_groups = []
    for param_group in param_groups:
        trainable_params = [p for p in param_group["params"] if p.requires_grad]
        if dummy:
            trainable_params = [trainable_params[0]]  # just take the first one
        param_group["params"] = trainable_params
        _param_groups.append(param_group)
        if dummy:
            # Only need one.
            break
    param_groups = _param_groups

    # If we're using mup, then the optimizer must be adam or sgd
    assert not neox_args.use_mup or (
        neox_args.optimizer_type.lower() == "adam"
        or neox_args.optimizer_type.lower() == "sgd"
    ), f"If use_mup == True, you must specify either the adam or sgd optimizers. You passed: {neox_args.optimizer_type.lower()}"

    if neox_args.optimizer_type.lower() in ["cpu_adam", "cpu_torch_adam"]:
        if neox_args.optimizer == "cpu_torch_adam":
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "onebitadam":
        assert neox_args.deepspeed
        optimizer = None
        # onebitadam needs to be instantiated within the deepspeed engine to work :|
    elif neox_args.optimizer_type.lower() == "sm3":
        from .optimizers import SM3

        optimizer = SM3(param_groups, **neox_args.optimizer["params"])
    elif neox_args.optimizer_type.lower() == "madgrad_wd":
        from .optimizers import madgrad_wd

        optimizer = madgrad_wd(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "lion":
        # if we want the deepspeed zero lion...megatron lion will throw DeepSpeed Error
        if neox_args.zero_optimization["stage"] != 0:
            from deepspeed.ops.lion import FusedLion

            lion_optimizer = FusedLion
        # if not zero
        else:
            from .optimizers import Lion

            lion_optimizer = Lion

        optimizer = lion_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "adam":
        # Use Adam
        if neox_args.use_mup:
            try:
                from mup import MuAdam

                adam_optimizer = MuAdam
            except ModuleNotFoundError:
                print("Please install mup https://github.com/microsoft/mup")
                raise Exception
        else:
            if neox_args.use_bnb_optimizer:
                try:
                    import bitsandbytes as bnb

                    adam_optimizer = bnb.optim.Adam8bit
                except ModuleNotFoundError:
                    print(
                        "Please install bitsandbytes following https://github.com/facebookresearch/bitsandbytes."
                    )
                    raise Exception
            else:
                try:
                    # default to apex as it's slightly faster
                    from apex.optimizers import FusedAdam as Adam
                except ImportError:
                    # if apex isn't installed, use deepspeed's FusedAdam
                    print(
                        "WARNING: APEX not installed - defaulting to deepspeed's fused adam"
                    )
                    from deepspeed.ops.adam import FusedAdam as Adam
                adam_optimizer = Adam
        optimizer = adam_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "sgd":
        try:
            from mup import MuSGD
        except ModuleNotFoundError:
            print("Please install mup https://github.com/microsoft/mup")
            raise Exception
        optimizer = MuSGD(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    else:
        raise ValueError(f"Optimizer type {neox_args.optimizer_type} not recognized")

    if neox_args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer, param_groups
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_learning_rate_scheduler(optimizer, neox_args):
    """Build the learning rate scheduler."""
    if (neox_args.no_load_optim) and not neox_args.deepspeed:
        # TODO: this should be configured as a separate arg
        return None
    if neox_args.deepspeed and neox_args.optimizer_type.lower() == "onebitadam":
        print_rank_0(
            "WARNING: onebitadam requires the lr scheduler be built by deepspeed - "
            "Make sure one is added to your deepspeed config"
        )
        return None

    # Add linear learning rate scheduler.
    if neox_args.lr_decay_iters is not None:
        num_iters = neox_args.lr_decay_iters
    elif neox_args.lr_decay_fraction is not None:
        num_iters = math.floor(neox_args.train_iters * neox_args.lr_decay_fraction)
    else:
        num_iters = neox_args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = neox_args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=neox_args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=neox_args.lr_decay_style,
        last_iter=init_step,
        min_lr=neox_args.min_lr,
        use_checkpoint_lr_scheduler=neox_args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=neox_args.override_lr_scheduler,
        use_mup=neox_args.use_mup,
    )

    return lr_scheduler


def setup_model_and_optimizer(neox_args, use_cache=False, iteration=None):
    """Setup memory profiler"""
    if neox_args.memory_profiling:
        torch.cuda.memory._record_memory_history(
            True,
            # keep a maximum 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            trace_alloc_record_context=True,
        )

    """Setup model and optimizer."""
    needs_reference_model = (
        (
            (neox_args.train_impl == "dpo")
            and (neox_args.precompute_model_name is None)
            and (not neox_args.dpo_reference_free)
        )
        or (
            (neox_args.train_impl == "kto")
            and (neox_args.precompute_model_name is None)
        )
        or ((neox_args.train_impl == "reinforce") and (neox_args.kl_div_beta > 0.0))
    )
    model = get_model(neox_args=neox_args, use_cache=use_cache)
    if needs_reference_model:
        reference_model = get_model(neox_args=neox_args, use_cache=use_cache)
    else:
        reference_model = None
    optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
    lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)
    if neox_args.deepspeed and needs_reference_model:
        # Need an optimizer & lr_scheduler so make a very small one to keep deepspeed happy...
        ref_optimizer, ref_param_groups = get_optimizer(
            model=reference_model, neox_args=neox_args, dummy=True
        )
        ref_lr_scheduler = get_learning_rate_scheduler(
            optimizer=ref_optimizer, neox_args=neox_args
        )
    else:
        ref_optimizer, ref_param_groups, ref_lr_scheduler = None, None, None
    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        _model_params = param_groups if optimizer is None else None
        _lr_scheduler = lr_scheduler

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=neox_args,
            lr_scheduler=_lr_scheduler,
            dist_init_required=False,
            model_parameters=_model_params,
            # Need to remove the below so that it doesn't conflict with --deepspeed_config required by autotuning
            # config_params=neox_args.deepspeed_config,
            mpu=mpu if not neox_args.is_pipe_parallel else None,
        )
        if needs_reference_model:
            reference_model, _, _, _ = deepspeed.initialize(
                model=reference_model,
                optimizer=ref_optimizer,
                args=neox_args,
                lr_scheduler=ref_lr_scheduler,
                dist_init_required=False,
                model_parameters=ref_param_groups,
                mpu=mpu if not neox_args.is_pipe_parallel else None,
            )
        mark_norms_for_sequence_parallel_grad_sync(model, neox_args)
        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        if neox_args.is_pipe_parallel:
            model.set_has_attention_mask(True)
            if neox_args.curriculum_learning:
                curr_scheduler = CurriculumScheduler(neox_args.curriculum_learning)
                if iteration is not None and iteration > 0:
                    curr_scheduler.update_difficulty(iteration)
            else:
                curr_scheduler = None
            model.set_batch_fn(
                partial(
                    get_batch_pipe, neox_args=neox_args, curr_scheduler=curr_scheduler
                )
            )
        else:
            model.module.set_batch_fn(
                partial(get_batch_sequential, neox_args=neox_args)
            )

    else:
        raise ValueError("Must be using deepspeed to run neox")

    if neox_args.load is not None:
        neox_args.iteration = load_checkpoint(
            neox_args=neox_args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            iteration=iteration,
        )
        if needs_reference_model:
            _ = load_checkpoint(
                neox_args=neox_args,
                model=reference_model,
                optimizer=ref_optimizer,
                lr_scheduler=ref_lr_scheduler,
                iteration=iteration,
            )
            reference_model.eval()
        print_rank_0(
            f"Loading checkpoint and starting from iteration {neox_args.iteration}"
        )
    else:
        neox_args.iteration = 0

    # need this for correct lr scheduling resume from ckpt
    # but it will not exist if this is being called for inference
    if lr_scheduler is not None:
        lr_scheduler.optimizer = model.optimizer

    return model, optimizer, lr_scheduler, reference_model


def backward_step(neox_args, timers, optimizer, model, loss):
    """Backward step."""

    # Backward pass.
    timers("backward-backward").start()
    if neox_args.deepspeed:
        model.backward(loss)
    else:
        raise ValueError("Must be using deepspeed to run neox")
    timers("backward-backward").stop()

    if neox_args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers("backward-allreduce").reset()
    else:
        raise ValueError("Must be using deepspeed to run neox")


def train_step(
    neox_args,
    timers,
    data_iterator,
    model,
    optimizer,
    lr_scheduler,
    reference_model=None,
):
    """Single training step."""
    # Pipeline parallelism schedules forward/backward/step
    if neox_args.is_pipe_parallel:
        reduced_loss = train_step_pipe(
            neox_args=neox_args, timers=timers, model=model, data_iterator=data_iterator
        )
        reduce_metrics = reduced_loss
        if (
            neox_args.memory_profiling
            and neox_args.iteration >= neox_args.profile_step_start
            and neox_args.iteration <= neox_args.profile_step_stop
            and torch.distributed.get_rank() == 0
        ):
            save_snapshot(neox_args)
    else:
        losses = []
        metric_dicts = defaultdict(list)
        for _ in range(neox_args.gradient_accumulation_steps):
            # Forward model for one step.
            timers("forward").start()
            loss, metric_dict = forward_step(
                neox_args=neox_args,
                timers=timers,
                data_iterator=data_iterator,
                model=model,
                is_train=True,
                reference_model=reference_model,
            )
            timers("forward").stop()
            losses.append(loss)
            for key in metric_dict.keys():
                metric_dicts[key].append(metric_dict[key])
            # Calculate gradients, reduce across processes, and clip.
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
            ):
                torch.cuda.nvtx.range_push(f"Backward pass")
            timers("backward").start()
            backward_step(
                neox_args=neox_args,
                timers=timers,
                optimizer=optimizer,
                model=model,
                loss=loss,
            )
            timers("backward").stop()
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
            ):
                torch.cuda.nvtx.range_pop()
            # Update parameters.
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
            ):
                torch.cuda.nvtx.range_push(f"Optimizer step")

            timers("optimizer").start()
            if neox_args.deepspeed:
                model.step()
            else:
                raise ValueError("Must be using deepspeed to run neox")
            timers("optimizer").stop()
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
            ):
                torch.cuda.nvtx.range_pop()
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
                and torch.distributed.get_rank() == 0
            ):
                save_snapshot(neox_args)
        # reduces metrics across machines for logging
        reduce_metrics = {
            key: reduce_losses(metric_dicts[key]).mean() for key in metric_dicts.keys()
        }
        reduce_metrics["lm_loss"] = reduce_losses(losses).mean()

    if neox_args.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    collect_loss_for_unit_test(reduce_metrics["lm_loss"])
    return reduce_metrics, skipped_iter


def train_step_pipe(neox_args, timers, model, data_iterator):
    """Single training step with DeepSpeed's pipeline parallel engine."""

    assert neox_args.deepspeed
    loss = model.train_batch(data_iter=data_iterator)
    loss_dict = {"lm_loss": loss}
    # Don't break Megatron's timers because we changed code paths.
    for t in [
        "forward",
        "backward",
        "allreduce",
        "optimizer",
        "batch generator",
        "data loader",
    ]:
        timers(t).reset()
    return loss_dict


def is_save_iter(neox_args, iteration):
    if neox_args.extra_save_iters and iteration in neox_args.extra_save_iters:
        return True

    if neox_args.checkpoint_factor:
        if neox_args.checkpoint_scale == "linear":
            assert float(
                neox_args.checkpoint_factor
            ).is_integer(), "checkpoint_factor must be a whole number when using linear checkpoint_scale"
            return iteration % neox_args.checkpoint_factor == 0
        elif neox_args.checkpoint_scale == "log":
            # Check if iteration is a power of checkpoint_factor
            assert neox_args.checkpoint_factor > 1
            power = 1
            while power < iteration + 1:
                if int(power) == iteration:
                    return True
                power *= neox_args.checkpoint_factor
            return False

    return False


def train(
    neox_args,
    timers,
    model,
    reference_model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
):
    """Train the model function."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    timers("interval time").start()
    report_memory_flag = True

    # get noise scale logger (if neox_args.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(neox_args)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)

    if neox_args.profile:
        schedule = torch.profiler.schedule(
            wait=neox_args.profile_step_start,
            warmup=1,
            active=neox_args.profile_step_stop - neox_args.profile_step_start,
        )
        prof = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                neox_args.tensorboard_dir
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
        )
        prof.start()
    while iteration < neox_args.train_iters:
        if neox_args.profile:
            prof.step()
        if neox_args.profile and iteration == neox_args.profile_step_start:
            torch.cuda.cudart().cudaProfilerStart()
        loss_dict, skipped_iter = train_step(
            neox_args=neox_args,
            timers=timers,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            reference_model=reference_model,
        )
        if neox_args.profile and iteration == neox_args.profile_step_stop:
            torch.cuda.cudart().cudaProfilerStop()
            prof.stop()
        iteration += 1
        neox_args.iteration = iteration
        if neox_args.precision == "fp16":
            overflow_monitor.check(skipped_iter)  # check for repeated overflow
        if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # Logging.
        report_memory_flag = training_log(
            neox_args=neox_args,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=lr,
            iteration=iteration,
            loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger,
        )

        # Checkpointing
        if neox_args.save and is_save_iter(neox_args, iteration):
            save_checkpoint(
                neox_args=neox_args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
        # Evaluation
        if (
            neox_args.eval_interval
            and iteration % neox_args.eval_interval == 0
            and neox_args.do_valid
        ):
            prefix = "iteration {}".format(iteration)
            evaluate_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterator=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=False,
                timers=timers,
                reference_model=reference_model,
            )

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rank = torch.distributed.get_rank()
            print_rank_0(
                "rank: {} | time: {} | exiting the program at iteration {}".format(
                    rank, time_str, iteration
                )
            )
            sys.exit()

    return iteration


def evaluate(
    neox_args,
    forward_step_fn,
    data_iterator,
    model,
    verbose=False,
    timers=None,
    reference_model=None,
):
    """Evaluation.
    neox_args: NeoX Arguments
    forward_step_fn: function with args `neox_args, timers,
                    data_iterator & model that will run a forward pass on the model
    data_iterator: Iterator that iterates over batches of data. Should return data in the form:
                    {'text': np.array([tokens], dtype=np.int64)}
                    where the size of the array is the model's context size + 1
                    (`get_batch` transforms it into inputs / labels)
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    losses = []
    metric_dicts = defaultdict(list)
    if neox_args.char_level_ppl:
        data_iterator = CharCounter(data_iterator, neox_args.tokenizer)

    with torch.no_grad():
        iteration = 0
        while iteration < neox_args.eval_iters:
            iteration += 1
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0(
                    "Evaluating iter {}/{}".format(iteration, neox_args.eval_iters)
                )

            # although we're not accumulating gradients here, we count one iter as train_batch_size_per_gpu * g.a.s
            # to be consistent with deepspeed's pipe parallel engine
            # since pipe parallel already takes gradient_accumulation_steps into account - default to 1 here if pipe parallel is true
            for _ in range(
                1
                if neox_args.is_pipe_parallel
                else neox_args.gradient_accumulation_steps
            ):
                # Forward evaluation
                loss, metric_dict = forward_step_fn(
                    model=model,
                    data_iterator=data_iterator,
                    neox_args=neox_args,
                    timers=timers,
                    reference_model=reference_model,
                )
                losses.append(loss)
                for key in metric_dict.keys():
                    metric_dicts[key].append(metric_dict[key])
            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

    # reduces losses across processes for logging & run eval harness tasks
    eval_results = {"lm_loss": reduce_losses(losses).mean().item()}
    for key in metric_dicts.keys():
        eval_results[key] = reduce_losses(metric_dicts[key]).mean().item()
    eval_results["lm_loss_ppl"] = math.exp(eval_results["lm_loss"])

    if neox_args.char_level_ppl:
        # calculate character level perplexity, if specified
        # if neox_args.char_level_ppl:
        # unwrap the data_iterator
        tokens_per_char = data_iterator.tokens_per_char()
        print_rank_0(f"Counting chars took {data_iterator.total_time} seconds")

        data_iterator = data_iterator.data_iterator
        eval_results["lm_loss_char_lvl_ppl"] = math.exp(
            eval_results["lm_loss"] * tokens_per_char
        )

    if neox_args.eval_tasks:
        from eval_tasks import run_eval_harness

        eval_results.update(
            run_eval_harness(
                model, forward_step_fn, neox_args, eval_tasks=neox_args.eval_tasks
            ).get("results")
        )
    # Move model back to the train mode.
    model.train()
    return eval_results


def collect_loss_for_unit_test(lm_ss):
    # Logic moved to separate function to allow tracking in unit tests with unittest.mock.patch
    pass


def evaluate_and_print_results(
    neox_args,
    prefix,
    forward_step_func,
    data_iterator,
    model,
    iteration,
    verbose=False,
    timers=None,
    chart_name="validation",
    reference_model=None,
):
    """Helper function to evaluate and dump results on screen."""
    total_loss_dict = evaluate(
        neox_args=neox_args,
        forward_step_fn=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        verbose=verbose,
        timers=timers,
        reference_model=reference_model,
    )
    string = f" {chart_name} results at {prefix} | "
    for k, v in total_loss_dict.items():
        if isinstance(v, dict):
            if neox_args.eval_tasks and "results" in v:
                v = v["results"]
                print(v)
            for k2, v2 in v.items():
                k3 = "_".join([k, k2])
                string += f"{k3} value: {v2:.6E} | "
                tb_wandb_log(
                    f"{chart_name}/{k3}",
                    v2,
                    iteration,
                    use_wandb=neox_args.use_wandb,
                    tensorboard_writer=neox_args.tensorboard_writer,
                    comet_experiment=neox_args.comet_experiment,
                )
        else:
            string += f"{k} value: {v:.6E} | "
            tb_wandb_log(
                f"{chart_name}/{k}",
                v,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
                comet_experiment=neox_args.comet_experiment,
            )

    length = len(string) + 1
    print_rank_0("-" * length)
    print_rank_0(string)
    print_rank_0("-" * length)


def save_snapshot(neox_args):
    assert (
        neox_args.memory_profiling_path is not None
    ), "Must pass memory_profiling_path config arg to use profiling"
    snapshot = torch.cuda.memory._snapshot()
    snapshot_path = os.path.join(neox_args.memory_profiling_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    with open(os.path.join(snapshot_path, "mem_snapshot.pickle"), "wb") as f:
        dump(snapshot, f)
