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

"""Utilities for generating text."""

import copy
import json
import math
import os
import time
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F

from megatron import print_rank_0
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids, is_mp_rank_0
from megatron.data.indexed_dataset import make_builder, make_dataset
from megatron.mpu.mappings import gather_from_model_parallel_region


def get_batch(neox_args, context_tokens: torch.Tensor):
    """
    Generate batch from context tokens. Attention mask and position ids are created. Returned tensors will be on CUDA.

    neox_args: NeoXArgs.
    context_tokens: torch tensor with dimensions [batch, context_size]

    returns: tuple of torch tensors (tokens, attention_mask, position_ids) on CUDA
    """

    # Move to GPU.
    tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )
    return tokens, attention_mask, position_ids


def pad_batch(
    context_tokens: List[List[int]], pad_id: int, pad_len: int, truncate: bool = False
):
    """
    pads context lengths in context_tokens with pad_id to equal neox_args.seq_length,
    and returns the padded batch and the new lengths.

    context_tokens: list of lists of tokens
    pad_id: int, integer to use as padding token
    pad_len: int, context length to be padded; all batch items will be padded to the same length
    truncate: bool, if True, truncate context tokens to pad_len if they are longer than pad_len

    returns: tuple of padded context tokens and a list of unpadded token count
    """

    context_lengths = []
    for i, tokens in enumerate(context_tokens):
        context_length = len(tokens)
        if context_length < pad_len:
            tokens.extend([pad_id] * (pad_len - context_length))
        elif context_length > pad_len:
            if not truncate:
                raise ValueError("context_length is bigger than to be padded length")
            context_tokens[i] = tokens[:pad_len]
            context_length = pad_len
        context_lengths.append(context_length)
    return context_tokens, context_lengths


def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filters the logits using top_k / top_p, filling any filtered vocab items with filter_value (defaults to -inf).

    This function has been mostly taken from huggingface conversational ai code at
    https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    When both top_k and top_p are specified, tokens are first filtered according to top_k, renormalized, and then filtered according to top_p.

    logits: torch.Tensor -> logits of megatron model.
    top_k: integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p: float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    returns: (filtered) logits"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def switch(val1, val2, boolean):
    """
    replaces items in val1 with items in val2 where boolean = True
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_model(model, model_inputs, is_pipe_parallel=False) -> torch.Tensor:
    """
    Runs model.forward(model_inputs)

    We need to create a wrapper for this function because deepspeed pipe parallel modules operate differently to normal models.

    model: a Megatron model.
    model_inputs: tuple containing model args

    returns: torch.Tensor containing the logits of the model
    """
    # because someone at deepspeed decided pipeline modules couldn't use kwargs,
    # we need to forward a pipe model differently to a normal model
    if not is_pipe_parallel:
        return model.module(model_inputs)
    else:
        # we need to format inputs this way because:
        # a) deepspeed pipeline only accepts iterables
        # b) deepspeed pipeline *requires* that you pass in labels for the loss, it's not easy to get around this
        # so we wrap the inputs in an iterable, and pad them (because internally, we get labels as inputs[:, 1:] and inputs as inputs[:, :-1])
        model_inputs = iter([{"text": F.pad(model_inputs[0], pad=(0, 1))}])

        # set num microbatches to 1 at inference time
        micro_batches_before = model.micro_batches
        model.micro_batches = 1

        # deepspeed sends metadata across pipeline stages only once in the first step, then assumes it will stay
        # constant. In inference, the metadata of the tensors being sent across pipe stages may change, so we need to set
        # these two flags in order for deepspeed to send the metadata every step, otherwise torch.distributed hangs
        # silently. Fun stuff.
        model.first_output_send = True
        model.pipe_recv_buf = None

        loss, logits = model.eval_batch(model_inputs, return_logits=True)
        model.micro_batches = micro_batches_before
        return logits


def broadcast_terminate_signal(terminate_runs: int):
    """Send signal to all workers to terminate if we've finished the process"""
    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
    torch.distributed.broadcast(
        terminate_runs_tensor,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    return terminate_runs_tensor[0].item()


def stop_tokens_in_completion(stop_tokens, context_tokens, batch_index, current_index):
    if stop_tokens is None:
        return False
    res = []
    for token_group in stop_tokens:
        context = context_tokens[batch_index, : current_index + 1]
        context = context[-len(token_group) :]
        if context.shape[0] == token_group.shape[0]:
            res.append(all(token_group == context))
        else:
            res.append(False)
    return any(res)


def stream_tokens(
    neox_args,
    model,
    context_tokens: List[List[int]],
    eos_token_id: int = None,
    maximum_tokens: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
):
    """
    iterator producing text completions

    neox_args: NeoXArgs.
    model: a Megatron model.
    context_tokens: the prompt to complete; unpadded list of lists of tokens ids
    context_lengths: lengths of context tokens of dimension [batch]; the context length records for each bach item how many non-padded tokens are provided
    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    attention_mask: attention mask for megatron model.
    position_ids: position ids for positional encoding.
    maximum_tokens: maximum number of tokens to be generated; careful! if a batch input is provided maximum_tokens specifies the maximum number of forwards.
                    longer batch items get less generated tokens.
    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)
    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0
    yields: (
                tokens (completions from model),
                token_generation_start_index (token index per batch item for the first generated token),
                token_generation_end_index (token index per batch item for the last generated token),
                logits (logits which are so far computed, zeros otherwise),
                is_done (flag for each bach item indicating whether an eod token was generated)
            )

            * each iteration adds a generated token to the context_tokens
            * output contains both context_tokens from input and generated tokens
            * if batch items have different lengths, the iterator will start at the first completion and return the unchanged input context token otherwise
    """

    model.eval()

    # pad batch in order to allow conversion to tensor
    context_tokens, context_lengths = pad_batch(
        copy.deepcopy(context_tokens),
        pad_id=neox_args.tokenizer.eod,
        pad_len=neox_args.seq_length,
    )

    # convert to tensor and broadcast
    context_tokens = torch.cuda.LongTensor(context_tokens)
    if stop_tokens:
        if len(stop_tokens) > 0 and type(stop_tokens[0]) is not list:
            stop_tokens = [stop_tokens]
        for i in range(0, len(stop_tokens)):
            stop_tokens[i] = torch.cuda.LongTensor(stop_tokens[i])

    # Make sure context tokens + start tokens are the same across all ranks
    token_generation_start_index = torch.cuda.LongTensor(context_lengths)
    torch.distributed.broadcast(
        context_tokens,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    torch.distributed.broadcast(
        token_generation_start_index,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )

    # get attention mask / position ids
    context_tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens)

    # set variables
    eos_token_id = eos_token_id or neox_args.tokenizer.eod
    maximum_tokens = maximum_tokens or (
        neox_args.seq_length - token_generation_start_index.max().item() - 1
    )
    batch_size = context_tokens.size(0)

    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate
    last_token_index_to_generate = min(
        neox_args.seq_length
        - 1,  # never generate more than the model's sequence length
        token_index_to_generate + maximum_tokens - 1,
    )

    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().cuda()
        token_generation_end_index = torch.ones([batch_size]).long().cuda() * (-1)
        generation_logits = (
            torch.empty(maximum_tokens, neox_args.padded_vocab_size).float().cuda()
        )

        while token_index_to_generate <= last_token_index_to_generate:
            if recompute:  # recompute all tokens
                model_inputs = (
                    context_tokens,
                    position_ids,
                    attention_mask,
                )
                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = logits[
                        :, token_index_to_generate - 1, :
                    ]  # [bs, seq, vocab_size] -> [bs, vocab_size]
            else:  # use kv cache
                if token_index_to_generate == first_token_index_to_generate:
                    tokens_to_use = context_tokens[:, :token_index_to_generate]
                    positions_to_use = position_ids[:, :token_index_to_generate]
                else:
                    tokens_to_use = context_tokens[:, token_index_to_generate - 1].view(
                        batch_size, -1
                    )
                    positions_to_use = position_ids[
                        :, token_index_to_generate - 1
                    ].view(batch_size, -1)

                model_inputs = (
                    tokens_to_use,  # input_ids
                    positions_to_use,  # position_ids
                    attention_mask,  # attention_mask
                )

                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)

                # Extract the logits tensor based on the pipeline parallelism setting
                if logits is not None:
                    # If pipeline parallelism is on, `logits` may be a tuple, so we extract the first element if it's a tensor
                    logits_tensor = logits[0] if isinstance(logits, tuple) and isinstance(logits[0], torch.Tensor) else logits
                    batch_size = logits_tensor.shape[0]
                    # Get the logits for the last generated token across the batch
                    generated_token_logits = logits_tensor[:, -1].view(batch_size, -1).contiguous()  # [bs, seq, vocab_size] -> [bs, vocab_size]

            if logits is not None:
                # sample token id of the to be generated token
                if temperature == 0.0 and top_k == 0 and top_p == 0.0:
                    generated_tokens = torch.argmax(
                        generated_token_logits, dim=-1
                    ).view(-1)
                else:
                    generated_token_logits = generated_token_logits.float()
                    if temperature > 0.0:
                        generated_token_logits /= temperature
                    generated_token_logits = filter_logits(
                        generated_token_logits, top_k=top_k, top_p=top_p
                    )
                    next_token_log_probs = F.softmax(generated_token_logits, dim=-1)
                    generated_tokens = torch.multinomial(
                        next_token_log_probs, num_samples=1
                    ).view(-1)

                if neox_args.return_logits:
                    generation_logits[
                        token_index_to_generate - 1
                    ] = generated_token_logits[0]

            if neox_args.is_pipe_parallel:
                # broadcast generated tokens to pipe parallel group
                src_rank = model.grid.stage_to_global(model.num_stages - 1)
                generated_tokens = (
                    generated_tokens
                    if logits is not None
                    else torch.zeros(batch_size, dtype=torch.long).cuda()
                )
                torch.distributed.broadcast(
                    tensor=generated_tokens,
                    src=src_rank,
                    group=mpu.get_pipe_parallel_group(),
                )

            # determine if state has started for each batch item
            state_started = (
                token_generation_start_index <= token_index_to_generate
            )  # check which batch items have been started

            # switch out padding tokens for generated tokens
            context_tokens[:, token_index_to_generate] = switch(
                context_tokens[:, token_index_to_generate].view(-1),
                generated_tokens,
                state_started,
            )

            # determine if state has finished for each batch item
            state_done = (
                generated_tokens == eos_token_id
            ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
            state_just_finished = (state_done & ~state_is_done).bool()
            state_is_done = state_is_done | state_done
            stop_tokens_produced = torch.zeros_like(state_is_done)
            for batch_idx, ctx in enumerate(context_tokens):
                stop_tokens_produced[batch_idx] = stop_tokens_in_completion(
                    stop_tokens, context_tokens, batch_idx, token_index_to_generate
                )
            state_is_done = state_is_done | stop_tokens_produced

            token_generation_end_index[
                (state_started.byte() & ~state_is_done).bool()
            ] = token_index_to_generate

            token_index_to_generate += 1

            yield context_tokens, token_generation_start_index, token_generation_end_index, generation_logits, state_is_done.bool()
            if torch.all(state_is_done):
                break


def generate_samples_from_prompt(
    neox_args,
    model,
    text: Union[List[str], str],
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
):
    """
    Generates samples from raw text and returns them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model
    text: either a single prompt (str) or a list of prompts (List[str]).

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds

    """
    eos_token_id = eos_token_id or neox_args.tokenizer.eod

    # type check
    assert any(
        [isinstance(text, str), isinstance(text, list)]
    ), "Text should be in string or list form"
    if isinstance(text, str):
        text = [text]

    input_count = len(text)
    input_pos = 0

    # generate completions
    generated_texts = []
    while True:

        start_time = time.time()
        # Tokenize text, and check whether we should terminate process
        terminate_runs = 0
        if input_pos == input_count:
            terminate_runs = 1
        else:
            raw_text = text[input_pos]
            input_pos += 1

            if raw_text == "":
                context_tokens = [eos_token_id]
            else:
                context_tokens = neox_args.tokenizer.tokenize(raw_text)
            context_length = len(context_tokens)

            if context_length >= (neox_args.seq_length // 2):
                print_rank_0(
                    "\nWarning! Context length",
                    context_length,
                    "\nPlease give smaller context (e.g. half of the "
                    "max sequence length)!",
                )
        if not is_mp_rank_0():
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)
            terminate_runs = 0

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return generated_texts

        for (
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_generated_token_logits,
            is_done,
        ) in stream_tokens(
            neox_args=neox_args,
            model=model,
            context_tokens=[context_tokens],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=stop_tokens,
        ):
            pass  # finish generation and use all results below

        batch_context_tokens = batch_context_tokens.cpu().numpy().tolist()
        batch_token_generation_start_index = (
            batch_token_generation_start_index.cpu().numpy().tolist()
        )
        batch_token_generation_end_index = (
            batch_token_generation_end_index.cpu().numpy().tolist()
        )
        batch_is_done = is_done.cpu().numpy().tolist()

        for tokens, start_index, end_index, is_done in zip(
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_is_done,
        ):

            if end_index >= start_index:
                generated_tokens = tokens[start_index : end_index + 1]
                try:
                    generated_text = neox_args.tokenizer.detokenize(generated_tokens)
                    message = None
                except KeyError:
                    generated_text = None
                    message = "WARNING: generated token which doesn't exist."
            else:
                generated_text = None
                generated_tokens = []
                # this will happen if the first generated token is a stop token or eos token
                message = "WARNING: text generation did not start; try different batching or adjust parameters"
            if is_mp_rank_0():
                data = {
                    "context": raw_text,
                    "text": generated_text,
                    "length": len(generated_tokens),
                    "finished": is_done,
                    "message": message,
                    "duration_seconds": float(time.time() - start_time),
                }

                if neox_args.return_logits:
                    data["logits"] = batch_generated_token_logits.cpu().numpy().tolist()

                generated_texts.append(data)

    return generated_texts


def generate_samples_input_from_file(
    neox_args,
    model,
    input_file,
    output_file=None,
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    prompt_end: str = "\n",
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    """
    Generates samples from an input file and writes them to an output file.

    Reads prompts from neox_args.sample_input_file and writes completions to neox_args.sample_output_file

    neox_args: NeoXArgs.
    model: a Megatron model

    input_file: path to input file. Each line in the input file will be treated as separate prompt. The line break at the end of the line is not included in the prompt.
    output_file: file where generation results are to be stored in jsonl format. defaults to input_file+'.output.jsonl' if not defined

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated
    prompt_end: end of a single input prompt. Defaults to newline character '\n'. Other prompt-end sequences may be useful when generating indent-aware completions (e.g. code)

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0


    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """
    # Read the sample file
    print_rank_0(
        "generate_samples_input_from_file() loading input from {}".format(input_file)
    )
    with open(input_file, "r", encoding="utf-8") as f:
        prompts = f.read()
        prompts = prompts.split(prompt_end)
    prompts = [p.strip() for p in prompts]
    prompts = [p for p in prompts if len(p) > 0]
    print_rank_0(
        "generate_samples_input_from_file() prompts loaded: {}".format(len(prompts))
    )

    if is_mp_rank_0():
        if output_file is None:
            output_file = str(input_file) + ".output.jsonl"
            print_rank_0(
                "generate_samples_input_from_file() setting default output file to {}".format(
                    output_file
                )
            )

    print_rank_0("generate_samples_input_from_file() generating...")
    generated_texts = generate_samples_from_prompt(
        neox_args=neox_args,
        model=model,
        text=prompts,
        eos_token_id=eos_token_id,
        maximum_tokens=maximum_tokens,
        recompute=recompute,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    if is_mp_rank_0():
        with open(output_file, "w") as f_out:
            for item in generated_texts:
                f_out.write(json.dumps(item) + "\n")
    print_rank_0("generate_samples_input_from_file() done")
    return generated_texts


def generate_samples_unconditional(
    neox_args,
    model,
    number_of_samples: int = 10,
    output_file=None,
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model

    number_of_samples (default 10): number of unconditional samples to be generated

    output_file: file where generation results are to be stored in jsonl format. no file will be stored if omitted

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated
    prompt_end: end of a single input prompt. Defaults to newline character '\n'. Other prompt-end sequences may be useful when generating indent-aware completions (e.g. code). The interactive mode will reroll the user-input request until the stop-char is met

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: dict containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """

    print_rank_0("generate_samples_unconditional() generating...")
    assert number_of_samples > 0, "number_of_samples must be > 0"
    generated_texts = generate_samples_from_prompt(
        neox_args=neox_args,
        model=model,
        text=["" for _ in range(number_of_samples)],
        eos_token_id=eos_token_id,
        maximum_tokens=maximum_tokens,
        recompute=recompute,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    if is_mp_rank_0():
        if output_file is not None:
            with open(output_file, "w") as f_out:
                for item in generated_texts:
                    f_out.write(json.dumps(item) + "\n")
    print_rank_0("generate_samples_unconditional() done")
    return generated_texts


def generate_samples_interactive(
    neox_args,
    model,
    maximum_tokens: int = 64,
    prompt_end: str = "\n",
    eos_token_id: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model

    maximum_tokens: maximum number of tokens to be generated
    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: dict containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """

    while True:
        model.module.clear_cache()  # clear kv cache between batches
        torch.distributed.barrier(group=mpu.get_model_parallel_group())
        terminate_runs = 0

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            os.system("clear")
            raw_text = ""
            while True:
                current_input = input("Context prompt >>> ")
                if (
                    prompt_end == "\n"
                ):  # we need to handle '\n' case as 'input' strips it and leads to lines being squashed
                    raw_text += current_input
                    break
                if prompt_end in current_input:
                    raw_text += current_input.split(prompt_end)[0]
                    break
                raw_text += (
                    current_input + "\n"
                )  # re-add newline since we stripped it on input
            context_tokens = neox_args.tokenizer.tokenize(raw_text)
            if len(context_tokens) == 0:
                context_tokens = [neox_args.tokenizer.eod]
            context_length = len(context_tokens)
            if context_length >= (neox_args.seq_length - 1):
                print_rank_0(
                    "\nContext length"
                    + str(context_length)
                    + "\nReached max sequence length!"
                )
                terminate_runs = 1
        else:
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return
        for (
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_generated_token_logits,
            is_done,
        ) in stream_tokens(
            neox_args=neox_args,
            model=model,
            context_tokens=[context_tokens],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ):
            if mpu.get_model_parallel_rank() == 0:
                generated_tokens = (
                    batch_context_tokens[0]
                    .cpu()
                    .numpy()
                    .tolist()[
                        batch_token_generation_start_index[0]
                        .item() : batch_token_generation_end_index[0]
                        .item()
                        + 1
                    ]
                )
                generated_text = neox_args.tokenizer.detokenize(generated_tokens)
                print_rank_0("Generated Text: " + generated_text)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            _ = input("\n<press enter to continue>")


def get_logp(logits, labels, force_fp32=False):
    if force_fp32:
        logits = logits.float()
    logp = logits.log_softmax(dim=-1)
    return torch.gather(logp, dim=2, index=labels.unsqueeze(2)).squeeze(2)


def precompute_logits(neox_args, model):
    """
    Precomputes logprobs from training/testing/validation datasets

    Saves it to the same directory as the dataset with the model name appended to it

    neox_args: NeoXArgs.
    model: a Megatron model

    """
    if neox_args.precompute_model_name is None:
        mdl_name = str(hash(neox_args.load))
    else:
        mdl_name = neox_args.precompute_model_name
    print_rank_0("Precomputing logprobs...")
    model.eval()
    data_paths = list()
    if neox_args.train_data_paths is not None:
        for path in neox_args.train_data_paths:
            data_paths.append(path)
        for path in neox_args.test_data_paths:
            data_paths.append(path)
        for path in neox_args.valid_data_paths:
            data_paths.append(path)
    elif neox_args.pos_train_data_paths is not None:
        # Pairwise data...
        for path in neox_args.pos_train_data_paths:
            data_paths.append(path)
        for path in neox_args.neg_train_data_paths:
            data_paths.append(path)
        for path in neox_args.pos_valid_data_paths:
            data_paths.append(path)
        for path in neox_args.neg_valid_data_paths:
            data_paths.append(path)
        for path in neox_args.pos_test_data_paths:
            data_paths.append(path)
        for path in neox_args.neg_test_data_paths:
            data_paths.append(path)
    for path in data_paths:
        print_rank_0(f"Precomputing logits for {path}")
        # Add hash to path...
        out_path = path + f"_{mdl_name}"
        if os.path.exists(out_path + ".idx"):
            continue
        dataset = make_dataset(path, neox_args.data_impl, not neox_args.mmap_warmup)
        if is_mp_rank_0():
            out_dataset = make_builder(out_path + ".bin", neox_args.data_impl)
            out_dataset._dtype = np.float32
        i = 0

        # TODO: Not sure why this requires a multiple of 8? Investigate later.
        while i < int(math.ceil(len(dataset) / 8.0) * 8):
            start = time.time()
            model.module.clear_cache()  # clear kv cache between batches
            if is_mp_rank_0():
                offset = (
                    mpu.get_data_parallel_rank()
                    * neox_args.train_micro_batch_size_per_gpu
                )
                context_tokens = [
                    [int(x) for x in dataset.get(j % len(dataset)).tolist()]
                    for j in range(
                        i + offset,
                        i + (neox_args.train_micro_batch_size_per_gpu + offset),
                    )
                ]
                # grab microbatch
                # pad batch in order to allow conversion to tensor
                context_tokens, context_lengths = pad_batch(
                    copy.deepcopy(context_tokens),
                    pad_id=0,
                    pad_len=neox_args.seq_length + 1,
                    truncate=True,
                )
                # print(context_tokens)
                label_tokens = [tokens[1:] for tokens in context_tokens]
                context_tokens = [tokens[:-1] for tokens in context_tokens]
            else:
                context_tokens = [
                    [0 for _ in range(neox_args.seq_length)]
                    for _ in range(neox_args.batch_size)
                ]
                label_tokens = [
                    [0 for _ in range(neox_args.seq_length)]
                    for _ in range(neox_args.batch_size)
                ]
                context_lengths = [0 for _ in range(neox_args.batch_size)]
            i += (
                neox_args.train_micro_batch_size_per_gpu
                * mpu.get_data_parallel_world_size()
            )
            # print(context_tokens)
            # convert to tensor and broadcast
            context_tokens = torch.cuda.LongTensor(context_tokens)
            label_tokens = torch.cuda.LongTensor(label_tokens)
            # Make sure context tokens + start tokens are the same across all ranks
            token_generation_start_index = torch.cuda.LongTensor(context_lengths)
            torch.distributed.broadcast(
                context_tokens,
                mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group(),
            )
            torch.distributed.broadcast(
                token_generation_start_index,
                mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group(),
            )
            torch.distributed.broadcast(
                label_tokens,
                mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group(),
            )
            # context_tokens = context_tokens[:, :chop_len].contiguous()
            # label_tokens = label_tokens[:, :chop_len].contiguous()
            with torch.no_grad():
                # get attention mask / position ids
                context_tokens, attention_mask, position_ids = get_batch(
                    neox_args, context_tokens
                )
                model_inputs = (
                    context_tokens,
                    position_ids,
                    attention_mask,
                )
                maybe_tuple = forward_model(
                    model, model_inputs, neox_args.is_pipe_parallel
                )
                if isinstance(maybe_tuple, tuple):
                    logits, _ = maybe_tuple
                else:
                    logits = maybe_tuple
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    logits = gather_from_model_parallel_region(logits)
                    logp = get_logp(logits, label_tokens, True).squeeze()
                if neox_args.is_pipe_parallel:
                    # broadcast generated tokens to pipe parallel group
                    src_rank = model.grid.stage_to_global(model.num_stages - 1)
                    logp = (
                        logp
                        if logits is not None
                        else torch.zeros(
                            neox_args.batch_size, dtype=torch.float32
                        ).cuda()
                    )
                    torch.distributed.broadcast(
                        tensor=logp,
                        src=src_rank,
                        group=mpu.get_pipe_parallel_group(),
                    )
                    logp = logp.squeeze()
                logp_list = [
                    torch.zeros_like(logp)
                    for _ in range(mpu.get_data_parallel_world_size())
                ]
                torch.distributed.all_gather(
                    logp_list, logp, group=mpu.get_data_parallel_group()
                )
                logp = torch.cat(logp_list, dim=0).cpu().numpy()
                if (mpu.get_model_parallel_rank() == 0) and (
                    mpu.get_data_parallel_rank() == 0
                ):
                    for j in range(logp.shape[0]):
                        out_dataset.add_item(logp[j])
                        out_dataset.end_document()
            print_rank_0(f"Processed {i} / {len(dataset)} in {time.time() - start}")
        if is_mp_rank_0():
            out_dataset.finalize(
                out_path + ".idx",
            )
        torch.distributed.barrier()
