import argparse
import json
import random
from collections import defaultdict
import os
import deepspeed
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange
import wandb
import socket
from wandb import UsageError

from gpt_neox import (GPTNeoX, AutoregressiveWrapper, TextSamplerDataset,
                      cycle, prepare_optimizer_parameters, decode_tokens, prepare_data,
                      GPTNeoX_Pipe)
from gpt_neox.datasets import GPT2Dataset
from gpt_neox.data_utils import get_tokenizer

from gpt_neox.utils import is_main, get_args, get_params, save_ds_checkpoint, load_ds_checkpoint

import gpt_neox

WORLD_SIZE = os.getenv('WORLD_SIZE')

def loss_function(x, y):
    losses = torch.nn.functional.cross_entropy(x, y, reduction='none')
    loss = losses.mean()
    return loss

def configure_checkpointing(model_engine):
    deepspeed.checkpointing.configure(model_engine.mpu, deepspeed_config=train_args.deepspeed_config)
    model_engine.mpu.checkpoint = deepspeed.checkpointing.checkpoint
    model_engine.mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    model_engine.mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed
    assert deepspeed.checkpointing.is_configured()

def prepare_dataset(dset_params, train_args):
    torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
    if is_main(train_args):
        prepare_data(dset_params["name"])
        torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
    else:
        torch.distributed.barrier()

if __name__ == '__main__':
    # arguments
    train_args = get_args()

    IS_MAIN = is_main(train_args)

    params = get_params(train_args.model)
    deepspeed.init_distributed(dist_backend='nccl')

    # tokenizer
    tokenizer = get_tokenizer(tokenizer_type=params["tokenizer"].get("type", None),
                            from_pretrained=params["tokenizer"].get("from_pretrained", True),
                            add_padding_token=params["tokenizer"].get("add_padding_token", False))
    vocab_size = len(tokenizer) if params["vocab_size"] is None else params["vocab_size"]

    # model
    model = GPTNeoX_Pipe(
        num_tokens=vocab_size,
        dim=params["hidden_dim"],
        seq_len=params["seq_len"],
        depth=params["n_layers"],
        heads=params["n_heads"],
        dim_head=params["dim_head"],
        loss_fn = loss_function,
        num_stages = params.get("pipeline_num_stages", 2),
        activation_checkpoint_interval=params.get('activation_checkpoint_interval', 1)
    )

    ## Wandb
    # only display system stats from one worker per machine
    wandb_settings = wandb.Settings() if is_main(train_args) else wandb.Settings(_disable_stats=True)
    name = f'{socket.gethostname()}-{train_args.local_rank}' if train_args.group_name else None

    use_wandb = True
    try:
        wandb.init(project="neox_train_pipeline", group=train_args.group_name, name=name, save_code=True, force=False,
                   entity=params.get('wandb', {}).get('team'), settings=wandb_settings)
    except UsageError as e:
        use_wandb = False
        print(e)
        print('Skipping wandb. Execute `wandb login` on local machine to enable.')

    # prepare data
    dset_params = params["dataset"]
    prepare_dataset(dset_params, train_args)

    train_dataset = GPT2Dataset(glob_pattern=dset_params["train_path"],
                                seq_len=params["seq_len"],
                                train=True,
                                mode='with_labels',
                                **dset_params)

    eval_dataset = GPT2Dataset(glob_pattern=dset_params["eval_path"],
                            seq_len=params["seq_len"],
                            train=False,
                            mode='with_labels',
                            **dset_params)

    val_loader = DataLoader(eval_dataset, batch_size=params["eval_batch_size"])
    val_loader = cycle(val_loader)

    # optimizer
    ds_model_params = prepare_optimizer_parameters(model)
    optim = torch.optim.Adam(ds_model_params, lr=params["learning_rate"])
    # deepspeed loader

    model, optim, train_loader, lr_scheduler = deepspeed.initialize(args=train_args,
                                                                    model=model,
                                                                    optimizer=optim,
                                                                    model_parameters=ds_model_params,
                                                                    training_data=train_dataset)
    configure_checkpointing(model)

    if use_wandb:
        wandb.config.update(params)
        wandb.watch(model, log_freq=10, log=params.get('wandb', {}).get('watch_model'))

    current_iteration = load_ds_checkpoint(model, params, iteration=None)

    pbar = trange(current_iteration, params.get('train_steps', 100000), mininterval=10., desc='Training Model', dynamic_ncols=True)
    for i in pbar:
        loss = model.train_batch()
        pbar.set_description(f'Training Loss: {loss.item():.4f}')
        pbar.update()
        if not i % params.get('checkpoint_save_frequency', 1000) and i != 0:
            save_ds_checkpoint(i, model, params, params.get('keep_n_latest_checkpoints', 5), IS_MAIN)

        if use_wandb:
            wandb.log({'loss': loss.item()})
