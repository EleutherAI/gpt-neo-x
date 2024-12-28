"""
Helper functions for performing coord check.
"""
import os
import gc
from copy import copy
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import deepspeed
from megatron import print_rank_0
from megatron.training import train_step


def get_coord_data(
    neox_args,
    timers,
    models,
    dataloader,
    nsteps=10,
    nseeds=2,
):
    df = {
        "seed": [],
        "step": [],
        "word_embedding_act_abs_std": [],
        "attn_output_act_abs_std": [],
        "ffn_output_act_abs_std": [],
        "output_logits_act_abs_std": [],
        "width": [],
    }

    df_mode = "mup" if neox_args.use_mup else "sp"
    if neox_args.use_mup:
        print_rank_0("muP Coord Check for mu Parameterization")
    else:
        print_rank_0("muP Coord Check for standard Parameterization")

    _df = None
    df_path = os.path.join(neox_args.mup_save, f"df_{df_mode}.csv")
    if (neox_args.mup_save is not None) and os.path.exists(df_path):
        _df = pd.read_csv(df_path)

    with torch.no_grad():
        torch.cuda.empty_cache()

    for width, model_obj in models.items():
        for i in range(nseeds):
            seed = (i + 1) * 100000
            torch.manual_seed(seed)

            model, optimizer, lr_scheduler = model_obj()
            model.train()
            print_rank_0(f">>> muP Coord Check: Running Model with width: {width} on seed: {seed}")
            print_rank_0(f">>> muP Coord Check: mup_width_multiplier set to {model.neox_args.mup_width_multiplier}")
            for step in range(nsteps + 1):

                word_embedding_act_abs_std_list = []
                attn_output_act_abs_std_list = []
                ffn_output_act_abs_std_list = []
                output_logits_act_abs_std_list = []
                remove_hooks = []

                def word_embedding_coord_check_hook(module, input, output):
                    with torch.no_grad():
                        word_embedding_act_abs_std_list.append(
                            output.cpu().abs().std().item()
                        )

                def attn_output_coord_check_hook(module, input, output):
                    with torch.no_grad():
                        attn_output_act_abs_std_list.append(
                            output[0].cpu().abs().std().item()
                        )

                def ffn_output_coord_check_hook(module, input, output):
                    with torch.no_grad():
                        ffn_output_act_abs_std_list.append(
                            output[0].cpu().abs().std().item()
                        )

                def output_logits_coord_check_hook(module, input, output):
                    with torch.no_grad():
                        output_logits_act_abs_std_list.append(
                            output[0].cpu().abs().std().item()
                        )

                for name, module in model.named_modules():
                    if name.endswith(".word_embeddings"):
                        remove_hooks.append(
                            module.register_forward_hook(
                                word_embedding_coord_check_hook
                            )
                        )
                    elif name.endswith(".attention.dense"):
                        remove_hooks.append(
                            module.register_forward_hook(attn_output_coord_check_hook)
                        )
                    elif name.endswith(".mlp.dense_4h_to_h"):
                        remove_hooks.append(
                            module.register_forward_hook(ffn_output_coord_check_hook)
                        )
                    elif name.endswith(".final_linear"):
                        remove_hooks.append(
                            module.register_forward_hook(output_logits_coord_check_hook)
                        )

                # train for a step
                train_step(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=dataloader,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )

                word_embedding_act_abs_std = None
                attn_output_act_abs_std = None
                ffn_output_act_abs_std = None
                output_logits_act_abs_std = None

                # remove hooks
                for handle in remove_hooks:
                    handle.remove()
                word_embedding_act_abs_std = np.mean(word_embedding_act_abs_std_list)
                attn_output_act_abs_std = np.mean(attn_output_act_abs_std_list)
                ffn_output_act_abs_std = np.mean(ffn_output_act_abs_std_list)
                output_logits_act_abs_std = np.mean(output_logits_act_abs_std_list)

                df["seed"].append(i)
                df["step"].append(step)
                df["word_embedding_act_abs_std"].append(word_embedding_act_abs_std)
                df["attn_output_act_abs_std"].append(attn_output_act_abs_std)
                df["ffn_output_act_abs_std"].append(ffn_output_act_abs_std)
                df["output_logits_act_abs_std"].append(output_logits_act_abs_std)
                df["width"].append(width)

            def del_obj_attrs(obj):
                attributes = [
                    attr for attr in vars(obj) if not callable(getattr(obj, attr))
                ]
                for attr in attributes:
                    try:
                        delattr(obj, attr)
                    except:
                        pass

            def unlink_hp_params(lp_param_list):
                for lp in lp_param_list:
                    lp._hp_mapping = None
                return

            for i, _ in enumerate(optimizer.optimizer.param_groups):
                unlink_hp_params(optimizer.bit16_groups[i])
            del_obj_attrs(optimizer)
            model.destroy()
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            deepspeed.runtime.utils.empty_cache()

            temp_df = pd.DataFrame(df)
            temp_df.to_csv(os.path.join(neox_args.mup_save, f"df_{df_mode}.csv"), index=False)

    return pd.DataFrame(df)
