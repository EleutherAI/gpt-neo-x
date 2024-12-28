# Copyright (c) 2024, EleutherAI.
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

import sys
import os

import torch

try:
    import wandb
except ModuleNotFoundError:
    pass

from megatron import mpu, print_rank_0
from megatron.utils import report_memory
import math


'''
class Tee:
    """Duplicate output to both stdout/err and file"""

    def __init__(self, file, err: bool = False) -> None:
        self.file = open(file, "w")
        self.err = err
        if not err:
            self.std = sys.stdout
            sys.stdout = self
        else:
            self.std = sys.stderr
            sys.stderr = self

    def __del__(self) -> None:
        if not self.err:
            sys.stdout = self.std
        else:
            sys.stderr = self.std
        self.file.close()

    def write(self, data) -> None:
        try:
            self.file.write(data)
        except OSError:
            pass
        try:
            self.std.write(data)
        except OSError:
            pass

    def flush(self) -> None:
        try:
            self.file.flush()
        except OSError:
            pass
'''

class Tee:
    """Duplicate output to both stdout/err and file"""

    def __init__(self, file, err: bool = False) -> None:
        self.err = err
        self.std = sys.stderr if err else sys.stdout

        if isinstance(file, str):
            try:
                # Ensure the directory exists if file is a path
                os.makedirs(os.path.dirname(file), exist_ok=True)
                self.file = open(file, "w")
            except IOError as e:
                print(f"Warning: Could not open file {file} for writing. {str(e)}", file=self.std)
                self.file = None
        elif hasattr(file, 'write') and hasattr(file, 'flush'):
            # If it's a file-like object, use it directly
            self.file = file
        else:
            raise ValueError("'file' must be either a file path or a file-like object")

        if not err:
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self) -> None:
        if not self.err:
            sys.stdout = self.std
        else:
            sys.stderr = self.std
        
        if self.file and hasattr(self.file, 'close'):
            self.file.close()

    def write(self, data) -> None:
        self.std.write(data)
        if self.file:
            try:
                self.file.write(data)
            except IOError as e:
                print(f"Warning: Could not write to file. {str(e)}", file=self.std)

    def flush(self) -> None:
        self.std.flush()
        if self.file:
            try:
                self.file.flush()
            except IOError as e:
                print(f"Warning: Could not flush file. {str(e)}", file=self.std)


def human_readable_flops(num) -> str:
    for unit in [
        "",
        "KFLOPS",
        "MFLOPS",
        "GFLOPS",
        "TFLOPS",
        "PFLOPS",
        "EFLOPS",
        "ZFLOPS",
    ]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")


def get_actual_flops(neox_args, iter_time_s) -> float:
    """
    This function finds the actual FLOPs achieved accounting for implementation and hardware details. Also used for HFU.

    For more detail on flop calculations, see https://github.com/EleutherAI/cookbook/tree/main/calc and https://github.com/Zyphra/zcookbook/tree/main/calc

    Use FLOPS calculation from Megatron-DeepSpeed:
    https://github.com/microsoft/Megatron-DeepSpeed/blob/cc3a94c636789f74be2bc6cfc62a3d723fd5d749/megatron/utils.py#L253
    They get it from https://arxiv.org/pdf/2104.04473.pdf
    """
    world_size = torch.distributed.get_world_size()
    vocab_size = neox_args.padded_vocab_size
    batch_size = neox_args.train_batch_size
    seq_len = neox_args.seq_length
    hidden_size = neox_args.hidden_size
    num_layers = neox_args.num_layers
    ckpt_activations_factor = 4 if neox_args.checkpoint_activations else 3
    if "rwkv" in neox_args.attention_config:
        num_heads = neox_args.num_attention_heads

        flops_per_iteration = (
            batch_size
            * seq_len
            * (
                78 * hidden_size * hidden_size * num_layers
                + 84 * hidden_size * num_layers
                + 16 * hidden_size
                + 12 * hidden_size * vocab_size
                + 18 * hidden_size * hidden_size * num_layers / num_heads
            )
        )
    elif "mamba" in neox_args.attention_config:
        # from https://github.com/Zyphra/zcookbook/blob/main/calc/calc_mamba_flops.py
        if neox_args.expansion_factor:
            d_inner = neox_args.hidden_size * neox_args.expansion_factor
        elif neox_args.intermediate_size:
            d_inner = neox_args.intermediate_size
        else:
            d_inner = neox_args.hidden_size * 2  # default expansion factor
        d_state = 16  # TODO make d_state an arg. Currently hardcoded in neox mamba definition and here
        conv_dimension = 4  # TODO make conv_dimension an arg. Currently hardcoded in neox mamba definition and here
        dt_rank = math.ceil(neox_args.hidden_size / 16)
        ssm_flops = (
            ckpt_activations_factor
            * d_inner
            * seq_len
            * batch_size
            * (11 * d_state + 4 * dt_rank + 1)
        )
        mamba_projectors_flops = (
            ckpt_activations_factor * seq_len * batch_size * 6 * d_inner * hidden_size
        )
        mamba_conv_flops = (
            ckpt_activations_factor
            * seq_len
            * batch_size
            * 2
            * d_inner
            * conv_dimension
        )
        mamba_flops = ssm_flops + mamba_projectors_flops + mamba_conv_flops
        embedding_flops = 6 * seq_len * batch_size * hidden_size * vocab_size
        flops_per_iteration = mamba_flops * num_layers + embedding_flops
    else:
        flops_per_iteration = (
            24
            * ckpt_activations_factor
            * batch_size
            * seq_len
            * num_layers
            * (hidden_size**2)
            * (
                1.0
                + (seq_len / (6.0 * hidden_size))
                + (vocab_size / (16.0 * num_layers * hidden_size))
            )
        )
    return flops_per_iteration / (iter_time_s * world_size)


def get_forward_backward_flops(neox_args, iter_time_s) -> float:
    """
    This function finds the estimated FLOPs required by a single forward+backward pass without accounting for implementation and hardware details. Also used for MFU.

    Mostly duplicated from get_actual_flops with just a change in activation checkpointing for now, but these may diverge over time as implementation details accumulate so I think 2 separate functions are appropriate.

    For more detail on flop calculations, see https://github.com/EleutherAI/cookbook/tree/main/calc and https://github.com/Zyphra/zcookbook/tree/main/calc

    Use FLOPS calculation from Megatron-DeepSpeed:
    https://github.com/microsoft/Megatron-DeepSpeed/blob/cc3a94c636789f74be2bc6cfc62a3d723fd5d749/megatron/utils.py#L253
    They get it from https://arxiv.org/pdf/2104.04473.pdf
    """
    world_size = torch.distributed.get_world_size()
    vocab_size = neox_args.padded_vocab_size
    batch_size = neox_args.train_batch_size
    seq_len = neox_args.seq_length
    hidden_size = neox_args.hidden_size
    num_layers = neox_args.num_layers
    fwd_bwd_factor = 3  # 1 for fwd, 2 for bwd and weight update
    if "rwkv" in neox_args.attention_config:
        num_heads = neox_args.num_attention_heads

        flops_per_iteration = (
            batch_size
            * seq_len
            * (
                78 * hidden_size * hidden_size * num_layers
                + 84 * hidden_size * num_layers
                + 16 * hidden_size
                + 12 * hidden_size * vocab_size
                + 18 * hidden_size * hidden_size * num_layers / num_heads
            )
        )
    elif "mamba" in neox_args.attention_config:
        # from https://github.com/Zyphra/zcookbook/blob/main/calc/calc_mamba_flops.py
        if neox_args.expansion_factor:
            d_inner = neox_args.hidden_size * neox_args.expansion_factor
        elif neox_args.intermediate_size:
            d_inner = neox_args.intermediate_size
        else:
            d_inner = neox_args.hidden_size * 2  # default expansion factor
        d_state = 16  # TODO make d_state an arg. Currently hardcoded in neox mamba definition and here
        conv_dimension = 4  # TODO make conv_dimension an arg. Currently hardcoded in neox mamba definition and here
        dt_rank = math.ceil(neox_args.hidden_size / 16)
        ssm_flops = (
            fwd_bwd_factor
            * d_inner
            * seq_len
            * batch_size
            * (11 * d_state + 4 * dt_rank + 1)
        )
        mamba_projectors_flops = (
            fwd_bwd_factor * seq_len * batch_size * 6 * d_inner * hidden_size
        )
        mamba_conv_flops = (
            fwd_bwd_factor * seq_len * batch_size * 2 * d_inner * conv_dimension
        )
        mamba_flops = ssm_flops + mamba_projectors_flops + mamba_conv_flops
        embedding_flops = 6 * seq_len * batch_size * hidden_size * vocab_size
        flops_per_iteration = mamba_flops * num_layers + embedding_flops
    else:
        flops_per_iteration = (
            24
            * fwd_bwd_factor
            * batch_size
            * seq_len
            * num_layers
            * (hidden_size**2)
            * (
                1.0
                + (seq_len / (6.0 * hidden_size))
                + (vocab_size / (16.0 * num_layers * hidden_size))
            )
        )
    return flops_per_iteration / (iter_time_s * world_size)


def training_log(
    neox_args,
    timers,
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    model,
    optimizer,
    noise_scale_logger,
):
    """Log training information such as losses, timing, etc."""

    # Update losses.
    skipped_iters_key = "skipped iterations"
    total_loss_dict[skipped_iters_key] = (
        total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    )
    got_nan_key = "got nan"

    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(key, 0.0) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan

    total_loss_dict[got_nan_key] = total_loss_dict.get(got_nan_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)

    if not neox_args.is_pipe_parallel:
        add_to_logging("forward")
        add_to_logging("backward")
        add_to_logging("backward-backward")
        add_to_logging("backward-allreduce")
        add_to_logging("backward-master-grad")
        add_to_logging("backward-clip-grad")
        add_to_logging("optimizer")
        add_to_logging("batch generator")

        # Log timer info to tensorboard and wandb
        normalizer = iteration % neox_args.log_interval
        if normalizer == 0:
            normalizer = neox_args.log_interval
        if torch.distributed.get_rank() == 0:
            timers.write(
                names=timers_to_log, iteration=iteration, normalizer=normalizer
            )
    else:
        # with pipeline parallel, the megatron timers are overridden by the deepspeed ones.
        # Try to grab timer values from model engine. Only recently added to deeperspeed, so check that the engine
        # has that attribute first
        if hasattr(model, "timer_values") and model.timer_values is not None:
            if (
                model.wall_clock_breakdown()
                and model.global_steps % model.steps_per_print() == 0
            ):
                timer_values = model.timer_values
                # deepspeed already logs to tensorboard / prints values, so just log to wandb
                if neox_args.use_wandb and torch.distributed.get_rank() == 0:
                    for key in timer_values:
                        tb_wandb_log(
                            f"timers/{key}",
                            timer_values[key],
                            iteration,
                            use_wandb=neox_args.use_wandb,
                            tensorboard_writer=neox_args.tensorboard_writer,
                            comet_experiment=neox_args.comet_experiment,
                        )

    # write losses, lr, etc. every step
    tb_wandb_log(
        "train/learning_rate",
        learning_rate,
        iteration,
        use_wandb=neox_args.use_wandb,
        tensorboard_writer=neox_args.tensorboard_writer,
        comet_experiment=neox_args.comet_experiment,
    )
    for key in loss_dict:
        tb_wandb_log(
            f'train/{key.replace(" ", "_")}',
            loss_dict[key],
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
            comet_experiment=neox_args.comet_experiment,
        )
    if neox_args.fp16:
        tb_wandb_log(
            f"train/loss_scale",
            loss_scale,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
            comet_experiment=neox_args.comet_experiment,
        )

    # log gradient noise scale
    if neox_args.log_gradient_noise_scale:
        if noise_scale_logger.noise_scale is not None:
            tb_wandb_log(
                f"train/noise_scale",
                noise_scale_logger.noise_scale,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
                comet_experiment=neox_args.comet_experiment,
            )

    # (optional) Log optimizer states to wandb / tb every step
    if neox_args.log_optimizer_states:
        for k, v in optimizer.state_dict()["optimizer_state_dict"]["state"].items():
            for ki, vi in v.items():  # step, module
                if ki != "step":
                    opt_state_norm = torch.norm(vi) if hasattr(vi, "dim") else vi
                    tb_wandb_log(
                        f"optimizer_state_norms/{k}_{ki}",
                        opt_state_norm,
                        iteration,
                        use_wandb=neox_args.use_wandb,
                        tensorboard_writer=neox_args.tensorboard_writer,
                        comet_experiment=neox_args.comet_experiment,
                    )

    # (optional) Log grad/param norms to wandb / tb every step
    if (
        neox_args.log_grad_pct_zeros
        or neox_args.log_grad_norm
        or neox_args.log_param_norm
    ):
        if neox_args.log_grad_pct_zeros or neox_args.log_grad_norm:
            model.store_gradients = True  # start storing gradients

        for i, (name, param) in enumerate(model.module.named_parameters()):
            if neox_args.log_grad_pct_zeros:
                if (
                    hasattr(model, "stored_gradients")
                    and model.stored_gradients is not None
                ):
                    grad = model.stored_gradients[i]
                    if grad is not None:
                        tb_wandb_log(
                            f"pct_grad_zeros/{name}",
                            (grad == 0).float().mean().item() * 100,
                            iteration,
                            use_wandb=neox_args.use_wandb,
                            tensorboard_writer=neox_args.tensorboard_writer,
                            comet_experiment=neox_args.comet_experiment,
                            all_ranks=True,
                        )
            if neox_args.log_grad_norm:
                if (
                    hasattr(model, "stored_gradients")
                    and model.stored_gradients is not None
                ):
                    grad = model.stored_gradients[i]
                    if grad is not None:
                        tb_wandb_log(
                            f"gradient_norms/{name}",
                            torch.norm(grad),
                            iteration,
                            use_wandb=neox_args.use_wandb,
                            tensorboard_writer=neox_args.tensorboard_writer,
                            comet_experiment=neox_args.comet_experiment,
                            all_ranks=True,
                        )
            if neox_args.log_param_norm:
                tb_wandb_log(
                    f"parameter_norms/{name}",
                    torch.norm(param),
                    iteration,
                    use_wandb=neox_args.use_wandb,
                    tensorboard_writer=neox_args.tensorboard_writer,
                    comet_experiment=neox_args.comet_experiment,
                    all_ranks=True,
                )

    if iteration % neox_args.log_interval == 0:
        # log other stuff every neox_args.log_interval iters
        elapsed_time = timers("interval time").elapsed()
        iteration_time = elapsed_time / neox_args.log_interval
        samples_per_sec = neox_args.train_batch_size / iteration_time
        steps_per_sec = 1 / iteration_time
        tokens_per_sec = samples_per_sec * neox_args.seq_length
        log_string = " samples/sec: {:.3f} |".format(samples_per_sec)
        tb_wandb_log(
            "runtime/samples_per_sec",
            samples_per_sec,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
            comet_experiment=neox_args.comet_experiment,
        )
        tb_wandb_log(
            "runtime/iteration_time",
            iteration_time,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
            comet_experiment=neox_args.comet_experiment,
        )
        tb_wandb_log(
            "runtime/steps_per_sec",
            steps_per_sec,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
            comet_experiment=neox_args.comet_experiment,
        )
        tb_wandb_log(
            "runtime/tokens_per_sec",
            tokens_per_sec,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
            comet_experiment=neox_args.comet_experiment,
        )
        log_string += " iteration {:8d}/{:8d} |".format(
            iteration, neox_args.train_iters
        )
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(
            elapsed_time * 1000.0 / neox_args.log_interval
        )
        log_string += " learning rate: {:.3E} |".format(learning_rate)
        num_iterations = max(
            1, neox_args.log_interval - total_loss_dict[skipped_iters_key]
        )

        # log curriculum learning
        if neox_args.curriculum_learning:
            tb_wandb_log(
                "curriculum_seqlen",
                neox_args.curriculum_seqlen,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
                comet_experiment=neox_args.comet_experiment,
            )

        # log tflop / gpu
        flops_per_s_per_gpu = get_actual_flops(neox_args, iteration_time)

        log_string += (
            f" approx flops per GPU: {human_readable_flops(flops_per_s_per_gpu)} |"
        )
        tb_wandb_log(
            "runtime/flops_per_sec_per_gpu",
            flops_per_s_per_gpu,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
            comet_experiment=neox_args.comet_experiment,
        )

        if neox_args.peak_theoretical_tflops:
            # Convert peak theoretical TFLOPS to FLOPS for consistent units
            peak_theoretical_flops = neox_args.peak_theoretical_tflops * (10**12)

            # Calculate MFU and HFU as percentages
            mfu = (
                get_forward_backward_flops(neox_args, iteration_time)
                / peak_theoretical_flops
            ) * 100
            hfu = (flops_per_s_per_gpu / peak_theoretical_flops) * 100

            # Add to log string
            log_string += f" MFU: {mfu:.2f}% | HFU: {hfu:.2f}% |"

            # Log to tracking systems
            tb_wandb_log(
                "runtime/model_flops_utilization",
                mfu,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
                comet_experiment=neox_args.comet_experiment,
            )

            tb_wandb_log(
                "runtime/hardware_flops_utilization",
                hfu,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
                comet_experiment=neox_args.comet_experiment,
            )

        for key in total_loss_dict:
            if key not in [skipped_iters_key, got_nan_key]:
                v = (
                    total_loss_dict[key].item()
                    if hasattr(total_loss_dict[key], "item")
                    else total_loss_dict[key]
                )
                avg = v / float(num_iterations)
                log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = 0.0
        if neox_args.precision == "fp16":
            log_string += " loss scale: {:.1f} |".format(loss_scale)
        log_string += " number of skipped iterations: {:3d} |".format(
            total_loss_dict[skipped_iters_key]
        )
        log_string += " number of nan iterations: {:3d} |".format(
            total_loss_dict[got_nan_key]
        )
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[got_nan_key] = 0
        print_rank_0(log_string)
        if report_memory_flag:
            report_memory("after {} iterations".format(iteration))
            report_memory_flag = False

        timers.log(timers_to_log, normalizer=neox_args.log_interval)

    return report_memory_flag


def tb_wandb_log(
    key: str,
    value: float,
    iteration_no: int,
    use_wandb: bool,
    tensorboard_writer=None,
    comet_experiment=None,
    all_ranks: bool = False,
):
    # logs to both tb and wandb (if present) from the zeroth rank
    do_log = torch.distributed.get_rank() == 0 or all_ranks
    if do_log and value is not None:
        if tensorboard_writer:
            tensorboard_writer.add_scalar(key, value, iteration_no)
        if use_wandb:
            wandb.log({key: value}, step=iteration_no)
        if comet_experiment:
            comet_experiment.__internal_api__log_metric__(
                key, value, framework="gpt-neox", step=iteration_no
            )
